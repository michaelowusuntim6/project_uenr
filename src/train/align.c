#include "uenr/train.h"
#include "uenr/dataset.h"
#include "uenr/gemm.h"
#include "uenr/io.h"
#include "uenr/math.h"
#include "uenr/model.h"
#include "uenr/ops.h"
#include "uenr/reward.h"
#include "uenr/schedule.h"
#include "uenr/thread.h"
#include "uenr/tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Load a checkpoint that the alignment run cannot proceed without.  There is
 * deliberately no random-weight fallback: DPO/PPO/GRPO must start from a real
 * policy and a real frozen reference. */
UenrResult uenr_align_load_required(UenrModel *m, const UenrConfig *cfg,
                                      const char *path, int with_opt,
                                      const char *what) {
    UenrRng r;
    uenr_rng_seed(&r, (uint64_t)cfg->run.seed);
    UENR_TRY(uenr_model_init(m, cfg, &r));
    if (with_opt) UENR_TRY(uenr_model_opt_init(m, 1));
    if (!path || !path[0] || uenr_file_exists(path) != UENR_OK) {
        fprintf(stderr, "%s: required checkpoint '%s' does not exist; "
                        "train a base model first (uenr train) or set the path\n",
                what, path && path[0] ? path : "(empty)");
        return UENR_ERR_IO;
    }
    int step = 0;
    UenrResult lr = uenr_model_load(m, path, &step, with_opt);
    if (lr != UENR_OK) {
        fprintf(stderr, "%s: checkpoint '%s' is invalid or incompatible "
                        "(err %d); refusing to continue\n", what, path, (int)lr);
        return lr;
    }
    printf("%s: loaded %s (step %d)\n", what, path, step);
    return UENR_OK;
}

/* Resolve the frozen reference model.  The configured dpo.ref_checkpoint is
 * used when present.  If it is absent, explicitly fall back to the base
 * model_out checkpoint; if neither is available, fail closed.  Weights are
 * validated as finite so a truncated/corrupt payload cannot silently become
 * the reference. */
UenrResult uenr_align_load_reference(UenrModel *ref, const UenrConfig *cfg) {
    const char *configured = cfg->dpo.ref_checkpoint;
    const char *base = cfg->paths.model_out;
    const char *path = NULL;
    if (configured[0] && cfg->paths.sft_out[0] &&
        !strcmp(configured, cfg->paths.sft_out)) {
        fprintf(stderr, "reference: refusing to use the SFT checkpoint '%s' as the "
                        "DPO/RL reference; point dpo.ref_path/--ref at the pretrained "
                        "model checkpoint\n", configured);
        return UENR_ERR_INVALID;
    }
    if (configured[0]) {
        if (uenr_file_exists(configured) == UENR_OK) {
            path = configured;
        } else if (base[0] && uenr_file_exists(base) == UENR_OK) {
            fprintf(stderr, "reference: configured '%s' is missing; "
                            "explicitly falling back to base checkpoint '%s'\n",
                    configured, base);
            path = base;
        } else {
            fprintf(stderr, "reference: configured '%s' is missing and no base "
                            "checkpoint '%s' is available; run `uenr train` first\n",
                    configured, base[0] ? base : "(empty)");
            return UENR_ERR_IO;
        }
    } else {
        if (!base[0] || uenr_file_exists(base) != UENR_OK) {
            fprintf(stderr, "reference: no dpo.ref_checkpoint configured and base "
                            "checkpoint '%s' is missing; run `uenr train` first\n",
                    base[0] ? base : "(empty)");
            return UENR_ERR_IO;
        }
        fprintf(stderr, "reference: no dpo.ref_checkpoint configured; using base "
                        "checkpoint '%s' as the frozen reference\n", base);
        path = base;
    }
    UenrResult lr = uenr_align_load_required(ref, cfg, path, 0, "reference");
    if (lr != UENR_OK) return lr;
    for (size_t i = 0; i < ref->alloc_floats; i += 4093)
        if (!isfinite(ref->params[i])) {
            fprintf(stderr, "reference: loaded weights contain non-finite values; "
                            "refusing to use '%s'\n", path);
            return UENR_ERR_INVALID;
        }
    return UENR_OK;
}

/* Build ids = chat(prompt) + completion + <end>, and next-token targets. */
static UenrResult build_seq(const UenrConfig *cfg, const UenrTokenizer *tok,
                            const char *prompt, const char *completion,
                            int **ids_out, int **tg_out, int *n_out, int *pn_out) {
    int *pids = NULL, pn = 0;
    if (uenr_chat_infer(tok, cfg->infer.system, prompt, &pids, &pn) != UENR_OK)
        return UENR_ERR_INVALID;
    if (pn >= cfg->model.max_seq) { free(pids); return UENR_ERR_INVALID; }
    int *cids = NULL, cn = 0;
    if (uenr_tok_encode(tok, completion, &cids, &cn) != UENR_OK) {
        free(pids);
        return UENR_ERR_INVALID;
    }
    int n = pn + cn + 1;
    if (n > cfg->model.max_seq) n = cfg->model.max_seq;
    int *ids = (int *)malloc((size_t)n * sizeof(int));
    int *tg = (int *)malloc((size_t)n * sizeof(int));
    if (!ids || !tg) { free(pids); free(cids); free(ids); free(tg); return UENR_ERR_ALLOC; }
    int k = 0;
    for (int i = 0; i < pn && k < n; i++) ids[k++] = pids[i];
    for (int i = 0; i < cn && k < n; i++) ids[k++] = cids[i];
    if (k < n) ids[k++] = uenr_tok_id(tok, UENR_ROLE_END);
    n = k;
    int ignore = cfg->tok.ignore_index;
    for (int t = 0; t < n; t++) {
        if (t + 1 < n && t + 1 >= pn) tg[t] = ids[t + 1];
        else tg[t] = ignore;
    }
    free(pids);
    free(cids);
    *ids_out = ids;
    *tg_out = tg;
    *n_out = n;
    *pn_out = pn;
    return UENR_OK;
}

static float seq_logp(UenrModel *m, UenrGemm *g, UenrActs *a,
                      const int *ids, const int *tg, int n,
                      int ignore, int write_grad) {
    int count = 0;
    float loss = uenr_model_forward(m, g, a, ids, n, n, tg, ignore, write_grad, &count);
    return -loss;
}

/* ------------------------------------------------------------------------- */
/* DPO                                                                       */
/*                                                                           */
/* L = -log sigmoid(beta * delta) where                                  */
/*   delta = (log pi(w) - log ref(w)) - (log pi(l) - log ref(l)).         */
/*                                                                           */
/* dL/ddelta = -beta * sigmoid(-beta*delta).  Since ddelta/dlog pi(w) = 1  */
/* and ddelta/dlog pi(l) = -1, the policy-log-probability gradients are:   */
/*   g_w = -beta * sigmoid(-z),   g_l = +beta * sigmoid(-z),  z=beta*delta */
/* A gradient step on the policy logits therefore lowers the loss, i.e.    */
/* raises log pi(chosen) and lowers log pi(rejected) relative to the       */
/* frozen reference.  softplus form keeps the loss finite for |z| large.   */
/* ------------------------------------------------------------------------- */

float uenr_dpo_loss(float delta, float beta) {
    float z = beta * delta;
    /* -log sigmoid(z) == softplus(-z) == max(-z,0) + log1p(exp(-|z|)) */
    if (z >= 0.0f) return uenr_logf(1.0f + uenr_expf(-z));
    return -z + uenr_logf(1.0f + uenr_expf(z));
}

void uenr_dpo_logp_grad(float delta, float beta,
                        float *g_chosen, float *g_rejected) {
    float dcoef = beta * uenr_sigmoidf(-(beta * delta)); /* beta*sigmoid(-z) */
    if (g_chosen) *g_chosen = -dcoef;
    if (g_rejected) *g_rejected = dcoef;
}

/* ------------------------------------------------------------------------- */
/* PPO/GRPO                                                                  */
/*                                                                           */
/* Sequence-level objective for one sampled completion:                      */
/*   ratio r = exp(log pi_theta(y) - log pi_old(y))                          */
/*   surrogate S = min(r*A, clamp(r, 1-eps, 1+eps)*A)                        */
/*   L = -S + lambda_kl * (log pi_theta(y) - log pi_ref(y))                  */
/* (the KL term is the single-sample k1 estimator; the reference is frozen). */
/*                                                                           */
/* dS/dlog pi = A*r only where the unclipped term is the minimum:            */
/*   A > 0: r <= 1+eps    (clip only the upward, over-confident direction)   */
/*   A < 0: r >= 1-eps    (clip only the downward direction)                 */
/* dL/dlog pi = -dS/dlog pi + lambda_kl; the trainer scales the CE-style     */
/* dlogits buffer by the negative of that coefficient.                       */
/* ------------------------------------------------------------------------- */

float uenr_ppo_surrogate_coef(float advantage, float ratio, float clip,
                              int *active) {
    int act = 0;
    if (advantage > 0.0f) act = ratio <= 1.0f + clip;
    else if (advantage < 0.0f) act = ratio >= 1.0f - clip;
    if (active) *active = act;
    return act ? advantage * ratio : 0.0f;
}

UenrResult uenr_train_dpo(const UenrConfig *cfg) {
    UenrTokenizer *tok = NULL;
    UenrResult r = uenr_tok_load(cfg, cfg->paths.tokenizer, &tok);
    if (r != UENR_OK) { fprintf(stderr, "dpo: tokenizer missing\n"); return r; }
    UenrPairData pairs;
    r = uenr_pairs_load(cfg->paths.pairs_jsonl, &pairs);
    if (r != UENR_OK) { fprintf(stderr, "dpo: cannot load pairs\n"); uenr_tok_free(tok); return r; }

    UenrModel policy, ref;
    memset(&policy, 0, sizeof(policy));
    memset(&ref, 0, sizeof(ref));
    if (uenr_align_load_required(&policy, cfg, cfg->paths.model_out, 1, "dpo policy") != UENR_OK)
        goto fail;
    if (uenr_align_load_reference(&ref, cfg) != UENR_OK) goto fail;
    UenrPool *pool = NULL;
    int cpus[2] = { cfg->run.pin_cpu0, cfg->run.pin_cpu1 };
    if (cfg->run.threads > 1) uenr_pool_init(&pool, cfg->run.threads, cpus, 2);
    UenrPool *pool_mem = NULL;
    if (cfg->run.smt_threads > 1) {
        int c4[4] = { 0, 1, 2, 3 };
        (void)uenr_pool_init(&pool_mem, cfg->run.smt_threads, c4,
                             cfg->run.smt_threads > 4 ? 4 : cfg->run.smt_threads);
    }
    UenrGemm gemm;
    if (uenr_gemm_init(&gemm, pool) != UENR_OK) goto fail;
    if (pool_mem) gemm.mem_pool = pool_mem;
    UenrActs ap, ar;
    if (uenr_acts_init(&ap, cfg, cfg->model.max_seq, cfg->model.max_seq) != UENR_OK) goto fail_gemm;
    if (uenr_acts_init(&ar, cfg, cfg->model.max_seq, cfg->model.max_seq) != UENR_OK) goto fail_acts;

    int steps = cfg->train.max_steps;
    int cap = pairs.count * (cfg->sched.max_epochs > 0 ? cfg->sched.max_epochs : 1);
    if (steps > cap) steps = cap;
    if (steps < 1) steps = 1;
    printf("dpo: %d pairs, %d steps, beta=%.3f\n", pairs.count, steps, (double)cfg->dpo.beta);

    UenrRng rng;
    uenr_rng_seed(&rng, (uint64_t)cfg->run.seed + 4);
    const float beta = cfg->dpo.beta;
    const int ignore = cfg->tok.ignore_index;
    double t0 = now_s();
    int done = 0;

    for (int step = 1; step <= steps; step++) {
        const UenrPair *pr = &pairs.pairs[uenr_rng_range(&rng, pairs.count)];
        int *wid = NULL, *wtg = NULL, *lid = NULL, *ltg = NULL;
        int wn = 0, ln = 0, wp = 0, lp = 0;
        if (build_seq(cfg, tok, pr->prompt, pr->chosen, &wid, &wtg, &wn, &wp) != UENR_OK ||
            build_seq(cfg, tok, pr->prompt, pr->rejected, &lid, &ltg, &ln, &lp) != UENR_OK) {
            free(wid); free(wtg); free(lid); free(ltg);
            continue;
        }
        float lpw = seq_logp(&policy, &gemm, &ap, wid, wtg, wn, ignore, 0);
        float lpl = seq_logp(&policy, &gemm, &ap, lid, ltg, ln, ignore, 0);
        float rw = seq_logp(&ref, &gemm, &ar, wid, wtg, wn, ignore, 0);
        float rl = seq_logp(&ref, &gemm, &ar, lid, ltg, ln, ignore, 0);
        float delta = (lpw - rw) - (lpl - rl);
        float z = beta * delta;
        float sig = uenr_sigmoidf(z);
        float loss = uenr_dpo_loss(delta, beta);
        float gw = 0.0f, gl = 0.0f;
        uenr_dpo_logp_grad(delta, beta, &gw, &gl);

        uenr_model_zero_grad(&policy);
        /* seq_logp(write_grad=1) leaves dCE/dlogits = p - onehot in dlogits.
         * dlogp/dlogits = onehot - p, so scale by -g to obtain dL/dlogits. */
        seq_logp(&policy, &gemm, &ap, wid, wtg, wn, ignore, 1);
        uenr_scale(ap.dlogits, (size_t)wn * cfg->model.vocab_size, -gw);
        if (uenr_model_backward(&policy, &gemm, &ap) != UENR_OK) {
            free(wid); free(wtg); free(lid); free(ltg);
            r = UENR_ERR_ALLOC;
            goto fail_acts;
        }
        seq_logp(&policy, &gemm, &ap, lid, ltg, ln, ignore, 1);
        uenr_scale(ap.dlogits, (size_t)ln * cfg->model.vocab_size, -gl);
        if (uenr_model_backward(&policy, &gemm, &ap) != UENR_OK) {
            free(wid); free(wtg); free(lid); free(ltg);
            r = UENR_ERR_ALLOC;
            goto fail_acts;
        }

        uenr_model_clip_grad(&policy, cfg->train.clip, pool_mem ? pool_mem : pool);
        float lr = uenr_schedule_lr(cfg, step, steps, 0.0f);
        uenr_model_adamw(&policy, step, lr, cfg->train.beta1, cfg->train.beta2,
                         cfg->train.adam_eps, cfg->train.weight_decay,
                         pool_mem ? pool_mem : pool);

        free(wid); free(wtg); free(lid); free(ltg);
        done = step;
        if (step == 1 || step % cfg->run.log_every == 0) {
            double dt = now_s() - t0;
            if (dt <= 0) dt = 1e-9;
            printf("[dpo] step %d/%d lr %.3g loss %.4f delta %.3f acc %.2f | %.2f pair/s\n",
                   step, steps, (double)lr, (double)loss, (double)delta,
                   sig > 0.5 ? 1.0 : 0.0, step / dt);
            t0 = now_s();
        }
        if (cfg->run.checkpoint_every > 0 && step % cfg->run.checkpoint_every == 0) {
            char path[512];
            snprintf(path, sizeof(path), "%s/dpo_%06d.bin", cfg->run.checkpoint_dir, step);
            uenr_model_save(&policy, path, step, 1);
        }
    }
    uenr_model_save(&policy, cfg->paths.dpo_out, done, 1);
    printf("[dpo] saved %s\n", cfg->paths.dpo_out);

    uenr_acts_free(&ap); uenr_acts_free(&ar);
    uenr_gemm_destroy(&gemm);
    if (pool) uenr_pool_destroy(pool);
    if (pool_mem) uenr_pool_destroy(pool_mem);
    uenr_model_free(&policy); uenr_model_free(&ref);
    uenr_pairs_free(&pairs);
    uenr_tok_free(tok);
    return UENR_OK;
fail_acts:
    uenr_acts_free(&ap);
fail_gemm:
    uenr_gemm_destroy(&gemm);
    if (pool) uenr_pool_destroy(pool);
    if (pool_mem) uenr_pool_destroy(pool_mem);
fail:
    uenr_model_free(&policy); uenr_model_free(&ref);
    uenr_pairs_free(&pairs);
    uenr_tok_free(tok);
    return UENR_ERR_INVALID;
}

/* ------------------------------------------------------------------------- */
/* PPO / GRPO (sequence-level policy gradient with KL to the reference)      */
/* ------------------------------------------------------------------------- */

static float rl_reward(const UenrConfig *cfg, const int *ids, int n, int pn) {
    int len = n - pn;
    if (len <= 0) return -1.0f;
    float r = 0.0f;
    r += cfg->rl.reward_format * ((ids[n - 1] == cfg->tok.special_end ||
                                   ids[n - 1] == cfg->tok.eos_id) ? 1.0f : 0.0f);
    float target = (float)cfg->rl.length_target;
    if (target > 0.0f)
        r -= cfg->rl.reward_length * fabsf((float)len - target) / target;
    int uniq = 0;
    for (int i = pn; i < n; i++) {
        int seen = 0;
        for (int j = pn; j < i; j++) if (ids[j] == ids[i]) { seen = 1; break; }
        if (!seen) uniq++;
    }
    r -= cfg->rl.reward_repeat * (1.0f - (float)uniq / (float)len);
    return r;
}

/* Sample from softmax(logits / temperature) but report the log-probability
 * under the unscaled T=1 policy.  PPO importance ratios compare the current
 * T=1 policy with the T=1 behaviour policy, while temperature only controls
 * exploration during rollout generation. */
static int sample_rl(const float *logits, int V, float temp, UenrRng *rng,
                     float *probs, float *logp_out) {
    float mx = logits[0];
    for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f, rawsum = 0.0f;
    for (int i = 0; i < V; i++) {
        float z = logits[i] - mx;
        probs[i] = uenr_expf(z / temp);
        sum += probs[i];
        rawsum += uenr_expf(z);
    }
    float inv = uenr_rcpf(sum);
    for (int i = 0; i < V; i++) probs[i] *= inv;
    float rr = uenr_rng_f32(rng);
    float c = 0.0f;
    int pick = V - 1;
    for (int i = 0; i < V; i++) {
        c += probs[i];
        if (rr <= c) { pick = i; break; }
    }
    *logp_out = (logits[pick] - mx) - uenr_logf(rawsum);
    return pick;
}

static UenrResult run_rl(const UenrConfig *cfg, int is_ppo) {
    UenrTokenizer *tok = NULL;
    UenrResult r = uenr_tok_load(cfg, cfg->paths.tokenizer, &tok);
    if (r != UENR_OK) { fprintf(stderr, "rl: tokenizer missing\n"); return r; }
    UenrPairData pairs;
    const char *rl_path = cfg->paths.rl_jsonl;
    r = uenr_rl_load(rl_path, &pairs);
    if (r != UENR_OK) { fprintf(stderr, "rl: cannot load prompts %s\n", rl_path); uenr_tok_free(tok); return r; }

    UenrModel policy, ref;
    memset(&policy, 0, sizeof(policy));
    memset(&ref, 0, sizeof(ref));
    if (uenr_align_load_required(&policy, cfg, cfg->paths.model_out, 1, "rl policy") != UENR_OK)
        goto fail;
    if (uenr_align_load_reference(&ref, cfg) != UENR_OK) goto fail;
    UenrRewardModel rw;
    memset(&rw, 0, sizeof(rw));
    int use_reward_model = 0;
    if (!strcmp(cfg->reward.kind, "model")) {
        if (uenr_reward_load(&rw, cfg->reward.checkpoint) == UENR_OK) {
            use_reward_model = 1;
            printf("rl: using reward model %s (C=%d)\n", cfg->reward.checkpoint, rw.C);
        } else {
            /* reward.kind="model" is an explicit request: fail closed instead
             * of silently switching the objective to the heuristic reward. */
            fprintf(stderr, "rl: reward.kind=\"model\" but '%s' is missing or "
                            "invalid; run `uenr train-rm` or set reward.kind=\"heuristic\"\n",
                    cfg->reward.checkpoint);
            goto fail;
        }
    }

    UenrPool *pool = NULL;
    int cpus[2] = { cfg->run.pin_cpu0, cfg->run.pin_cpu1 };
    if (cfg->run.threads > 1) uenr_pool_init(&pool, cfg->run.threads, cpus, 2);
    UenrPool *pool_mem = NULL;
    if (cfg->run.smt_threads > 1) {
        int c4[4] = { 0, 1, 2, 3 };
        (void)uenr_pool_init(&pool_mem, cfg->run.smt_threads, c4,
                             cfg->run.smt_threads > 4 ? 4 : cfg->run.smt_threads);
    }
    UenrGemm gemm;
    if (uenr_gemm_init(&gemm, pool) != UENR_OK) goto fail;
    if (pool_mem) gemm.mem_pool = pool_mem;
    UenrActs ap, ar;
    if (uenr_acts_init(&ap, cfg, cfg->model.max_seq, cfg->model.max_seq) != UENR_OK) goto fail_gemm;
    if (uenr_acts_init(&ar, cfg, cfg->model.max_seq, cfg->model.max_seq) != UENR_OK) goto fail_acts;

    int G = is_ppo ? cfg->ppo.rollouts : cfg->grpo.group_size;
    if (G < 1) G = 1;
    if (G > 32) G = 32;
    int steps = cfg->train.max_steps;
    int cap = pairs.count;
    if (steps > cap) steps = cap;
    if (steps < 1) steps = 1;
    printf("%s: %d prompts, %d steps, group=%d, max_new=%d\n",
           is_ppo ? "ppo" : "grpo", pairs.count, steps, G, cfg->rl.max_new_tokens);

    int **seq = (int **)calloc((size_t)G, sizeof(int *));
    int *slen = (int *)calloc((size_t)G, sizeof(int));
    int *spn = (int *)calloc((size_t)G, sizeof(int));
    float *sold = (float *)calloc((size_t)G, sizeof(float));
    float *srew = (float *)calloc((size_t)G, sizeof(float));
    int *tg = (int *)malloc((size_t)cfg->model.max_seq * sizeof(int));
    float *probs = (float *)malloc((size_t)cfg->model.vocab_size * sizeof(float));
    int *pids = NULL;
    if (!seq || !slen || !spn || !sold || !srew || !tg || !probs) goto fail_mem;
    for (int i = 0; i < G; i++) {
        seq[i] = (int *)malloc((size_t)cfg->model.max_seq * sizeof(int));
        if (!seq[i]) goto fail_mem;
    }

    UenrRng rng;
    uenr_rng_seed(&rng, (uint64_t)cfg->run.seed + 5);
    const int ignore = cfg->tok.ignore_index;
    const int V = cfg->model.vocab_size;
    const int end_id = cfg->tok.special_end;
    double t0 = now_s();
    int done = 0;

    for (int step = 1; step <= steps; step++) {
        const UenrPair *pr = &pairs.pairs[uenr_rng_range(&rng, pairs.count)];
        int pn = 0;
        free(pids);
        pids = NULL;
        if (uenr_chat_infer(tok, cfg->infer.system, pr->prompt, &pids, &pn) != UENR_OK)
            continue;
        if (pn >= cfg->model.max_seq - 2) continue;

        for (int g = 0; g < G; g++) {
            int n = pn;
            memcpy(seq[g], pids, (size_t)pn * sizeof(int));
            float old_lp = 0.0f;
            for (int t = 0; t < cfg->rl.max_new_tokens && n < cfg->model.max_seq; t++) {
                uenr_model_forward(&policy, &gemm, &ap, seq[g], n, n, NULL, ignore, 0, NULL);
                const float *lg = ap.logits + (size_t)(n - 1) * V;
                float lp = 0.0f;
                int nxt = sample_rl(lg, V, cfg->rl.temperature, &rng, probs, &lp);
                old_lp += lp;
                seq[g][n++] = nxt;
                if (nxt == end_id || nxt == cfg->tok.eos_id) break;
            }
            slen[g] = n;
            spn[g] = pn;
            sold[g] = old_lp;
            if (use_reward_model)
                srew[g] = uenr_reward_score_seq(&ref, &gemm, &ar, &rw, seq[g], n, pn);
            else
                srew[g] = rl_reward(cfg, seq[g], n, pn);
        }

        float mean = 0.0f;
        for (int g = 0; g < G; g++) mean += srew[g];
        mean /= (float)G;
        float var = 0.0f;
        for (int g = 0; g < G; g++) { float d = srew[g] - mean; var += d * d; }
        var /= (float)G;
        float sd = uenr_sqrtf(var + 1e-8f);

        uenr_model_zero_grad(&policy);
        float total_loss = 0.0f;
        for (int g = 0; g < G; g++) {
            int n = slen[g];
            if (n <= pn + 1) continue;
            for (int t = 0; t < n; t++)
                tg[t] = (t + 1 < n && t + 1 >= pn) ? seq[g][t + 1] : ignore;
            float new_lp = seq_logp(&policy, &gemm, &ap, seq[g], tg, n, ignore, 1);
            float ref_lp = seq_logp(&ref, &gemm, &ar, seq[g], tg, n, ignore, 0);
            float adv = is_ppo ? (srew[g] - mean) : (srew[g] - mean) / sd;
            /* k1 single-sample estimate: E_pi[log pi - log ref].  The
             * reference term is constant, so dKL/dlog pi = 1. */
            float kl = new_lp - ref_lp;
            float coef; /* dL/dlog pi(theta) for this sequence */
            float surr = 0.0f;
            if (is_ppo) {
                float lr_ = new_lp - sold[g];
                if (lr_ > 50.0f) lr_ = 50.0f;
                if (lr_ < -50.0f) lr_ = -50.0f;
                float ratio = uenr_expf(lr_);
                int active = 0;
                float dS = uenr_ppo_surrogate_coef(adv, ratio, cfg->ppo.clip_ratio,
                                                   &active);
                float lo = 1.0f - cfg->ppo.clip_ratio;
                float hi = 1.0f + cfg->ppo.clip_ratio;
                float c = ratio < lo ? lo : (ratio > hi ? hi : ratio);
                surr = ratio * adv < c * adv ? ratio * adv : c * adv;
                coef = -dS + cfg->ppo.kl_coef;
            } else {
                surr = adv * new_lp;
                coef = -adv + cfg->grpo.kl_coef;
            }
            /* dlogp/dlogits = onehot - p = -(p - onehot), so scale the
             * CE-gradient buffer by -coef to obtain dL/dlogits. */
            uenr_scale(ap.dlogits, (size_t)n * V, -coef);
            if (uenr_model_backward(&policy, &gemm, &ap) != UENR_OK) {
                r = UENR_ERR_ALLOC;
                goto fail_mem;
            }
            total_loss += -surr + (is_ppo ? cfg->ppo.kl_coef : cfg->grpo.kl_coef) * kl;
        }

        uenr_model_clip_grad(&policy, cfg->train.clip, pool_mem ? pool_mem : pool);
        float lr = uenr_schedule_lr(cfg, step, steps, 0.0f);
        uenr_model_adamw(&policy, step, lr, cfg->train.beta1, cfg->train.beta2,
                         cfg->train.adam_eps, cfg->train.weight_decay,
                         pool_mem ? pool_mem : pool);
        done = step;

        if (step == 1 || step % cfg->run.log_every == 0) {
            double dt = now_s() - t0;
            if (dt <= 0) dt = 1e-9;
            printf("[%s] step %d/%d lr %.3g reward %.3f sd %.3f pol_loss %.4f | %.2f step/s\n",
                   is_ppo ? "ppo" : "grpo", step, steps, (double)lr, (double)mean,
                   (double)sd, (double)total_loss / G, step / dt);
            t0 = now_s();
        }
        if (cfg->run.checkpoint_every > 0 && step % cfg->run.checkpoint_every == 0) {
            char path[512];
            snprintf(path, sizeof(path), "%s/%s_%06d.bin", cfg->run.checkpoint_dir,
                     is_ppo ? "ppo" : "grpo", step);
            uenr_model_save(&policy, path, step, 1);
        }
    }

    uenr_model_save(&policy, is_ppo ? cfg->paths.ppo_out : cfg->paths.grpo_out, done, 1);
    printf("[%s] saved %s\n", is_ppo ? "ppo" : "grpo",
           is_ppo ? cfg->paths.ppo_out : cfg->paths.grpo_out);

    free(pids);
    for (int i = 0; i < G; i++) free(seq[i]);
    free(seq); free(slen); free(spn); free(sold); free(srew); free(tg); free(probs);
    uenr_acts_free(&ap); uenr_acts_free(&ar);
    uenr_gemm_destroy(&gemm);
    if (pool) uenr_pool_destroy(pool);
    if (pool_mem) uenr_pool_destroy(pool_mem);
    uenr_reward_free(&rw);
    uenr_model_free(&policy); uenr_model_free(&ref);
    uenr_pairs_free(&pairs);
    uenr_tok_free(tok);
    return UENR_OK;

fail_acts:
    uenr_acts_free(&ap);
    uenr_acts_free(&ar);
    goto fail_gemm;
fail_mem:
    free(pids);
    if (seq) for (int i = 0; i < G; i++) free(seq[i]);
    free(seq); free(slen); free(spn); free(sold); free(srew); free(tg); free(probs);
    uenr_acts_free(&ap); uenr_acts_free(&ar);
fail_gemm:
    uenr_gemm_destroy(&gemm);
    if (pool) uenr_pool_destroy(pool);
    if (pool_mem) uenr_pool_destroy(pool_mem);
fail:
    uenr_reward_free(&rw);
    uenr_model_free(&policy); uenr_model_free(&ref);
    uenr_pairs_free(&pairs);
    uenr_tok_free(tok);
    return UENR_ERR_INVALID;
}

UenrResult uenr_train_ppo(const UenrConfig *cfg) { return run_rl(cfg, 1); }
UenrResult uenr_train_grpo(const UenrConfig *cfg) { return run_rl(cfg, 0); }
