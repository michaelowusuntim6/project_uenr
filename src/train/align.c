#include "uenr/train.h"
#include "uenr/dataset.h"
#include "uenr/gemm.h"
#include "uenr/io.h"
#include "uenr/math.h"
#include "uenr/model.h"
#include "uenr/ops.h"
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

static UenrResult model_load_or_init(UenrModel *m, const UenrConfig *cfg,
                                     const char *path, int with_opt) {
    UenrRng r;
    uenr_rng_seed(&r, (uint64_t)cfg->run.seed);
    UENR_TRY(uenr_model_init(m, cfg, &r));
    if (with_opt) UENR_TRY(uenr_model_opt_init(m, 1));
    if (path && uenr_file_exists(path) == UENR_OK) {
        int step = 0;
        UenrResult lr = uenr_model_load(m, path, &step, with_opt);
        if (lr == UENR_OK) printf("loaded %s (step %d)\n", path, step);
        else fprintf(stderr, "warning: cannot load %s (%d)\n", path, (int)lr);
    } else {
        fprintf(stderr, "warning: %s missing; using fresh weights\n", path ? path : "(null)");
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

static float sigmoidf(float x) { return uenr_sigmoidf(x); }

/* ------------------------------------------------------------------------- */
/* DPO                                                                       */
/* ------------------------------------------------------------------------- */

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
    if (model_load_or_init(&policy, cfg, cfg->paths.model_out, 1) != UENR_OK) goto fail;
    const char *ref_path = cfg->dpo.ref_checkpoint[0] ? cfg->dpo.ref_checkpoint : cfg->paths.model_out;
    if (model_load_or_init(&ref, cfg, ref_path, 0) != UENR_OK) goto fail;

    UenrPool *pool = NULL;
    int cpus[2] = { cfg->run.pin_cpu0, cfg->run.pin_cpu1 };
    if (cfg->run.threads > 1) uenr_pool_init(&pool, cfg->run.threads, cpus, 2);
    UenrGemm gemm;
    if (uenr_gemm_init(&gemm, pool) != UENR_OK) goto fail;
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
        float sig = sigmoidf(z);
        float loss = -uenr_logf(sig + 1e-12f);
        float dcoef = beta * (1.0f - sig);

        uenr_model_zero_grad(&policy);
        seq_logp(&policy, &gemm, &ap, wid, wtg, wn, ignore, 1);
        uenr_scale(ap.dlogits, (size_t)wn * cfg->model.vocab_size, -dcoef);
        uenr_model_backward(&policy, &gemm, &ap);
        seq_logp(&policy, &gemm, &ap, lid, ltg, ln, ignore, 1);
        uenr_scale(ap.dlogits, (size_t)ln * cfg->model.vocab_size, dcoef);
        uenr_model_backward(&policy, &gemm, &ap);

        uenr_model_clip_grad(&policy, cfg->train.clip, pool);
        float lr = uenr_schedule_lr(cfg, step, steps, 0.0f);
        uenr_model_adamw(&policy, step, lr, cfg->train.beta1, cfg->train.beta2,
                         cfg->train.adam_eps, cfg->train.weight_decay, pool);
        uenr_model_zero_grad(&policy);

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
    uenr_model_free(&policy); uenr_model_free(&ref);
    uenr_pairs_free(&pairs);
    uenr_tok_free(tok);
    return UENR_OK;
fail_acts:
    uenr_acts_free(&ap);
fail_gemm:
    uenr_gemm_destroy(&gemm);
    if (pool) uenr_pool_destroy(pool);
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

static int sample_rl(const float *logits, int V, float temp, UenrRng *rng,
                     float *probs, float *logp_out) {
    float mx = logits[0];
    for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        probs[i] = uenr_expf((logits[i] - mx) / temp);
        sum += probs[i];
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
    *logp_out = uenr_logf(probs[pick] + 1e-12f);
    return pick;
}

static UenrResult run_rl(const UenrConfig *cfg, int is_ppo) {
    UenrTokenizer *tok = NULL;
    UenrResult r = uenr_tok_load(cfg, cfg->paths.tokenizer, &tok);
    if (r != UENR_OK) { fprintf(stderr, "rl: tokenizer missing\n"); return r; }
    UenrPairData pairs;
    r = uenr_pairs_load(cfg->paths.pairs_jsonl, &pairs);
    if (r != UENR_OK) { fprintf(stderr, "rl: cannot load pairs\n"); uenr_tok_free(tok); return r; }

    UenrModel policy, ref;
    memset(&policy, 0, sizeof(policy));
    memset(&ref, 0, sizeof(ref));
    if (model_load_or_init(&policy, cfg, cfg->paths.model_out, 1) != UENR_OK) goto fail;
    const char *ref_path = cfg->dpo.ref_checkpoint[0] ? cfg->dpo.ref_checkpoint : cfg->paths.model_out;
    if (model_load_or_init(&ref, cfg, ref_path, 0) != UENR_OK) goto fail;

    UenrPool *pool = NULL;
    int cpus[2] = { cfg->run.pin_cpu0, cfg->run.pin_cpu1 };
    if (cfg->run.threads > 1) uenr_pool_init(&pool, cfg->run.threads, cpus, 2);
    UenrGemm gemm;
    if (uenr_gemm_init(&gemm, pool) != UENR_OK) goto fail;
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
                int nxt = sample_rl(lg, V, 1.0f, &rng, probs, &lp);
                old_lp += lp;
                seq[g][n++] = nxt;
                if (nxt == end_id || nxt == cfg->tok.eos_id) break;
            }
            slen[g] = n;
            spn[g] = pn;
            sold[g] = old_lp;
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
            float coef;
            if (is_ppo) {
                float ratio = uenr_expf(new_lp - sold[g]);
                float lo = 1.0f - cfg->ppo.clip_ratio;
                float hi = 1.0f + cfg->ppo.clip_ratio;
                coef = (ratio >= lo && ratio <= hi) ? (-adv * ratio) : 0.0f;
                coef += cfg->grpo.kl_coef;
            } else {
                coef = -adv + cfg->grpo.kl_coef;
            }
            (void)ref_lp;
            uenr_scale(ap.dlogits, (size_t)n * V, coef);
            uenr_model_backward(&policy, &gemm, &ap);
            total_loss += -adv * new_lp;
        }

        uenr_model_clip_grad(&policy, cfg->train.clip, pool);
        float lr = uenr_schedule_lr(cfg, step, steps, 0.0f);
        uenr_model_adamw(&policy, step, lr, cfg->train.beta1, cfg->train.beta2,
                         cfg->train.adam_eps, cfg->train.weight_decay, pool);
        uenr_model_zero_grad(&policy);
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
fail:
    uenr_model_free(&policy); uenr_model_free(&ref);
    uenr_pairs_free(&pairs);
    uenr_tok_free(tok);
    return UENR_ERR_INVALID;
}

UenrResult uenr_train_ppo(const UenrConfig *cfg) { return run_rl(cfg, 1); }
UenrResult uenr_train_grpo(const UenrConfig *cfg) { return run_rl(cfg, 0); }
