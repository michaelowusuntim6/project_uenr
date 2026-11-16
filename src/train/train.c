#include "uenr/train.h"
#include "uenr/dataset.h"
#include "uenr/gemm.h"
#include "uenr/io.h"
#include "uenr/model.h"
#include "uenr/ops.h"
#include "uenr/schedule.h"
#include "uenr/thread.h"
#include "uenr/tokenizer.h"

#include <dirent.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static volatile sig_atomic_t g_stop = 0;

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

/* Read only a checkpoint header; returns the saved step or -1. */
static int ckpt_step(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t h[20];
    size_t got = fread(h, sizeof(uint32_t), 20, f);
    fclose(f);
    if (got != 20 || h[0] != 0x4c444d55u || (h[1] != 1 && h[1] != 2)) return -1;
    return (int)h[13];
}

/* Newest step_XXXXXX.bin in dir; returns -1 when none. */
static int latest_checkpoint(const char *dir, char *out, size_t cap) {
    DIR *d = opendir(dir);
    if (!d) return -1;
    int best = -1;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        int st = -1;
        if (sscanf(e->d_name, "step_%d.bin", &st) != 1 || st < 0) continue;
        if (st > best) {
            snprintf(out, cap, "%s/%s", dir, e->d_name);
            best = st;
        }
    }
    closedir(d);
    return best;
}

/* Keep the newest `keep` step checkpoints; delete older ones. */
static void prune_checkpoints(const char *dir, int keep) {
    DIR *d = opendir(dir);
    if (!d) return;
    int steps[256], n = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL && n < 256) {
        int st = -1;
        if (sscanf(e->d_name, "step_%d.bin", &st) == 1 && st >= 0)
            steps[n++] = st;
    }
    closedir(d);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (steps[j] > steps[i]) { int t = steps[i]; steps[i] = steps[j]; steps[j] = t; }
    for (int i = keep; i < n; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/step_%06d.bin", dir, steps[i]);
        remove(path);
    }
}

typedef int (*UenrBatchFn)(void *ctx, int seq, int nseq, int *inputs, int *labels);

typedef struct {
    const UenrTokenData *d;
    UenrRng *rng;
} PretrainCtx;

static int pretrain_batch(void *p, int seq, int nseq, int *inputs, int *labels) {
    PretrainCtx *c = (PretrainCtx *)p;
    return uenr_data_batch(c->d, c->rng, seq, nseq, inputs, labels);
}

typedef struct {
    const UenrSFTData *d;
    UenrRng *rng;
    int pad, ignore;
} SftCtx;

static int sft_batch(void *p, int seq, int nseq, int *inputs, int *labels) {
    SftCtx *c = (SftCtx *)p;
    return uenr_sft_batch(c->d, c->rng, seq, nseq, inputs, labels, c->pad, c->ignore);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static UenrResult train_generic(const UenrConfig *cfg, const char *label,
                                UenrBatchFn fn, void *ctx, size_t corpus_tokens,
                                const char *init_path, int load_init,
                                int resume, const char *out_path, int lora) {
    if (!cfg || !fn || corpus_tokens == 0) return UENR_ERR_INVALID;
    UenrRng rng;
    uenr_rng_seed(&rng, (uint64_t)cfg->run.seed);

    UenrModel model;
    UENR_TRY(uenr_model_init(&model, cfg, &rng));

    int start_step = 0;
    int total_steps = uenr_schedule_plan_steps(cfg, corpus_tokens);
    if (load_init && init_path && uenr_file_exists(init_path) == UENR_OK) {
        /* Resume from the newest of: periodic step checkpoints, the output
         * checkpoint (written on Ctrl-C/SIGTERM), or the configured init. */
        const char *pick = init_path;
        int best = ckpt_step(init_path);
        char latest[512];
        if (resume && !lora && latest_checkpoint(cfg->run.checkpoint_dir,
                                                 latest, sizeof(latest)) >= 0) {
            int ls = ckpt_step(latest);
            if (ls > best) { best = ls; pick = latest; }
        }
        int saved_step = 0;
        UenrResult lr = uenr_model_load(&model, pick, &saved_step, resume);
        if (lr == UENR_OK) {
            if (resume) start_step = saved_step;
            printf("[%s] loaded %s (step %d)\n", label, pick, saved_step);
        } else {
            printf("[%s] could not load %s (err %d); using fresh weights\n",
                   label, pick, (int)lr);
        }
    }
    if (lora) {
        UENR_TRY(uenr_model_lora_init(&model, cfg, &rng));
        printf("[%s] LoRA rank=%d scale=%.3f, base frozen\n",
               label, cfg->lora.rank, (double)model.lora_scale);
        if (cfg->train.qlora) {
            uenr_model_weight_qat(&model, 4);
            printf("[%s] QLoRA: frozen base quantize-dequantized to INT4\n", label);
        }
    }
    /* Allocate optimizer state after LoRA freezes the base: QLoRA/LoRA never
     * allocate base gradients or moments, only a bounded grad sink. */
    UENR_TRY(uenr_model_opt_init(&model, lora ? 0 : 1));
    if (lora)
        printf("[%s] low-RAM base: no base grad/optimizer state allocated\n", label);
    if (cfg->train.ema_decay > 0.0f) {
        UENR_TRY(uenr_model_ema_init(&model));
        printf("[%s] EMA enabled (decay %.4f)\n", label, (double)cfg->train.ema_decay);
    }
    if (start_step >= total_steps) total_steps = start_step + cfg->sched.min_steps_before_stop;

    const int seq0 = cfg->train.seq_len;
    const int seq_max = cfg->model.max_seq;
    const int micro = uenr_micro_batch(cfg);
    const int accum = cfg->train.grad_accum;
    const int seq_alloc = cfg->train.seq_ramp_steps > 0 ? seq_max : seq0;
    const int T_alloc = micro * seq_alloc;
    const int V = cfg->model.vocab_size;

    int *inputs = (int *)malloc((size_t)T_alloc * sizeof(int));
    int *labels = (int *)malloc((size_t)T_alloc * sizeof(int));
    if (!inputs || !labels) { free(inputs); free(labels); uenr_model_free(&model); return UENR_ERR_ALLOC; }

    UenrPool *pool = NULL;
    int cpus[2] = { cfg->run.pin_cpu0, cfg->run.pin_cpu1 };
    UenrResult pr = uenr_pool_init(&pool, cfg->run.threads, cpus, 2);
    if (pr != UENR_OK) {
        fprintf(stderr, "[%s] pool init failed (%d); running single-threaded\n", label, (int)pr);
        pool = NULL;
    } else if (cfg->run.threads > 1) {
        uenr_pin_cpu(cfg->run.pin_cpu0);
    }
    UenrPool *pool_mem = NULL;
    if (cfg->run.smt_threads > 1) {
        int cpus4[4] = { 0, 1, 2, 3 };
        (void)uenr_pool_init(&pool_mem, cfg->run.smt_threads, cpus4,
                             cfg->run.smt_threads > 4 ? 4 : cfg->run.smt_threads);
    }
    UenrGemm gemm;
    if (uenr_gemm_init(&gemm, pool) != UENR_OK) {
        if (pool) uenr_pool_destroy(pool);
        free(inputs); free(labels); uenr_model_free(&model);
        return UENR_ERR_ALLOC;
    }
    if (pool_mem) gemm.mem_pool = pool_mem;
    UenrActs acts;
    if (uenr_acts_init(&acts, cfg, T_alloc, seq_alloc) != UENR_OK) {
        uenr_gemm_destroy(&gemm);
        if (pool) uenr_pool_destroy(pool);
        free(inputs); free(labels); uenr_model_free(&model);
        return UENR_ERR_ALLOC;
    }

    uenr_model_report(&model);
    printf("[%s] plan: %d steps (%d micro x %d seq, accum %d), corpus %zu tokens\n",
           label, total_steps, micro, seq0, accum, corpus_tokens);

    UenrSchedState sched;
    uenr_sched_state_init(&sched);
    g_stop = 0;
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    FILE *csv = fopen(cfg->paths.log_csv, "a");
    if (csv) {
        if (ftell(csv) == 0)
            fprintf(csv, "step,lr,loss,ema,tokens_per_s,flops_per_s,mfu_pct\n");
    }

    const double active = (double)uenr_active_params(cfg) + (double)V * cfg->model.hidden;
    double t0 = now_s();
    double tokens_window = 0.0;
    int done = 0;

    for (int step = start_step + 1; step <= total_steps && !g_stop; step++) {
        int seq = seq0;
        if (cfg->train.seq_ramp_steps > 0) {
            int ramp = cfg->train.seq_ramp_steps;
            int d = step - start_step - 1;
            if (d > ramp) d = ramp;
            seq = seq0 + (seq_max - seq0) * d / ramp;
            seq = (seq / 8) * 8;
            if (seq < seq0) seq = seq0;
        }
        int T = micro * seq;
        uenr_model_zero_grad(&model);
        double step_loss = 0.0;
        long long step_tokens = 0;

        for (int mb = 0; mb < accum; mb++) {
            if (fn(ctx, seq, micro, inputs, labels) != 0) {
                fprintf(stderr, "[%s] batch fetch failed at step %d\n", label, step);
                done = 0;
                goto out;
            }
            int count = 0;
            float loss = uenr_model_forward(&model, &gemm, &acts, inputs, T, seq,
                                            labels, cfg->tok.ignore_index, 1, &count);
            if (count > 0) {
                float scale = 1.0f / ((float)accum * (float)count);
                uenr_scale(acts.dlogits, (size_t)T * V, scale);
                uenr_model_backward(&model, &gemm, &acts);
            }
            step_loss += (double)loss;
            step_tokens += count;
        }
        if (step_tokens == 0) step_tokens = 1;

        float gnorm = uenr_model_clip_grad(&model, cfg->train.clip, pool_mem ? pool_mem : pool);
        float lr = uenr_schedule_lr(cfg, step, total_steps, sched.ema);
        uenr_model_adamw(&model, step, lr, cfg->train.beta1, cfg->train.beta2,
                         cfg->train.adam_eps, cfg->train.weight_decay,
                         pool_mem ? pool_mem : pool);
        if (cfg->train.weight_qat_bits)
            uenr_model_weight_qat(&model, cfg->train.weight_qat_bits);
        if (cfg->train.ema_decay > 0.0f)
            uenr_model_ema_update(&model, cfg->train.ema_decay);

        float loss = (float)(step_loss / (double)step_tokens);
        uenr_sched_state_update(&sched, cfg, loss);

        tokens_window += (double)step_tokens;
        if (step == start_step + 1 || step % cfg->run.log_every == 0) {
            double dt = now_s() - t0;
            if (dt <= 0) dt = 1e-9;
            double tps = tokens_window / dt;
            double flops = 6.0 * active * (tokens_window / dt);
            double peak = 33.9e9 * (double)cfg->run.threads;
            double mfu = flops / peak * 100.0;
            printf("[%s] step %d/%d lr %.3g loss %.4f ema %.4f | %.0f tok/s %.2f GFLOP/s mfu %.1f%% |gn| %.3g\n",
                   label, step, total_steps, (double)lr, (double)loss,
                   (double)sched.ema, tps, flops / 1e9, mfu, (double)gnorm);
            if (csv) {
                fprintf(csv, "%d,%.8g,%.6f,%.6f,%.1f,%.3f,%.2f\n", step,
                        (double)lr, (double)loss, (double)sched.ema, tps, flops, mfu);
                fflush(csv);
            }
            t0 = now_s();
            tokens_window = 0.0;
        }

        if (!lora && cfg->run.checkpoint_every > 0 && step % cfg->run.checkpoint_every == 0) {
            char path[512];
            snprintf(path, sizeof(path), "%s/step_%06d.bin", cfg->run.checkpoint_dir, step);
            uenr_model_save(&model, path, step, 1);
            prune_checkpoints(cfg->run.checkpoint_dir, 3);
            printf("[%s] checkpoint %s\n", label, path);
        }

        if (uenr_sched_should_stop(&sched, cfg, step, total_steps)) {
            printf("[%s] early stop at step %d (ema %.4f, target %.3f)\n",
                   label, step, (double)sched.ema, (double)cfg->sched.target_loss);
            done = step;
            break;
        }
        done = step;
    }

out:
    if (g_stop && done > 0)
        printf("[%s] interrupted at step %d; saving resume checkpoint\n", label, done);
    if (done > 0) {
        if (lora) {
            uenr_model_lora_merge(&model, &gemm);
            printf("[%s] merged LoRA adapters into base weights\n", label);
        }
        UenrResult sr = uenr_model_save(&model, out_path, done, 1);
        printf("[%s] saved %s (step %d, err %d)\n", label, out_path, done, (int)sr);
        if (model.ema && cfg->paths.ema_out[0]) {
            UenrModel em = model;
            em.params = model.ema;
            em.opt_ready = 0;
            UenrResult er = uenr_model_save(&em, cfg->paths.ema_out, done, 0);
            printf("[%s] saved EMA weights %s (err %d)\n", label, cfg->paths.ema_out, (int)er);
        }
    }
    if (csv) fclose(csv);
    uenr_acts_free(&acts);
    uenr_gemm_destroy(&gemm);
    if (pool) uenr_pool_destroy(pool);
    if (pool_mem) uenr_pool_destroy(pool_mem);
    free(inputs);
    free(labels);
    uenr_model_free(&model);
    return UENR_OK;
}

UenrResult uenr_train_pretrain(const UenrConfig *cfg) {
    UenrTokenData data;
    UenrResult r = uenr_data_load(cfg->paths.train_bin, &data);
    if (r != UENR_OK) {
        fprintf(stderr, "train: cannot load %s (run `uenr prepare` first)\n", cfg->paths.train_bin);
        return r;
    }
    if (data.vocab != cfg->model.vocab_size) {
        fprintf(stderr, "train: data vocab %d != model vocab %d\n",
                data.vocab, cfg->model.vocab_size);
        uenr_data_free(&data);
        return UENR_ERR_SHAPE;
    }
    UenrRng rng;
    uenr_rng_seed(&rng, (uint64_t)cfg->run.seed + 1);
    PretrainCtx ctx = { &data, &rng };
    r = train_generic(cfg, "pretrain", pretrain_batch, &ctx, data.count,
                      cfg->paths.model_out, 1, cfg->run.resume,
                      cfg->paths.model_out, 0);
    uenr_data_free(&data);
    return r;
}

UenrResult uenr_train_sft(const UenrConfig *cfg) {
    UenrTokenizer *tok = NULL;
    UenrResult r = uenr_tok_load(cfg, cfg->paths.tokenizer, &tok);
    if (r != UENR_OK) {
        fprintf(stderr, "sft: cannot load tokenizer %s\n", cfg->paths.tokenizer);
        return r;
    }
    UenrSFTData data;
    r = uenr_sft_load(cfg, tok, cfg->paths.sft_jsonl, &data);
    uenr_tok_free(tok);
    if (r != UENR_OK) {
        fprintf(stderr, "sft: cannot load %s (run `uenr prepare` first)\n", cfg->paths.sft_jsonl);
        return r;
    }
    size_t tokens = 0;
    for (int i = 0; i < data.count; i++) tokens += (size_t)data.samples[i].n;

    UenrRng rng;
    uenr_rng_seed(&rng, (uint64_t)cfg->run.seed + 2);
    SftCtx ctx = { &data, &rng, cfg->tok.pad_id, cfg->tok.ignore_index };
    r = train_generic(cfg, "sft", sft_batch, &ctx, tokens ? tokens : 1,
                      cfg->paths.model_out, 1, 0, cfg->paths.sft_out, 0);
    uenr_sft_free(&data);
    return r;
}

UenrResult uenr_train_lora(const UenrConfig *cfg) {
    UenrTokenizer *tok = NULL;
    UenrResult r = uenr_tok_load(cfg, cfg->paths.tokenizer, &tok);
    if (r != UENR_OK) {
        fprintf(stderr, "lora: cannot load tokenizer %s\n", cfg->paths.tokenizer);
        return r;
    }
    UenrSFTData data;
    r = uenr_sft_load(cfg, tok, cfg->paths.sft_jsonl, &data);
    uenr_tok_free(tok);
    if (r != UENR_OK) {
        fprintf(stderr, "lora: cannot load %s\n", cfg->paths.sft_jsonl);
        return r;
    }
    size_t tokens = 0;
    for (int i = 0; i < data.count; i++) tokens += (size_t)data.samples[i].n;

    UenrRng rng;
    uenr_rng_seed(&rng, (uint64_t)cfg->run.seed + 3);
    SftCtx ctx = { &data, &rng, cfg->tok.pad_id, cfg->tok.ignore_index };
    r = train_generic(cfg, "lora", sft_batch, &ctx, tokens ? tokens : 1,
                      cfg->paths.model_out, 1, 0, cfg->paths.lora_out, 1);
    uenr_sft_free(&data);
    return r;
}
