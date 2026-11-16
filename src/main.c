#include "uenr/uenr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(void) {
    printf(
        "uenr — dependency-free tiny MoE LLM for Ivy Bridge\n"
        "usage: uenr [--config PATH] <command> [options]\n\n"
        "commands:\n"
        "  config                    print the resolved configuration\n"
        "  tok-train                 train the byte-level BPE tokenizer\n"
        "  prepare                   tokenize data + build SFT/pair files\n"
        "  train   [--steps N]       pretrain\n"
        "  sft     [--steps N]       supervised finetune (assistant-only loss)\n"
        "  train-rm                  train the pairwise reward model\n"
        "  lora    [--steps N]       LoRA finetune on frozen base\n"
        "  dpo     [--steps N]       direct preference optimization\n"
        "  ppo     [--steps N]       clipped policy gradient (RLHF-style)\n"
        "  grpo    [--steps N]       group-relative policy gradient\n"
        "  infer   [--prompt S] [--tokens N] [--model P] [--greedy] [--chat]\n"
        "  bench   [--size N]        GEMM throughput vs the measured roof\n"
        "  selftest                  quick end-to-end smoke check\n");
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int bench_gemm(const UenrConfig *cfg, int n) {
    size_t bytes = ((size_t)n * n * sizeof(float) + 63) & ~(size_t)63;
    float *A = (float *)aligned_alloc(64, bytes);
    float *B = (float *)aligned_alloc(64, bytes);
    float *C = (float *)aligned_alloc(64, bytes);
    if (!A || !B || !C) { free(A); free(B); free(C); return 1; }
    UenrRng rng;
    uenr_rng_seed(&rng, 7);
    for (int i = 0; i < n * n; i++) { A[i] = uenr_rng_normal(&rng); B[i] = uenr_rng_normal(&rng); }

    UenrPool *pool = NULL;
    int cpus[2] = { cfg->run.pin_cpu0, cfg->run.pin_cpu1 };
    if (cfg->run.threads > 1) uenr_pool_init(&pool, cfg->run.threads, cpus, 2);
    UenrGemm g;
    if (uenr_gemm_init(&g, pool) != UENR_OK) return 1;

    uenr_gemm(&g, n, n, n, A, n, 0, B, n, 0, C, n, 1.0f, 0.0f);
    int iters = 0;
    double t0 = now_s();
    double elapsed = 0.0;
    do {
        uenr_gemm(&g, n, n, n, A, n, 0, B, n, 0, C, n, 1.0f, 0.0f);
        iters++;
        elapsed = now_s() - t0;
    } while (elapsed < 0.5);
    double flops = 2.0 * (double)n * n * n * iters / elapsed;
    double peak = 33.9e9 * (double)cfg->run.threads;
    printf("gemm %dx%dx%d: %.2f GFLOP/s over %d iters (%.1f%% of measured roof)\n",
           n, n, n, flops / 1e9, iters, flops / peak * 100.0);

    uenr_gemm_destroy(&g);
    if (pool) uenr_pool_destroy(pool);
    free(A); free(B); free(C);
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    const char *config_path = NULL;
    const char *cmd = NULL;
    const char *prompt = NULL;
    const char *model_path = NULL;
    int steps = 0, tokens = 0, greedy = 0, chat = 0, size = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--config") && i + 1 < argc) config_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--greedy")) greedy = 1;
        else if (!strcmp(argv[i], "--chat")) chat = 1;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(); return 0; }
        else if (!cmd) cmd = argv[i];
        else { fprintf(stderr, "uenr: unknown argument '%s'\n", argv[i]); return 2; }
    }
    if (!cmd) { usage(); return 0; }

    UenrConfig cfg;
    UenrResult r = uenr_config_load(config_path, &cfg);
    if (r != UENR_OK) {
        fprintf(stderr, "uenr: cannot load config '%s' (err %d)\n",
                config_path ? config_path : UENR_DEFAULT_CONFIG, (int)r);
        return 1;
    }
    if (steps > 0) {
        cfg.train.max_steps = steps;
        if (cfg.sched.min_steps_before_stop > steps)
            cfg.sched.min_steps_before_stop = steps;
        if (cfg.sched.plateau_window > steps)
            cfg.sched.plateau_window = steps;
    }

    if (!strcmp(cmd, "config")) {
        uenr_config_print(&cfg);
        return 0;
    }
    if (!strcmp(cmd, "tok-train")) {
        return uenr_tok_train(&cfg, cfg.paths.data, cfg.paths.tokenizer,
                              cfg.paths.vocab_file, cfg.paths.merges_file) == UENR_OK ? 0 : 1;
    }
    if (!strcmp(cmd, "prepare")) {
        UenrTokenizer *tok = NULL;
        if (uenr_tok_load(&cfg, cfg.paths.tokenizer, &tok) != UENR_OK) {
            fprintf(stderr, "prepare: tokenizer missing; run `uenr tok-train`\n");
            return 1;
        }
        r = uenr_data_prepare(&cfg, tok);
        if (r != UENR_OK) fprintf(stderr, "prepare: error %d\n", (int)r);
        uenr_tok_free(tok);
        return r == UENR_OK ? 0 : 1;
    }
    if (!strcmp(cmd, "train")) {
        return uenr_train_pretrain(&cfg) == UENR_OK ? 0 : 1;
    }
    if (!strcmp(cmd, "sft")) {
        return uenr_train_sft(&cfg) == UENR_OK ? 0 : 1;
    }
    if (!strcmp(cmd, "train-rm")) {
        return uenr_reward_train(&cfg) == UENR_OK ? 0 : 1;
    }
    if (!strcmp(cmd, "lora")) {
        return uenr_train_lora(&cfg) == UENR_OK ? 0 : 1;
    }
    if (!strcmp(cmd, "dpo")) {
        return uenr_train_dpo(&cfg) == UENR_OK ? 0 : 1;
    }
    if (!strcmp(cmd, "ppo")) {
        return uenr_train_ppo(&cfg) == UENR_OK ? 0 : 1;
    }
    if (!strcmp(cmd, "grpo")) {
        return uenr_train_grpo(&cfg) == UENR_OK ? 0 : 1;
    }
    if (!strcmp(cmd, "infer")) {
        if (chat || !prompt)
            return uenr_infer_chat(&cfg, model_path, tokens, greedy) == UENR_OK ? 0 : 1;
        return uenr_infer(&cfg, model_path, NULL, prompt, tokens, greedy) == UENR_OK ? 0 : 1;
    }
    if (!strcmp(cmd, "bench")) {
        return bench_gemm(&cfg, size > 0 ? size : cfg.bench.gemm_size);
    }
    if (!strcmp(cmd, "selftest")) {
        printf("selftest: use `make check` for the full suite\n");
        printf("config: params=%d active=%d\n", uenr_total_params(&cfg), uenr_active_params(&cfg));
        return bench_gemm(&cfg, 128);
    }

    fprintf(stderr, "uenr: unknown command '%s'\n", cmd);
    usage();
    return 2;
}
