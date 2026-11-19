#include "uenr/uenr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

static void usage(void) {
    printf(
        "uenr — dependency-free tiny MoE LLM for Ivy Bridge\n"
        "usage: uenr [--config PATH] <command> [options]\n\n"
        "commands:\n"
        "  config                    print the resolved configuration\n"
        "  tok-train                 train the byte-level BPE tokenizer\n"
        "  prepare                   tokenize data + build SFT/pair files\n"
        "  train   [--steps N] [--resume] [--batch-size N] [--lr F]  pretrain\n"
        "  sft     [--steps N] [--load P]   supervised finetune\n"
        "  dpo     [--steps N] [--load P] [--ref P]  direct preference optimization\n"
        "  grpo    [--steps N] [--load P] [--ref P]  group-relative policy gradient\n"
        "  (pipeline: scripts/train_pipeline.sh --short | --steps N)\n"
        "  infer   [--prompt S] [--batch F] [--tokens N] [--model P] [--greedy] [--chat]\n"
        "  bench   [--size N] [--simd auto|avx1|avx2|avx512] [--threads N]\n"
        "          [--pin0 N] [--pin1 N] [--tile 0|1]  GEMM throughput\n"
        "  eval    [--split val] [--batches N] [--seq N]  validation loss/perplexity\n"
        "  selftest                  quick end-to-end smoke check\n");
}

/* Count distinct physical cores from the kernel's SMT sibling lists.
 * Falls back to the logical CPU count when sysfs is unavailable. */
static int physical_core_count(void) {
    int n = 0;
    char seen[256][64];
    char path[128], buf[32];
    for (int cpu = 0; cpu < 256; cpu++) {
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        if (!fgets(buf, sizeof(buf), f)) { fclose(f); continue; }
        fclose(f);
        buf[strcspn(buf, "\n")] = '\0';
        int dup = 0;
        for (int i = 0; i < n; i++)
            if (!strcmp(seen[i], buf)) { dup = 1; break; }
        if (!dup && n < 256) snprintf(seen[n++], sizeof(seen[0]), "%s", buf);
    }
    if (n > 0) return n;
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    return online > 0 ? (int)online : 1;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int bench_gemm(const UenrConfig *cfg, int n, int tile) {
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
    if (tile >= 0) uenr_gemm_set_avx2_tile(&g, tile);

    /* Best of three 0.5 s windows: hosted runners are shared and a single
     * window can be delayed by a noisy neighbour. */
    double flops = 0.0;
    int iters = 0;
    for (int rep = 0; rep < 3; rep++) {
        uenr_gemm(&g, n, n, n, A, n, 0, B, n, 0, C, n, 1.0f, 0.0f);
        int it = 0;
        double t0 = now_s();
        double elapsed = 0.0;
        do {
            uenr_gemm(&g, n, n, n, A, n, 0, B, n, 0, C, n, 1.0f, 0.0f);
            it++;
            elapsed = now_s() - t0;
        } while (elapsed < 0.5);
        double f = 2.0 * (double)n * n * n * it / elapsed;
        if (f > flops) { flops = f; iters = it; }
    }
    /* Report raw throughput and per-physical-core throughput.  The Ivy
     * Bridge 33.9 GFLOP/s/core figure in the docs is a laptop measurement;
     * cloud hosts boost higher, so a percentage-of-laptop-roof would be
     * misleading on the runner. */
    int cores = physical_core_count();
    printf("gemm %dx%dx%d: %.2f GFLOP/s (%.1f GFLOP/s/physical core, %d cores) "
           "over %d iters, simd=%s threads=%d tile=%d\n",
           n, n, n, flops / 1e9, flops / 1e9 / (double)cores, cores, iters,
           uenr_simd_name(), cfg->run.threads, g.avx2_tile);

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
    int batches = 0, seqlen = 0, resume = 0;
    const char *split = NULL;
    const char *batch_path = NULL;
    const char *load_path = NULL;
    const char *ref_path = NULL;
    const char *phase = "all";
    const char *simd_mode = NULL;
    int threads_override = 0, pin0_override = -1, pin1_override = -1;
    int batch_override = 0, accum_override = 0;
    float lr_override = 0.0f;
    int bench_tile = -1;
    const char *conv_from = NULL, *conv_to = NULL, *conv_in = NULL, *conv_out = NULL;
    int conv_limit = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--config") && i + 1 < argc) config_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batches") && i + 1 < argc) batches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq") && i + 1 < argc) seqlen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--split") && i + 1 < argc) split = argv[++i];
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch_path = argv[++i];
        else if (!strcmp(argv[i], "--load") && i + 1 < argc) load_path = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref_path = argv[++i];
        else if (!strcmp(argv[i], "--phase") && i + 1 < argc) phase = argv[++i];
        else if (!strcmp(argv[i], "--simd") && i + 1 < argc) simd_mode = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads_override = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch-size") && i + 1 < argc) batch_override = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--grad-accum") && i + 1 < argc) accum_override = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc) lr_override = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--pin0") && i + 1 < argc) pin0_override = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pin1") && i + 1 < argc) pin1_override = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tile") && i + 1 < argc) bench_tile = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--from") && i + 1 < argc) conv_from = argv[++i];
        else if (!strcmp(argv[i], "--to") && i + 1 < argc) conv_to = argv[++i];
        else if (!strcmp(argv[i], "--in") && i + 1 < argc) conv_in = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) conv_out = argv[++i];
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) conv_limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--greedy")) greedy = 1;
        else if (!strcmp(argv[i], "--resume")) resume = 1;
        else if (!strcmp(argv[i], "--chat")) chat = 1;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(); return 0; }
        else if (!cmd) cmd = argv[i];
        else { fprintf(stderr, "uenr: unknown argument '%s'\n", argv[i]); return 2; }
    }
    if (!cmd) { usage(); return 0; }
    if (!strcmp(cmd, "convert")) {
        if (!conv_from || !conv_to || !conv_in || !conv_out) {
            fprintf(stderr, "uenr convert: need --from --to --in --out [--limit]\n");
            return 2;
        }
        return uenr_convert_file(conv_from, conv_to, conv_in, conv_out, conv_limit) ? 0 : 1;
    }

    UenrConfig cfg;
    UenrResult r = uenr_config_load(config_path, &cfg);
    if (r != UENR_OK) {
        fprintf(stderr, "uenr: cannot load config '%s' (err %d)\n",
                config_path ? config_path : UENR_DEFAULT_CONFIG, (int)r);
        return 1;
    }
    if (threads_override > 0) cfg.run.threads = threads_override;
    if (batch_override > 0) cfg.train.batch_size = batch_override;
    if (accum_override > 0) cfg.train.grad_accum = accum_override;
    if (lr_override > 0.0f) cfg.train.lr = lr_override;
    if (cfg.train.batch_size <= 0 || cfg.train.grad_accum <= 0 ||
        cfg.train.batch_size % cfg.train.grad_accum != 0 ||
        cfg.train.lr <= 0.0f) {
        fprintf(stderr, "uenr: invalid batch-size/lr/accumulation override\n");
        return 2;
    }
    if (pin0_override >= 0) cfg.run.pin_cpu0 = pin0_override;
    if (pin1_override >= 0) cfg.run.pin_cpu1 = pin1_override;
    if (simd_mode) snprintf(cfg.run.simd, sizeof(cfg.run.simd), "%s", simd_mode);
    if (uenr_simd_configure(cfg.run.simd[0] ? cfg.run.simd : "auto") != UENR_OK)
        return 1;
    if (load_path) {
        snprintf(cfg.paths.model_out, sizeof(cfg.paths.model_out), "%s", load_path);
        cfg.run.require_load = 1;
    }
    if (ref_path) snprintf(cfg.dpo.ref_checkpoint, sizeof(cfg.dpo.ref_checkpoint), "%s", ref_path);
    if (resume) cfg.run.resume = 1;
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
        int needs_tok = !strcmp(phase, "all") || !strcmp(phase, "pretrain");
        if (needs_tok && uenr_tok_load(&cfg, cfg.paths.tokenizer, &tok) != UENR_OK) {
            fprintf(stderr, "prepare: tokenizer missing; run `uenr tok-train`\n");
            return 1;
        }
        r = uenr_data_prepare_phase(&cfg, tok, phase);
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
    if (!strcmp(cmd, "dpo")) {
        return uenr_train_dpo(&cfg) == UENR_OK ? 0 : 1;
    }
    if (!strcmp(cmd, "grpo")) {
        return uenr_train_grpo(&cfg) == UENR_OK ? 0 : 1;
    }
    if (!strcmp(cmd, "infer")) {
        if (batch_path) {
            UenrResult br = cfg.infer.continuous
                ? uenr_infer_continuous(&cfg, model_path, batch_path, tokens, greedy)
                : uenr_infer_batch(&cfg, model_path, batch_path, tokens, greedy);
            return br == UENR_OK ? 0 : 1;
        }
        if (chat || !prompt)
            return uenr_infer_chat(&cfg, model_path, tokens, greedy) == UENR_OK ? 0 : 1;
        return uenr_infer(&cfg, model_path, NULL, prompt, tokens, greedy) == UENR_OK ? 0 : 1;
    }
    if (!strcmp(cmd, "eval")) {
        return uenr_eval(&cfg, model_path, split, batches, seqlen, 0) == UENR_OK ? 0 : 1;
    }
    if (!strcmp(cmd, "bench")) {
        return bench_gemm(&cfg, size > 0 ? size : cfg.bench.gemm_size, bench_tile);
    }
    if (!strcmp(cmd, "selftest")) {
        printf("selftest: use `make check` for the full suite\n");
        printf("config: params=%d active=%d\n", uenr_total_params(&cfg), uenr_active_params(&cfg));
        return bench_gemm(&cfg, 128, bench_tile);
    }

    fprintf(stderr, "uenr: unknown command '%s'\n", cmd);
    usage();
    return 2;
}
