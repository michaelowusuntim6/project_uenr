/* KV decode benchmark: fp32/fp16/int8/q4/q4+rotation at context P. */
#include "uenr/uenr.h"
#include "uenr/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *mp = argc > 1 ? argv[1] : "checkpoints/model.bin";
    int P = argc > 2 ? atoi(argv[2]) : 2048;
    int steps = 100;
    UenrConfig cfg;
    uenr_config_load("config/uenr.toml", &cfg);
    cfg.model.max_seq = P + steps + 8;
    UenrRng rng; uenr_rng_seed(&rng, 1);
    UenrModel m;
    if (uenr_model_init(&m, &cfg, &rng) != UENR_OK) return 1;
    int st = 0;
    if (uenr_model_load(&m, mp, &st, 0) != UENR_OK) { printf("load fail\n"); return 1; }
    UenrGemm g; uenr_gemm_init(&g, NULL);
    const int V = m.V;
    float *lo = malloc((size_t)V * sizeof(float));
    const char *names[6] = { "fp32", "fp16", "int8", "q4", "q4rot", "q2" };
    for (int k = 0; k < 6; k++) {
        cfg.infer.kv_int8 = cfg.infer.kv_fp16 = cfg.infer.kv_q4 = cfg.infer.kv_rotate = 0;
        if (k == 1) cfg.infer.kv_fp16 = 1;
        if (k == 2) cfg.infer.kv_int8 = 1;
        if (k == 3) cfg.infer.kv_q4 = 1;
        if (k == 4) { cfg.infer.kv_q4 = 1; cfg.infer.kv_rotate = 1; }
        if (k == 5) cfg.infer.kv_q2 = 1;
        UenrKVCache c;
        uenr_kv_cache_init(&c, &cfg, cfg.model.max_seq);
        int *toks = malloc((size_t)P * sizeof(int));
        for (int i = 0; i < P; i++) toks[i] = (i * 37 + 5) % (V - 16) + 16;
        double t0 = now();
        for (int i = 0; i < P; i++) uenr_model_forward_step(&m, &g, &c, toks[i], i, lo);
        double t1 = now();
        for (int i = 0; i < steps; i++) uenr_model_forward_step(&m, &g, &c, toks[i % P], P + i, lo);
        double t2 = now();
        double rows = (double)P + (cfg.model.sliding_window > 0 ? cfg.model.sliding_window : P);
        double bpe = (k == 0) ? 8.0 : (k == 1) ? 4.0 : (k == 2) ? 2.0 : (k == 5) ? 0.5 : 1.0;
        double bytes = rows * c.KV * bpe + ((k >= 3) ? rows * c.G * 8.0 : 0.0);
        if (k == 5) bytes += 2.0 * c.n_slots * cfg.infer.kv_recent * c.KV * 4.0 +
                             2.0 * c.n_slots * cfg.infer.kv_sinks * c.KV * 4.0;
        printf("%-6s prefill %5.0f tok/s | decode %5.0f tok/s | KV@P %.2f MB\n",
               names[k], P / (t1 - t0), steps / (t2 - t1), bytes / 1e6);
        free(toks);
        uenr_kv_cache_free(&c);
    }
    return 0;
}
