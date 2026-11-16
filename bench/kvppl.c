/* KV-cache perplexity at long context: fp32 vs eviction/sinks. */
#include "uenr/uenr.h"
#include "uenr/model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *mp = argc > 1 ? argv[1] : "checkpoints/model.bin";
    int S = argc > 2 ? atoi(argv[2]) : 2048;
    const char *mode = argc > 3 ? argv[3] : "fp32";
    int W = argc > 4 ? atoi(argv[4]) : 256;
    int sinks = argc > 5 ? atoi(argv[5]) : 4;
    UenrConfig cfg;
    if (uenr_config_load("config/uenr.toml", &cfg) != UENR_OK) return 1;
    cfg.model.max_seq = S > cfg.model.max_seq ? S : cfg.model.max_seq;
    UenrTokenData d;
    if (uenr_data_load(cfg.paths.val_bin, &d) != UENR_OK) { printf("no val.bin\n"); return 1; }
    if ((size_t)S + 1 > d.count) S = (int)d.count - 1;
    cfg.infer.kv_evict = 0;
    if (!strcmp(mode, "evict")) {
        cfg.infer.kv_evict = 1;
        cfg.infer.kv_evict_window = W;
        cfg.infer.kv_sinks = sinks;
    } else if (mode[0] == 'l' && mode[1] == 'r') {
        cfg.infer.kv_lowrank = atoi(mode + 2);
    }
    UenrRng rng; uenr_rng_seed(&rng, 123);
    UenrModel m;
    if (uenr_model_init(&m, &cfg, &rng) != UENR_OK) return 1;
    int st = 0;
    if (uenr_model_load(&m, mp, &st, 0) != UENR_OK) { printf("load fail\n"); return 1; }
    UenrGemm g; uenr_gemm_init(&g, NULL);
    UenrKVCache c;
    uenr_kv_cache_init(&c, &cfg, cfg.model.max_seq);
    float *lo = malloc((size_t)m.V * sizeof(float));
    size_t start = uenr_rng_range(&rng, (int)(d.count - S - 1));
    double t0 = now(), ce = 0.0;
    int n = 0;
    for (int t = 0; t < S; t++) {
        uenr_model_forward_step(&m, &g, &c, (int)d.tokens[start + t], t, lo);
        if (t + 1 < S) {
            int y = (int)d.tokens[start + t + 1];
            float mx = lo[0];
            for (int i = 1; i < m.V; i++) if (lo[i] > mx) mx = lo[i];
            double sum = 0.0;
            for (int i = 0; i < m.V; i++) sum += exp((double)lo[i] - mx);
            ce += (log(sum) + mx) - lo[y];
            n++;
        }
    }
    double dt = now() - t0;
    double bytes = (c.lr > 0) ? (double)c.slot_rows * c.G * c.lr * 8.0
                              : (double)c.slot_rows * c.KV * 8.0;
    printf("%s S=%d W=%d sinks=%d: ppl %.3f  %.0f tok/s  slots %zu (%.2f MB)%s\n",
           mode, S, W, sinks, exp(ce / (n ? n : 1)), S / dt, c.slot_rows, bytes / 1e6,
           c.lr ? " lowrank" : " fp32");
    return 0;
}
