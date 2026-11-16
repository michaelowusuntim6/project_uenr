/* q4 KV accuracy check: compare cached decode logits with FP32 KV. */
#include "uenr/uenr.h"
#include "uenr/model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const char *mp = argc > 1 ? argv[1] : "checkpoints/model.bin";
    int rotate = argc > 2 ? atoi(argv[2]) : 0;
    UenrConfig cfg;
    if (uenr_config_load("config/uenr.toml", &cfg) != UENR_OK) return 1;
    UenrRng rng; uenr_rng_seed(&rng, 1);
    UenrModel m;
    if (uenr_model_init(&m, &cfg, &rng) != UENR_OK) return 1;
    int step = 0;
    if (uenr_model_load(&m, mp, &step, 0) != UENR_OK) { printf("load fail\n"); return 1; }
    UenrGemm g; uenr_gemm_init(&g, NULL);
    const int V = m.V, N = 64;
    int *tok = malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) tok[i] = (i * 37 + 5) % (V - 16) + 16;
    float *l0 = malloc((size_t)N * V * sizeof(float));
    float *lq = malloc((size_t)N * V * sizeof(float));
    UenrKVCache c0; uenr_kv_cache_init(&c0, &cfg, cfg.model.max_seq);
    for (int i = 0; i < N; i++) uenr_model_forward_step(&m, &g, &c0, tok[i], i, l0 + (size_t)i * V);
    cfg.infer.kv_q4 = 1;
    cfg.infer.kv_rotate = rotate;
    UenrKVCache cq; uenr_kv_cache_init(&cq, &cfg, cfg.model.max_seq);
    for (int i = 0; i < N; i++) uenr_model_forward_step(&m, &g, &cq, tok[i], i, lq + (size_t)i * V);
    double maxe = 0, sum = 0, maxlog = 0;
    long long agree = 0;
    for (size_t i = 0; i < (size_t)N * V; i++) {
        double e = fabs((double)l0[i] - lq[i]);
        if (e > maxe) maxe = e;
        sum += e;
        if (fabs(l0[i]) > maxlog) maxlog = fabs(l0[i]);
    }
    for (int t = 0; t < N; t++) {
        int b0 = 0, bq = 0;
        for (int v = 1; v < V; v++) {
            if (l0[(size_t)t * V + v] > l0[(size_t)t * V + b0]) b0 = v;
            if (lq[(size_t)t * V + v] > lq[(size_t)t * V + bq]) bq = v;
        }
        agree += (b0 == bq);
    }
    printf("q4 rotate=%d: max|err| %.4g mean|err| %.4g max|logit| %.4g argmax %lld/%d\n",
           rotate, maxe, sum / ((double)N * V), maxlog, agree, N);
    return 0;
}
