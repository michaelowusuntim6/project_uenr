#include "uenr/model.h"
#include "uenr/uenr.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void) {
    UenrConfig cfg;
    uenr_config_load("config/uenr.toml", &cfg);
    cfg.model.vocab_size = 16;
    cfg.model.hidden = 8;
    cfg.model.layers = 4;
    cfg.model.kv_shared_layers = 2;
    snprintf(cfg.model.attn_pattern, sizeof(cfg.model.attn_pattern), "SGSG");
    cfg.model.heads = 2;
    cfg.model.kv_heads = 1;
    cfg.model.head_dim = 4;
    cfg.model.ffn = 8;
    cfg.model.max_seq = 8;
    cfg.model.sliding_window = 3;
    cfg.model.flash_attn = 0;
    cfg.moe.enabled = 1;
    cfg.moe.num_experts = 1;
    cfg.moe.experts_per_tok = 1;
    cfg.moe.expert_ffn = 4;
    cfg.moe.shared_expert = 1;
    cfg.moe.router_aux_coef = 0.0f;
    cfg.run.threads = 1;

    UenrRng rng;
    uenr_rng_seed(&rng, 2024);
    UenrModel m;
    uenr_model_init(&m, &cfg, &rng);
    UenrGemm g;
    uenr_gemm_init(&g, NULL);
    UenrActs a;
    uenr_acts_init(&a, &cfg, 8, 8);

    int tok[6] = { 1, 4, 7, 2, 9, 3 };
    int V = cfg.model.vocab_size;
    uenr_model_forward(&m, &g, &a, tok, 6, 6, NULL, cfg.tok.ignore_index, 0, NULL);

    UenrKVCache c;
    if (uenr_kv_cache_init(&c, &cfg, 8) != UENR_OK) { printf("FAIL cache init\n"); return 1; }
    float *step_logits = malloc((size_t)V * sizeof(float));
    float worst = 0.0f;
    for (int pos = 0; pos < 6; pos++) {
        uenr_model_forward_step(&m, &g, &c, tok[pos], pos, step_logits);
        for (int v = 0; v < V; v++) {
            float e = fabsf(step_logits[v] - a.logits[(size_t)pos * V + v]);
            if (e > worst) worst = e;
        }
    }
    size_t cbytes = c.slot_rows * (size_t)c.KV * sizeof(float) * 2;
    size_t oldbytes = (size_t)cfg.model.layers * 8 * (size_t)c.KV * sizeof(float) * 2;
    printf("kv cache rows %zu vs %d (%.2fx smaller, %zu bytes)\n",
           c.slot_rows, cfg.model.layers * 8, (double)oldbytes / (double)cbytes, cbytes);
    printf("kv cache vs full recompute: max logit error %.3e\n", worst);
    int fails = 0;
    if (worst > 1e-3f) { printf("FAIL kv cache mismatch\n"); fails++; }
    else printf("ok   kv cache matches full recompute\n");

    /* int8 cache: same trajectory within quantization tolerance */
    cfg.infer.kv_int8 = 1;
    UenrKVCache c8;
    if (uenr_kv_cache_init(&c8, &cfg, 8) != UENR_OK) { printf("FAIL int8 cache init\n"); return 1; }
    float worst8 = 0.0f;
    for (int pos = 0; pos < 6; pos++) {
        uenr_model_forward_step(&m, &g, &c8, tok[pos], pos, step_logits);
        for (int v = 0; v < V; v++) {
            float e = fabsf(step_logits[v] - a.logits[(size_t)pos * V + v]);
            if (e > worst8) worst8 = e;
        }
    }
    {
        int donor = cfg.model.layers - 1; /* shared global layer 3 -> donor 1 */
        int ok_alias = c8.slot_off[donor] == c8.slot_off[1] &&
                       c8.slot_cap[donor] == c8.slot_cap[1];
        printf("int8 donor scale/cache aliasing: %s\n", ok_alias ? "ok" : "BAD");
        printf("int8 kv cache bytes %.0f (rows %zu)\n",
               (double)c8.slot_rows * (double)c8.KV * 2.0, c8.slot_rows);
        if (!ok_alias) fails++;
    }
    printf("int8 kv cache vs full recompute: max logit error %.3e\n", worst8);
    if (worst8 > 0.25f) { printf("FAIL int8 kv cache too inaccurate\n"); fails++; }
    else printf("ok   int8 kv cache within tolerance\n");
    uenr_kv_cache_free(&c8);
    /* int8 weights */
    cfg.infer.kv_int8 = 0;
    cfg.infer.weight_int8 = 1;
    if (uenr_model_quantize_int8(&m) == UENR_OK) {
        UenrKVCache cq;
        uenr_kv_cache_init(&cq, &cfg, 8);
        float worstq = 0.0f;
        for (int pos = 0; pos < 6; pos++) {
            uenr_model_forward_step(&m, &g, &cq, tok[pos], pos, step_logits);
            for (int v = 0; v < V; v++) {
                float e = fabsf(step_logits[v] - a.logits[(size_t)pos * V + v]);
                if (e > worstq) worstq = e;
            }
        }
        printf("int8 weights vs full recompute: max logit error %.3e\n", worstq);
        if (worstq > 1.0f) { printf("FAIL int8 weights too inaccurate\n"); fails++; }
        else printf("ok   int8 weights within tolerance\n");
        uenr_kv_cache_free(&cq);
    }
    free(step_logits);
    uenr_kv_cache_free(&c);
    uenr_acts_free(&a);
    uenr_gemm_destroy(&g);
    uenr_model_free(&m);
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
