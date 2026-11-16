#include "uenr/model.h"
#include "uenr/uenr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    UenrConfig cfg;
    uenr_config_load("config/uenr.toml", &cfg);
    cfg.model.vocab_size = 16;
    cfg.model.hidden = 8;
    cfg.model.layers = 1;
    cfg.model.heads = 2;
    cfg.model.kv_heads = 1;
    cfg.model.head_dim = 4;
    cfg.model.ffn = 8;
    cfg.model.max_seq = 8;
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
    printf("kv cache vs full recompute: max logit error %.3e\n", worst);
    int fails = 0;
    if (worst > 1e-3f) { printf("FAIL kv cache mismatch\n"); fails++; }
    else printf("ok   kv cache matches full recompute\n");

    free(step_logits);
    uenr_kv_cache_free(&c);
    uenr_acts_free(&a);
    uenr_gemm_destroy(&g);
    uenr_model_free(&m);
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
