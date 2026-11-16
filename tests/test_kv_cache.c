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

    /* batched span (speculative verification) matches full recompute */
    {
        UenrKVCache cs;
        uenr_kv_cache_init(&cs, &cfg, 8);
        for (int pos = 0; pos < 3; pos++)
            uenr_model_forward_step(&m, &g, &cs, tok[pos], pos, step_logits);
        float *sl = malloc((size_t)3 * V * sizeof(float));
        int r = uenr_model_forward_span(&m, &g, &cs, tok + 3, 3, 3, sl);
        float worsts = 0.0f;
        for (int i = 0; i < 3 * V; i++) {
            float e = fabsf(sl[i] - a.logits[(size_t)(3 + i / V) * V + (i % V)]);
            if (e > worsts) worsts = e;
        }
        printf("span vs full recompute: err %d max %.3e\n", r, worsts);
        if (r != UENR_OK || worsts > 1e-3f) { printf("FAIL span mismatch\n"); fails++; }
        else printf("ok   batched span matches full recompute\n");
        free(sl);
        uenr_kv_cache_free(&cs);
    }

    /* batched multi-sequence decode matches per-sequence forward_step */
    {
        UenrKVCache c1, c2;
        uenr_kv_cache_init(&c1, &cfg, 8);
        uenr_kv_cache_init(&c2, &cfg, 8);
        for (int pos = 0; pos < 5; pos++) {
            uenr_model_forward_step(&m, &g, &c1, tok[pos], pos, step_logits);
            uenr_model_forward_step(&m, &g, &c2, tok[(pos + 3) % 6], pos, step_logits);
        }
        UenrKVCache *cs[2] = { &c1, &c2 };
        int mt[2] = { tok[5], tok[2] };
        int mp[2] = { 5, 5 };
        float *ml = malloc((size_t)2 * V * sizeof(float));
        int r = uenr_model_forward_multi(&m, &g, cs, 2, mt, mp, ml);
        uenr_model_forward_step(&m, &g, &c1, tok[5], 5, step_logits);
        float worstm = 0.0f;
        for (int v = 0; v < V; v++) {
            float e = fabsf(ml[v] - step_logits[v]);
            if (e > worstm) worstm = e;
        }
        printf("multi vs step: err %d max %.3e\n", r, worstm);
        if (r != UENR_OK || worstm > 1e-4f) { printf("FAIL multi mismatch\n"); fails++; }
        else printf("ok   batched multi-sequence decode matches step\n");
        free(ml);
        uenr_kv_cache_free(&c1); uenr_kv_cache_free(&c2);
    }

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
    /* fp16 cache */
    cfg.infer.kv_int8 = 0;
    cfg.infer.kv_fp16 = 1;
    {
        UenrKVCache ch;
        if (uenr_kv_cache_init(&ch, &cfg, 8) != UENR_OK) { printf("FAIL fp16 init\n"); return 1; }
        float worsth = 0.0f;
        for (int pos = 0; pos < 6; pos++) {
            uenr_model_forward_step(&m, &g, &ch, tok[pos], pos, step_logits);
            for (int v = 0; v < V; v++) {
                float e = fabsf(step_logits[v] - a.logits[(size_t)pos * V + v]);
                if (e > worsth) worsth = e;
            }
        }
        printf("fp16 kv cache: max logit error %.3e, bytes %.0f\n", worsth,
               (double)ch.slot_rows * (double)ch.KV * 4.0);
        if (worsth > 0.01f) { printf("FAIL fp16 kv too inaccurate\n"); fails++; }
        else printf("ok   fp16 kv within tolerance\n");
        uenr_kv_cache_free(&ch);
    }
    cfg.infer.kv_fp16 = 0;

    /* q4 cache */
    cfg.infer.kv_q4 = 1;
    {
        UenrKVCache cq;
        if (uenr_kv_cache_init(&cq, &cfg, 8) != UENR_OK) { printf("FAIL q4 init\n"); return 1; }
        float worstq = 0.0f;
        for (int pos = 0; pos < 6; pos++) {
            uenr_model_forward_step(&m, &g, &cq, tok[pos], pos, step_logits);
            for (int v = 0; v < V; v++) {
                float e = fabsf(step_logits[v] - a.logits[(size_t)pos * V + v]);
                if (e > worstq) worstq = e;
            }
        }
        double bytes = (double)cq.slot_rows * cq.KV * 1.0 +
                       (double)cq.slot_rows * cq.G * sizeof(float) * 2.0;
        printf("q4 kv cache: max logit error %.3e, bytes %.0f\n", worstq, bytes);
        if (worstq > 0.5f) { printf("FAIL q4 kv too inaccurate\n"); fails++; }
        else printf("ok   q4 kv within tolerance\n");
        uenr_kv_cache_free(&cq);
    }
    cfg.infer.kv_q4 = 0;

    /* q4 + Hadamard rotation: dot/weighted-sum invariant */
    cfg.infer.kv_q4 = 1;
    cfg.infer.kv_rotate = 1;
    {
        UenrKVCache cr;
        uenr_kv_cache_init(&cr, &cfg, 8);
        float worstr = 0.0f;
        for (int pos = 0; pos < 6; pos++) {
            uenr_model_forward_step(&m, &g, &cr, tok[pos], pos, step_logits);
            for (int v = 0; v < V; v++) {
                float e = fabsf(step_logits[v] - a.logits[(size_t)pos * V + v]);
                if (e > worstr) worstr = e;
            }
        }
        printf("q4+rotate kv cache: max logit error %.3e\n", worstr);
        if (worstr > 0.6f) { printf("FAIL q4+rotate too inaccurate\n"); fails++; }
        else printf("ok   q4+rotate within tolerance\n");
        uenr_kv_cache_free(&cr);
    }
    cfg.infer.kv_q4 = 0;
    cfg.infer.kv_rotate = 0;

    /* ternary q2 + mixed recent/sinks */
    cfg.infer.kv_q2 = 1;
    cfg.infer.kv_recent = 2;
    cfg.infer.kv_sinks = 1;
    {
        UenrKVCache c2;
        if (uenr_kv_cache_init(&c2, &cfg, 8) != UENR_OK) { printf("FAIL q2 init\n"); return 1; }
        float worst2 = 0.0f;
        for (int pos = 0; pos < 6; pos++) {
            uenr_model_forward_step(&m, &g, &c2, tok[pos], pos, step_logits);
            for (int v = 0; v < V; v++) {
                float e = fabsf(step_logits[v] - a.logits[(size_t)pos * V + v]);
                if (e > worst2) worst2 = e;
            }
        }
        double bytes2 = (double)c2.slot_rows * c2.KV * 0.5 +
                        (double)c2.slot_rows * c2.G * 8.0;
        printf("q2 mixed kv: max logit error %.3e, bytes %.0f\n", worst2, bytes2);
        if (worst2 > 1.5f) { printf("FAIL q2 kv too inaccurate\n"); fails++; }
        else printf("ok   q2 mixed kv within tolerance\n");
        uenr_kv_cache_free(&c2);
    }
    cfg.infer.kv_q2 = 0;

    /* int8 weights */
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
