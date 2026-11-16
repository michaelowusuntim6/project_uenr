#include "uenr/model.h"
#include "uenr/uenr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static UenrConfig tiny_cfg(void) {
    UenrConfig c;
    uenr_config_load("config/uenr.toml", &c);
    c.model.vocab_size = 32;
    c.model.hidden = 8;
    c.model.layers = 1;
    c.model.heads = 2;
    c.model.kv_heads = 1;
    c.model.head_dim = 4;
    c.model.ffn = 8;
    c.model.max_seq = 8;
    c.model.tie_embeddings = 1;
    c.model.rope_theta = 10000.0f;
    c.moe.enabled = 1;
    c.moe.num_experts = 1;
    c.moe.experts_per_tok = 1;
    c.moe.expert_ffn = 4;
    c.moe.shared_expert = 1;
    c.moe.router_aux_coef = 0.0f;
    c.train.seq_len = 8;
    c.train.batch_size = 1;
    c.train.grad_accum = 1;
    c.run.threads = 1;
    return c;
}

static int run_case(UenrConfig cfg, const char *label) {
    const int T = 8, S = 8;
    UenrRng rng;
    uenr_rng_seed(&rng, 12345);

    UenrModel m;
    if (uenr_model_init(&m, &cfg, &rng) != UENR_OK) { printf("FAIL %s model init\n", label); return 1; }
    if (uenr_model_opt_init(&m, 0) != UENR_OK) { printf("FAIL %s opt init\n", label); return 1; }
    UenrGemm g;
    if (uenr_gemm_init(&g, NULL) != UENR_OK) { printf("FAIL %s gemm init\n", label); return 1; }
    UenrActs a;
    if (uenr_acts_init(&a, &cfg, T, S) != UENR_OK) { printf("FAIL %s acts init\n", label); return 1; }

    int tokens[8], targets[8];
    for (int i = 0; i < T; i++) {
        tokens[i] = (i * 7 + 3) % cfg.model.vocab_size;
        targets[i] = (i * 5 + 1) % cfg.model.vocab_size;
    }

    int count = 0;
    float loss = uenr_model_forward(&m, &g, &a, tokens, T, S, targets,
                                    cfg.tok.ignore_index, 1, &count);
    uenr_model_zero_grad(&m);
    uenr_model_backward(&m, &g, &a);

    const float eps = 2e-3f;
    float worst = 0.0f;
    int checked = 0, fails = 0;
    printf("[%s] loss=%.6f count=%d\n", label, loss, count);

    for (int ti = 0; ti < m.n_tensors && checked < 48; ti++) {
        UenrTensor *t = &m.tensors[ti];
        if (t->n == 0) continue;
        int picks[4] = { 0, (int)(t->n / 3), (int)(t->n / 2), (int)(t->n - 1) };
        for (int pi = 0; pi < 4 && checked < 48; pi++) {
            size_t idx = (size_t)picks[pi];
            float old = t->p[idx];
            t->p[idx] = old + eps;
            int c1 = 0;
            float lp = uenr_model_forward(&m, &g, &a, tokens, T, S, targets,
                                          cfg.tok.ignore_index, 0, &c1);
            t->p[idx] = old - eps;
            float lm = uenr_model_forward(&m, &g, &a, tokens, T, S, targets,
                                          cfg.tok.ignore_index, 0, &c1);
            t->p[idx] = old;

            float num = (lp - lm) / (2.0f * eps);
            float ana = m.grad[t->off + idx];
            float denom = fabsf(num) > fabsf(ana) ? fabsf(num) : fabsf(ana);
            if (denom < 1e-3f) denom = 1e-3f;
            float rel = fabsf(num - ana) / denom;
            if (rel > worst) worst = rel;
            checked++;
            if (rel > 0.10f && fabsf(num - ana) > 1.5e-3f) {
                printf("FAIL [%s] %-12s[%zu] num=%.6g ana=%.6g rel=%.3g\n",
                       label, t->name, idx, num, ana, rel);
                fails++;
            }
        }
    }
    printf("[%s] %d params, worst rel %.3g\n", label, checked, worst);
    uenr_acts_free(&a);
    uenr_gemm_destroy(&g);
    uenr_model_free(&m);
    return fails;
}

int main(void) {
    int fails = 0;
    UenrConfig dense = tiny_cfg();
    dense.moe.enabled = 0;
    fails += run_case(dense, "dense");

    UenrConfig moe1 = tiny_cfg();
    moe1.moe.enabled = 1;
    moe1.moe.num_experts = 1;
    fails += run_case(moe1, "moe-1expert");

    UenrConfig shared = tiny_cfg();
    shared.model.layers = 4;
    shared.model.kv_shared_layers = 2;
    strcpy(shared.model.attn_pattern, "SGSG");
    shared.model.kv_heads = 1;
    shared.model.sliding_window = 2;
    shared.model.attn_softcap = 20.0f;
    fails += run_case(shared, "shared-kv");

    /* One borrower on a single donor: the donor wk/wv gradient must include
     * both its own attention and the borrower's contribution. run_case's
     * first-48 tensor picks land on layer-0 wk/wv, so FD catches a missing
     * accumulation term. */
    UenrConfig donor = tiny_cfg();
    donor.model.layers = 2;
    donor.model.kv_shared_layers = 1;
    strcpy(donor.model.attn_pattern, "GG");
    donor.model.kv_heads = 1;
    donor.model.sliding_window = 0;
    donor.model.attn_softcap = 20.0f;
    fails += run_case(donor, "donor-accum");

    UenrConfig adap = tiny_cfg();
    adap.model.layers = 4;
    adap.model.kv_shared_layers = 2;
    strcpy(adap.model.attn_pattern, "SGSG");
    adap.model.kv_heads = 1;
    adap.model.kv_adapter = 1;
    adap.model.sliding_window = 2;
    fails += run_case(adap, "donor-adapter");

    UenrConfig ckpt = tiny_cfg();
    ckpt.model.layers = 2;
    ckpt.model.kv_heads = 1;
    ckpt.model.kv_shared_layers = 1;
    strcpy(ckpt.model.attn_pattern, "GG");
    ckpt.train.recompute_attn = 1;
    fails += run_case(ckpt, "recompute-attn");

    UenrConfig blk = tiny_cfg();
    blk.model.layers = 4;
    blk.model.kv_heads = 1;
    blk.model.kv_shared_layers = 2;
    strcpy(blk.model.attn_pattern, "SGSG");
    blk.model.sliding_window = 2;
    blk.train.block_checkpoint = 1;
    fails += run_case(blk, "block-checkpoint");

    UenrConfig blka = blk;
    blka.model.kv_adapter = 1;
    fails += run_case(blka, "block+donor-adapter");

    UenrConfig f16 = blk;
    f16.train.fp16_activations = 1;
    fails += run_case(f16, "block+fp16-probs");

    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
