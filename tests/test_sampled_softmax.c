#include "uenr/model.h"
#include "uenr/uenr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static UenrConfig cfg_for(int sampled, int negatives) {
    UenrConfig c;
    uenr_config_load("config/uenr.toml", &c);
    c.model.vocab_size = 16;
    c.model.hidden = 8;
    c.model.layers = 1;
    c.model.heads = 2;
    c.model.kv_heads = 1;
    c.model.head_dim = 4;
    c.model.ffn = 8;
    c.model.max_seq = 8;
    c.model.tie_embeddings = 1;
    c.moe.enabled = 1;
    c.moe.num_experts = 1;
    c.moe.experts_per_tok = 1;
    c.moe.expert_ffn = 4;
    c.moe.shared_expert = 1;
    c.moe.router_aux_coef = 0.0f;
    c.train.seq_len = 8;
    c.train.batch_size = 1;
    c.train.grad_accum = 1;
    c.train.sampled_softmax = sampled;
    c.train.sampled_negatives = negatives;
    c.run.threads = 1;
    return c;
}

static float run_once(UenrConfig cfg, UenrModel *m, UenrActs *a, UenrGemm *g,
                      const int *tok, const int *tgt) {
    int count = 0;
    float loss = uenr_model_forward(m, g, a, tok, 8, 8, tgt,
                                    cfg.tok.ignore_index, 1, &count);
    uenr_model_zero_grad(m);
    uenr_model_backward(m, g, a);
    return loss;
}

int main(void) {
    UenrConfig full = cfg_for(0, 0);
    UenrConfig samp = cfg_for(1, full.model.vocab_size - 1);

    UenrRng r1, r2;
    uenr_rng_seed(&r1, 4242);
    uenr_rng_seed(&r2, 4242);
    UenrModel m1, m2;
    uenr_model_init(&m1, &full, &r1);
    uenr_model_init(&m2, &samp, &r2);
    uenr_model_opt_init(&m1, 0);
    uenr_model_opt_init(&m2, 0);
    UenrGemm g1, g2;
    uenr_gemm_init(&g1, NULL);
    uenr_gemm_init(&g2, NULL);
    UenrActs a1, a2;
    uenr_acts_init(&a1, &full, 8, 8);
    uenr_acts_init(&a2, &samp, 8, 8);

    int tok[8], tgt[8];
    for (int i = 0; i < 8; i++) { tok[i] = (i * 3 + 1) % 16; tgt[i] = (i * 5 + 2) % 16; }
    float l1 = run_once(full, &m1, &a1, &g1, tok, tgt);
    float l2 = run_once(samp, &m2, &a2, &g2, tok, tgt);

    int fails = 0;
    if (fabsf(l1 - l2) > 1e-4f) {
        printf("FAIL loss full=%g sampled=%g\n", l1, l2);
        fails++;
    } else printf("ok   loss matches (full negatives) %.6f\n", l1);

    float maxerr = 0.0f;
    for (int ti = 0; ti < m1.n_tensors; ti++) {
        UenrTensor *t1 = &m1.tensors[ti];
        UenrTensor *t2 = &m2.tensors[ti];
        for (size_t i = 0; i < t1->n; i++) {
            float e = fabsf(m1.grad[t1->off + i] - m2.grad[t2->off + i]);
            if (e > maxerr) maxerr = e;
        }
    }
    if (maxerr > 2e-3f) { printf("FAIL grad maxerr=%g\n", maxerr); fails++; }
    else printf("ok   gradients match (full negatives), maxerr=%.2e\n", maxerr);

    /* sampled subset path: just ensure it runs and produces finite loss/grad */
    UenrConfig part = cfg_for(1, 4);
    UenrRng r3; uenr_rng_seed(&r3, 99);
    UenrModel m3; uenr_model_init(&m3, &part, &r3); uenr_model_opt_init(&m3, 0);
    UenrGemm g3; uenr_gemm_init(&g3, NULL);
    UenrActs a3; uenr_acts_init(&a3, &part, 8, 8);
    float l3 = run_once(part, &m3, &a3, &g3, tok, tgt);
    if (!isfinite(l3)) { printf("FAIL sampled subset loss not finite\n"); fails++; }
    else printf("ok   sampled subset loss finite %.4f\n", l3);

    uenr_acts_free(&a1); uenr_acts_free(&a2); uenr_acts_free(&a3);
    uenr_gemm_destroy(&g1); uenr_gemm_destroy(&g2); uenr_gemm_destroy(&g3);
    uenr_model_free(&m1); uenr_model_free(&m2); uenr_model_free(&m3);
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
