#include "uenr/model.h"
#include "uenr/io.h"
#include "uenr/math.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* allocation helpers                                                        */
/* ------------------------------------------------------------------------- */

static void *zalloc64(size_t bytes) {
    size_t n = (bytes + 63) & ~(size_t)63;
    if (n == 0) n = 64;
    void *p = aligned_alloc(64, n);
    if (p) memset(p, 0, n);
    return p;
}

static float *alloc_f(size_t n) { return (float *)zalloc64(n * sizeof(float)); }
static int   *alloc_i(size_t n) { return (int *)zalloc64(n * sizeof(int)); }

static float *tadd(UenrModel *m, size_t n, const char *name, int no_decay) {
    if (m->n_tensors >= UENR_MAX_TENSORS) return NULL;
    UenrTensor *t = &m->tensors[m->n_tensors++];
    t->off = m->alloc_floats;
    t->n = n;
    t->name = name;
    t->no_decay = no_decay;
    t->p = m->params + m->alloc_floats;
    m->alloc_floats += (n + 15) & ~(size_t)15;
    return t->p;
}

static inline float *grad_of(const UenrModel *m, const float *p) {
    return m->grad + (size_t)(p - m->params);
}
static inline float *adam_m_of(const UenrModel *m, const float *p) {
    return m->adam_m + (size_t)(p - m->params);
}
static inline float *adam_v_of(const UenrModel *m, const float *p) {
    return m->adam_v + (size_t)(p - m->params);
}

static float dotf(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static void softmax_inplace(float *p, int n) {
    float mx = p[0];
    for (int i = 1; i < n; i++) if (p[i] > mx) mx = p[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { p[i] = uenr_expf(p[i] - mx); sum += p[i]; }
    float inv = uenr_rcpf(sum);
    for (int i = 0; i < n; i++) p[i] *= inv;
}

static void topk_select(const float *p, int n, int k, int *idx, float *val) {
    for (int s = 0; s < k; s++) {
        int best = -1;
        float bv = -3.402823466e+38f;
        for (int j = 0; j < n; j++) {
            int taken = 0;
            for (int q = 0; q < s; q++) if (idx[q] == j) { taken = 1; break; }
            if (!taken && p[j] > bv) { bv = p[j]; best = j; }
        }
        idx[s] = best;
        val[s] = bv;
    }
}

/* ------------------------------------------------------------------------- */
/* LoRA helpers                                                              */
/* ------------------------------------------------------------------------- */

static int lora_target(const UenrConfig *cfg, const char *name) {
    for (int i = 0; i < cfg->lora.n_targets; i++)
        if (!strcmp(cfg->lora.targets[i], name)) return 1;
    return 0;
}

static void lora_fwd(UenrGemm *g, float *y, const float *x,
                     const float *A, const float *B,
                     int T, int in, int out, int r, float scale, float *h) {
    /* h = x A^T ; y += scale * h B^T */
    uenr_gemm(g, T, r, in, x, in, 0, A, r, 1, h, r, 1.0f, 0.0f);
    uenr_gemm(g, T, out, r, h, r, 0, B, r, 1, y, out, scale, 1.0f);
}

static void lora_bwd(UenrGemm *g, float *dx, float *dA, float *dB,
                     const float *dy, const float *x,
                     const float *A, const float *B,
                     int T, int in, int out, int r, float scale,
                     float *h, float *dh) {
    uenr_gemm(g, T, r, in, x, in, 0, A, r, 1, h, r, 1.0f, 0.0f);
    uenr_gemm(g, out, r, T, dy, out, 1, h, r, 0, dB, r, scale, 1.0f);
    uenr_gemm(g, T, r, out, dy, out, 0, B, r, 0, dh, r, scale, 0.0f);
    uenr_gemm(g, r, in, T, dh, r, 1, x, in, 0, dA, in, 1.0f, 1.0f);
    uenr_gemm(g, T, in, r, dh, r, 0, A, in, 0, dx, in, 1.0f, 1.0f);
}

UenrResult uenr_model_lora_init(UenrModel *m, const UenrConfig *cfg, UenrRng *rng) {
    if (!m || !cfg || !rng) return UENR_ERR_INVALID;
    int r = cfg->lora.rank;
    if (r <= 0) return UENR_ERR_INVALID;
    int C = m->C, KV = m->KV, L = m->L;
    m->lora_rank = r;
    m->lora_scale = cfg->lora.alpha / (float)r;
    m->lora_layers = (UenrLoraLayer *)calloc((size_t)L, sizeof(UenrLoraLayer));
    if (!m->lora_layers) return UENR_ERR_ALLOC;

    int tq = lora_target(cfg, "q_proj");
    int tk = lora_target(cfg, "k_proj");
    int tv = lora_target(cfg, "v_proj");
    int to = lora_target(cfg, "o_proj");
    size_t total = 0;
    for (int l = 0; l < L; l++) {
        if (tq) total += ((size_t)r * C + (size_t)C * r + 30) & ~(size_t)15;
        if (tk) total += ((size_t)r * C + (size_t)KV * r + 30) & ~(size_t)15;
        if (tv) total += ((size_t)r * C + (size_t)KV * r + 30) & ~(size_t)15;
        if (to) total += ((size_t)r * C + (size_t)C * r + 30) & ~(size_t)15;
    }
    if (total == 0) return UENR_ERR_INVALID;
    m->lora_p = alloc_f(total);
    m->lora_g = alloc_f(total);
    m->lora_m = alloc_f(total);
    m->lora_v = alloc_f(total);
    if (!m->lora_p || !m->lora_g || !m->lora_m || !m->lora_v) return UENR_ERR_ALLOC;
    m->lora_n = total;

    size_t off = 0;
    for (int l = 0; l < L; l++) {
        UenrLoraLayer *y = &m->lora_layers[l];
        if (tq) {
            y->Aq = m->lora_p + off; off += ((size_t)r * C + 15) & ~(size_t)15;
            y->Bq = m->lora_p + off; off += ((size_t)C * r + 15) & ~(size_t)15;
            y->has_q = 1;
        }
        if (tk) {
            y->Ak = m->lora_p + off; off += ((size_t)r * C + 15) & ~(size_t)15;
            y->Bk = m->lora_p + off; off += ((size_t)KV * r + 15) & ~(size_t)15;
            y->has_k = 1;
        }
        if (tv) {
            y->Av = m->lora_p + off; off += ((size_t)r * C + 15) & ~(size_t)15;
            y->Bv = m->lora_p + off; off += ((size_t)KV * r + 15) & ~(size_t)15;
            y->has_v = 1;
        }
        if (to) {
            y->Ao = m->lora_p + off; off += ((size_t)r * C + 15) & ~(size_t)15;
            y->Bo = m->lora_p + off; off += ((size_t)C * r + 15) & ~(size_t)15;
            y->has_o = 1;
        }
    }
    for (int l = 0; l < L; l++) {
        UenrLoraLayer *y = &m->lora_layers[l];
        if (y->has_q) {
            for (int i = 0; i < r * C; i++) y->Aq[i] = 0.02f * uenr_rng_normal(rng);
            for (int i = 0; i < C * r; i++) y->Bq[i] = 0.0f;
        }
        if (y->has_k) {
            for (int i = 0; i < r * C; i++) y->Ak[i] = 0.02f * uenr_rng_normal(rng);
            for (int i = 0; i < KV * r; i++) y->Bk[i] = 0.0f;
        }
        if (y->has_v) {
            for (int i = 0; i < r * C; i++) y->Av[i] = 0.02f * uenr_rng_normal(rng);
            for (int i = 0; i < KV * r; i++) y->Bv[i] = 0.0f;
        }
        if (y->has_o) {
            for (int i = 0; i < r * C; i++) y->Ao[i] = 0.02f * uenr_rng_normal(rng);
            for (int i = 0; i < C * r; i++) y->Bo[i] = 0.0f;
        }
    }
    m->lora_on = 1;
    m->freeze_base = 1;
    return UENR_OK;
}

void uenr_model_lora_merge(UenrModel *m, UenrGemm *g) {
    if (!m || !g || !m->lora_on || !m->lora_layers) return;
    int C = m->C, KV = m->KV, r = m->lora_rank;
    float s = m->lora_scale;
    for (int l = 0; l < m->L; l++) {
        UenrLoraLayer *y = &m->lora_layers[l];
        UenrLayer *b = &m->layers[l];
        if (y->has_q)
            uenr_gemm(g, C, C, r, y->Bq, r, 0, y->Aq, C, 0, b->wq, C, s, 1.0f);
        if (y->has_k)
            uenr_gemm(g, KV, C, r, y->Bk, r, 0, y->Ak, C, 0, b->wk, C, s, 1.0f);
        if (y->has_v)
            uenr_gemm(g, KV, C, r, y->Bv, r, 0, y->Av, C, 0, b->wv, C, s, 1.0f);
        if (y->has_o)
            uenr_gemm(g, C, C, r, y->Bo, r, 0, y->Ao, C, 0, b->wo, C, s, 1.0f);
    }
    m->lora_on = 0;
}

/* ------------------------------------------------------------------------- */
/* init                                                                      */
/* ------------------------------------------------------------------------- */

UenrResult uenr_model_init(UenrModel *m, const UenrConfig *cfg, UenrRng *rng) {
    if (!m || !cfg || !rng) return UENR_ERR_INVALID;
    memset(m, 0, sizeof(*m));
    m->cfg = *cfg;
    m->V = cfg->model.vocab_size;
    m->C = cfg->model.hidden;
    m->L = cfg->model.layers;
    m->H = cfg->model.heads;
    m->G = cfg->model.kv_heads;
    m->hd = cfg->model.head_dim;
    m->KV = m->G * m->hd;
    m->E = cfg->moe.num_experts;
    m->EF = cfg->moe.expert_ffn;
    m->FF = cfg->model.ffn;
    m->K = cfg->moe.experts_per_tok;
    m->tie = cfg->model.tie_embeddings;

    int expected = uenr_total_params(cfg);
    size_t cap = (size_t)expected + 16 * UENR_MAX_TENSORS;
    m->params = (float *)zalloc64(cap * sizeof(float));
    if (!m->params) return UENR_ERR_ALLOC;

    m->layers = (UenrLayer *)calloc((size_t)m->L, sizeof(UenrLayer));
    if (!m->layers) { free(m->params); m->params = NULL; return UENR_ERR_ALLOC; }

    m->tok_emb = tadd(m, (size_t)m->V * m->C, "tok_emb", 0);
    for (int l = 0; l < m->L; l++) {
        UenrLayer *y = &m->layers[l];
        y->rms_attn = tadd(m, (size_t)m->C, "rms_attn", 1);
        y->wq = tadd(m, (size_t)m->C * m->C, "wq", 0);
        y->wk = tadd(m, (size_t)m->KV * m->C, "wk", 0);
        y->wv = tadd(m, (size_t)m->KV * m->C, "wv", 0);
        y->wo = tadd(m, (size_t)m->C * m->C, "wo", 0);
        y->q_norm = tadd(m, (size_t)m->hd, "q_norm", 1);
        y->k_norm = tadd(m, (size_t)m->hd, "k_norm", 1);
        y->rms_ffn = tadd(m, (size_t)m->C, "rms_ffn", 1);
        y->router = tadd(m, (size_t)m->E * m->C, "router", 0);
        if (cfg->moe.enabled) {
            y->expert_wg = tadd(m, (size_t)m->E * m->EF * m->C, "expert_wg", 0);
            y->expert_wu = tadd(m, (size_t)m->E * m->EF * m->C, "expert_wu", 0);
            y->expert_wd = tadd(m, (size_t)m->E * m->C * m->EF, "expert_wd", 0);
        }
        if (!cfg->moe.enabled || cfg->moe.shared_expert) {
            y->shared_wg = tadd(m, (size_t)m->FF * m->C, "shared_wg", 0);
            y->shared_wu = tadd(m, (size_t)m->FF * m->C, "shared_wu", 0);
            y->shared_wd = tadd(m, (size_t)m->C * m->FF, "shared_wd", 0);
        }
        if (!y->rms_attn || !y->wq || !y->wk || !y->wv || !y->wo || !y->q_norm ||
            !y->k_norm || !y->rms_ffn || !y->router ||
            (cfg->moe.enabled && (!y->expert_wg || !y->expert_wu || !y->expert_wd))) {
            uenr_model_free(m);
            return UENR_ERR_ALLOC;
        }
    }
    m->rms_final = tadd(m, (size_t)m->C, "rms_final", 1);
    if (m->tie) {
        m->lm_head = m->tok_emb;
    } else {
        m->lm_head = tadd(m, (size_t)m->V * m->C, "lm_head", 0);
    }
    if (!m->tok_emb || !m->rms_final || !m->lm_head) {
        uenr_model_free(m);
        return UENR_ERR_ALLOC;
    }

    size_t sum = 0;
    for (int i = 0; i < m->n_tensors; i++) sum += m->tensors[i].n;
    if ((int)sum != expected) {
        fprintf(stderr, "model: tensor sum %zu != expected %d\n", sum, expected);
        uenr_model_free(m);
        return UENR_ERR_SHAPE;
    }

    /* Init: N(0, 0.02) for matrices, 1.0 for norm weights. */
    for (int i = 0; i < m->n_tensors; i++) {
        UenrTensor *t = &m->tensors[i];
        if (t->no_decay) {
            for (size_t j = 0; j < t->n; j++) t->p[j] = 1.0f;
        } else {
            for (size_t j = 0; j < t->n; j++) t->p[j] = 0.02f * uenr_rng_normal(rng);
        }
    }
    float res_scale = 1.0f / uenr_sqrtf(2.0f * (float)m->L);
    for (int l = 0; l < m->L; l++) {
        UenrLayer *y = &m->layers[l];
        uenr_scale(y->wo, (size_t)m->C * m->C, res_scale);
        if (y->expert_wd) uenr_scale(y->expert_wd, (size_t)m->E * m->C * m->EF, res_scale);
        if (y->shared_wd) uenr_scale(y->shared_wd, (size_t)m->C * m->FF, res_scale);
    }

    if (uenr_rope_init(&m->rope, cfg->model.max_seq, m->hd / 2, cfg->model.rope_theta) != UENR_OK) {
        uenr_model_free(m);
        return UENR_ERR_ALLOC;
    }
    return UENR_OK;
}

void uenr_model_free(UenrModel *m) {
    if (!m) return;
    uenr_rope_destroy(&m->rope);
    free(m->layers);
    free(m->params);
    free(m->grad);
    free(m->adam_m);
    free(m->adam_v);
    free(m->lora_layers);
    free(m->lora_p);
    free(m->lora_g);
    free(m->lora_m);
    free(m->lora_v);
    memset(m, 0, sizeof(*m));
}

UenrResult uenr_model_opt_init(UenrModel *m, int with_adam) {
    if (!m || !m->params) return UENR_ERR_INVALID;
    if (!m->grad) {
        m->grad = alloc_f(m->alloc_floats);
        if (!m->grad) return UENR_ERR_ALLOC;
    }
    if (with_adam) {
        if (!m->adam_m) {
            m->adam_m = alloc_f(m->alloc_floats);
            if (!m->adam_m) return UENR_ERR_ALLOC;
        }
        if (!m->adam_v) {
            m->adam_v = alloc_f(m->alloc_floats);
            if (!m->adam_v) return UENR_ERR_ALLOC;
        }
        m->opt_ready = 1;
    }
    return UENR_OK;
}

size_t uenr_model_param_count(const UenrModel *m) {
    if (!m) return 0;
    size_t n = 0;
    for (int i = 0; i < m->n_tensors; i++) n += m->tensors[i].n;
    return n;
}

void uenr_model_report(const UenrModel *m) {
    if (!m) return;
    printf("model: V=%d C=%d L=%d H=%d KV=%d hd=%d E=%d EF=%d FF=%d top=%d\n",
           m->V, m->C, m->L, m->H, m->G, m->hd, m->E, m->EF, m->FF, m->K);
    printf("model: %d tensors, %zu params, %.1f MiB fp32\n",
           m->n_tensors, uenr_model_param_count(m),
           (double)uenr_model_param_count(m) * 4.0 / (1024.0 * 1024.0));
}

/* ------------------------------------------------------------------------- */
/* activations                                                               */
/* ------------------------------------------------------------------------- */

UenrResult uenr_acts_init(UenrActs *a, const UenrConfig *cfg, int T, int S) {
    if (!a || !cfg || T <= 0 || S <= 0 || T % S != 0) return UENR_ERR_INVALID;
    memset(a, 0, sizeof(*a));
    a->T = T; a->S = S; a->B = T / S;
    a->C = cfg->model.hidden;
    a->L = cfg->model.layers;
    a->H = cfg->model.heads;
    a->G = cfg->model.kv_heads;
    a->hd = cfg->model.head_dim;
    a->KV = a->G * a->hd;
    a->E = cfg->moe.num_experts;
    a->EF = cfg->moe.expert_ffn;
    a->FF = cfg->model.ffn;
    a->V = cfg->model.vocab_size;
    a->K = cfg->moe.experts_per_tok;
    a->flash = cfg->model.flash_attn;
    a->sampled = cfg->train.sampled_softmax;
    a->sn = a->sampled ? cfg->train.sampled_negatives + 1 : 0;

    size_t T_ = (size_t)T, C = (size_t)a->C, L = (size_t)a->L, B = (size_t)a->B;
    size_t H = (size_t)a->H, G = (size_t)a->G, S_ = (size_t)S, KV = (size_t)a->KV;
    size_t E = (size_t)a->E, K = (size_t)a->K, EF = (size_t)a->EF, FF = (size_t)a->FF, V = (size_t)a->V;
    size_t TC = T_ * C, TKV = T_ * KV;

#define AF(field, count) do { a->field = alloc_f(count); if (!a->field) goto fail; } while (0)
#define AI(field, count) do { a->field = alloc_i(count); if (!a->field) goto fail; } while (0)

    AF(x, (L + 1) * TC);
    AF(xa, L * TC);
    AF(q_pre, L * TC);
    AF(q, L * TC);
    AF(k_pre, L * TKV);
    AF(k, L * TKV);
    AF(v, L * TKV);
    if (!a->flash) AF(probs, L * B * H * S_ * S_);
    AF(lse, L * T_ * H);
    AF(attn_out, L * TC);
    AF(x2, L * TC);
    AF(xf, L * TC);
    AF(rprobs, L * T_ * E);
    AI(eidx, L * T_ * K);
    AF(egate, L * T_ * K);
    AF(routed_y, L * T_ * K * C);
    AF(eout, L * TC);
    AI(ecount, L * E);
    AI(eoff, E + 1);
    AI(elist, T_ * K + 1);
    AI(cursor, E + 1);
    AF(rstd_attn, L * T_);
    AF(rstd_ffn, L * T_);
    AF(rstd_q, L * T_ * H);
    AF(rstd_k, L * T_ * G);
    AF(rstd_final, T_);
    AF(xfinal, TC);
    AF(logits, T_ * V);
    a->dlogits = a->logits; /* loss overwrites logits in place (item 8) */
    AF(dx, TC);
    AF(dxa, TC);
    AF(dattn, TC);
    AF(dq, TC);
    AF(dk, TKV);
    AF(dv, TKV);
    AF(dxf, TC);
    AF(dx2, TC);
    AF(drouter, T_ * E);
    AF(tmp, TC);
    AF(gate_dp, T_ * K + 1);
    AF(lora_h, TC);
    AF(lora_dh, TC);
    if (a->sampled) {
        AF(slogits, T_ * (size_t)a->sn);
        AI(sids, T_ * (size_t)a->sn);
        uenr_rng_seed(&a->sample_rng, (uint64_t)cfg->run.seed + 0x51ed270bULL);
    }
    AF(gxs, TC);
    AF(gys, TC);
    AF(gdys, TC);
    size_t sw = 5 * (T_ * EF > T_ * FF ? T_ * EF : T_ * FF);
    AF(sscratch, sw);
#undef AF
#undef AI
    return UENR_OK;

fail:
    uenr_acts_free(a);
    return UENR_ERR_ALLOC;
}

void uenr_acts_free(UenrActs *a) {
    if (!a) return;
    free(a->x); free(a->xa); free(a->q_pre); free(a->q);
    free(a->k_pre); free(a->k); free(a->v); free(a->probs); free(a->lse);
    free(a->attn_out); free(a->x2); free(a->xf); free(a->rprobs);
    free(a->eidx); free(a->egate); free(a->routed_y); free(a->eout);
    free(a->ecount); free(a->eoff); free(a->elist); free(a->cursor);
    free(a->rstd_attn); free(a->rstd_ffn); free(a->rstd_q); free(a->rstd_k);
    free(a->rstd_final); free(a->xfinal); free(a->logits);
    a->dlogits = NULL;
    free(a->dx); free(a->dxa); free(a->dattn); free(a->dq); free(a->dk);
    free(a->dv); free(a->dxf); free(a->dx2); free(a->drouter); free(a->tmp);
    free(a->gate_dp); free(a->lora_h); free(a->lora_dh);
    free(a->slogits); free(a->sids);
    free(a->gxs); free(a->gys); free(a->gdys); free(a->sscratch);
    memset(a, 0, sizeof(*a));
}

/* ------------------------------------------------------------------------- */
/* forward                                                                   */
/* ------------------------------------------------------------------------- */

static void layer_rms(float *dst, float *rstd, const float *src, const float *w,
                      int T, int D, float eps) {
    uenr_rmsnorm_fwd(dst, rstd, src, w, T, D, eps);
}

static inline float dot_row(const float *a, const float *b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8)));
    }
    float s = uenr_v_hsum(_mm256_add_ps(s0, s1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static float sampled_softmax_loss(UenrModel *m, UenrActs *a,
                                  const int *targets, int T, int V,
                                  int ignore, int *valid_count) {
    const int sn = a->sn, C = m->C;
    const float *W = m->lm_head;
    const float *xf = a->xfinal;
    float loss = 0.0f;
    int valid = 0;
    for (int t = 0; t < T; t++) {
        int base = t * sn;
        int y = targets[t];
        if (y == ignore || y < 0 || y >= V) {
            for (int i = 0; i < sn; i++) { a->sids[base + i] = -1; a->slogits[base + i] = 0.0f; }
            continue;
        }
        valid++;
        a->sids[base] = y;
        for (int i = 1; i < sn; i++) {
            int id;
            for (;;) {
                id = uenr_rng_range(&a->sample_rng, V);
                int dup = 0;
                for (int j = 0; j < i; j++) if (a->sids[base + j] == id) { dup = 1; break; }
                if (!dup && id != y) break;
            }
            a->sids[base + i] = id;
        }
        const float *xr = xf + (size_t)t * C;
        float mx = -3.402823466e+38f;
        for (int i = 0; i < sn; i++) {
            float z = dot_row(xr, W + (size_t)a->sids[base + i] * C, C);
            a->slogits[base + i] = z;
            if (z > mx) mx = z;
        }
        float zy = a->slogits[base];
        float sum = 0.0f;
        for (int i = 0; i < sn; i++) {
            float e = uenr_expf(a->slogits[base + i] - mx);
            a->slogits[base + i] = e;
            sum += e;
        }
        float inv = uenr_rcpf(sum);
        for (int i = 0; i < sn; i++) a->slogits[base + i] *= inv;
        a->slogits[base] -= 1.0f;
        loss += (uenr_logf(sum) + mx) - zy;
    }
    if (valid_count) *valid_count = valid;
    return loss;
}

float uenr_model_forward(UenrModel *m, UenrGemm *g, UenrActs *a,
                         const int *tokens, int T, int S,
                         const int *targets, int ignore_index,
                         int write_grad, int *valid_count) {
    if (!m || !g || !a || !tokens || T <= 0 || S <= 0 || T % S != 0) return 0.0f;
    const int C = m->C, L = m->L, H = m->H, G = m->G, hd = m->hd, KV = m->KV;
    const int E = m->E, EF = m->EF, FF = m->FF, V = m->V, K = m->K;
    const int B = T / S;
    const float eps = m->cfg.model.rms_eps;
    const float scale = uenr_rsqrtf((float)hd);
    a->T_cur = T; a->S_cur = S; a->tokens_cur = tokens;

    size_t TC = (size_t)T * C, TKV = (size_t)T * KV;

    /* embeddings */
    for (int t = 0; t < T; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= V) tok = 0;
        memcpy(a->x + (size_t)t * C, m->tok_emb + (size_t)tok * C, (size_t)C * sizeof(float));
    }

    for (int l = 0; l < L; l++) {
        UenrLayer *y = &m->layers[l];
        float *xl = a->x + (size_t)l * TC;
        float *xo = a->x + (size_t)(l + 1) * TC;
        float *xa = a->xa + (size_t)l * TC;
        float *qp = a->q_pre + (size_t)l * TC;
        float *q = a->q + (size_t)l * TC;
        float *kp = a->k_pre + (size_t)l * TKV;
        float *k = a->k + (size_t)l * TKV;
        float *vv = a->v + (size_t)l * TKV;
        float *ao = a->attn_out + (size_t)l * TC;
        float *x2 = a->x2 + (size_t)l * TC;
        float *xf = a->xf + (size_t)l * TC;
        float *rp = a->rprobs + (size_t)l * T * E;
        float *eg = a->egate + (size_t)l * T * K;
        int   *ei = a->eidx + (size_t)l * T * K;
        float *ry = a->routed_y + (size_t)l * T * K * C;
        int   *ec = a->ecount + (size_t)l * E;
        float *rstd_a = a->rstd_attn + (size_t)l * T;
        float *rstd_f = a->rstd_ffn + (size_t)l * T;
        float *rstd_q = a->rstd_q + (size_t)l * T * H;
        float *rstd_k = a->rstd_k + (size_t)l * T * G;
        float *probs = a->flash ? NULL : a->probs + (size_t)l * B * H * S * S;
        float *lse = a->lse + (size_t)l * T * H;

        layer_rms(xa, rstd_a, xl, y->rms_attn, T, C, eps);
        uenr_linear_fwd(g, qp, xa, y->wq, T, C, C);
        uenr_linear_fwd(g, kp, xa, y->wk, T, C, KV);
        uenr_linear_fwd(g, vv, xa, y->wv, T, C, KV);
        if (m->lora_on && m->lora_layers) {
            UenrLoraLayer *ll = &m->lora_layers[l];
            if (ll->has_q) lora_fwd(g, qp, xa, ll->Aq, ll->Bq, T, C, C, m->lora_rank, m->lora_scale, a->lora_h);
            if (ll->has_k) lora_fwd(g, kp, xa, ll->Ak, ll->Bk, T, C, KV, m->lora_rank, m->lora_scale, a->lora_h);
            if (ll->has_v) lora_fwd(g, vv, xa, ll->Av, ll->Bv, T, C, KV, m->lora_rank, m->lora_scale, a->lora_h);
        }
        uenr_rmsnorm_rope_fwd(q, rstd_q, qp, y->q_norm, &m->rope,
                              T, S, H, hd, eps);
        uenr_rmsnorm_rope_fwd(k, rstd_k, kp, y->k_norm, &m->rope,
                              T, S, G, hd, eps);

        if (a->flash)
            uenr_flash_attn_fwd(ao, lse, q, k, vv, B, S, H, G, hd, scale, g->pool);
        else
            uenr_attn_fwd(ao, probs, q, k, vv, B, S, H, G, hd, scale, g->pool);

        uenr_linear_fwd(g, a->tmp, ao, y->wo, T, C, C);
        if (m->lora_on && m->lora_layers && m->lora_layers[l].has_o)
            lora_fwd(g, a->tmp, ao, m->lora_layers[l].Ao, m->lora_layers[l].Bo,
                     T, C, C, m->lora_rank, m->lora_scale, a->lora_h);
        uenr_residual_rmsnorm_fwd(x2, xf, rstd_f, xl, a->tmp,
                                  y->rms_ffn, T, C, eps);

        /* router */
        uenr_linear_fwd(g, rp, xf, y->router, T, C, E);
        for (int t = 0; t < T; t++) {
            float *p = rp + (size_t)t * E;
            softmax_inplace(p, E);
            topk_select(p, E, K, ei + (size_t)t * K, eg + (size_t)t * K);
        }

        /* MoE / dense FFN */
        memset(a->eout, 0, TC * sizeof(float));
        if (m->cfg.moe.enabled) {
            memset(ec, 0, (size_t)E * sizeof(int));
            int total = T * K;
            for (int qx = 0; qx < total; qx++) ec[ei[qx]]++;
            a->eoff[0] = 0;
            for (int e = 0; e < E; e++) a->eoff[e + 1] = a->eoff[e] + ec[e];
            memcpy(a->cursor, a->eoff, (size_t)E * sizeof(int));
            for (int qx = 0; qx < total; qx++) {
                int e = ei[qx];
                a->elist[a->cursor[e]++] = qx;
            }
            for (int e = 0; e < E; e++) {
                int start = a->eoff[e], end = a->eoff[e + 1];
                int ne = end - start;
                if (ne <= 0) continue;
                for (int i = 0; i < ne; i++) {
                    int qx = a->elist[start + i];
                    int t = qx / K;
                    memcpy(a->gxs + (size_t)i * C, xf + (size_t)t * C, (size_t)C * sizeof(float));
                }
                const float *Wg = y->expert_wg + (size_t)e * EF * C;
                const float *Wu = y->expert_wu + (size_t)e * EF * C;
                const float *Wd = y->expert_wd + (size_t)e * C * EF;
                uenr_swiglu_fwd(g, ne, C, EF, a->gxs, Wg, Wu, Wd,
                                a->gys, a->sscratch, a->sscratch + (size_t)ne * EF);
                for (int i = 0; i < ne; i++) {
                    int qx = a->elist[start + i];
                    int t = qx / K;
                    float gate = eg[qx];
                    uenr_add_scaled(a->eout + (size_t)t * C, a->gys + (size_t)i * C, gate, C);
                    memcpy(ry + (size_t)qx * C, a->gys + (size_t)i * C, (size_t)C * sizeof(float));
                }
            }
            if (m->cfg.moe.shared_expert && y->shared_wg) {
                uenr_swiglu_fwd(g, T, C, FF, xf, y->shared_wg, y->shared_wu, y->shared_wd,
                                a->gys, a->sscratch, a->sscratch + (size_t)T * FF);
                uenr_add_scaled(a->eout, a->gys, 1.0f, TC);
            }
        } else {
            uenr_swiglu_fwd(g, T, C, FF, xf, y->shared_wg, y->shared_wu, y->shared_wd,
                            a->eout, a->sscratch, a->sscratch + (size_t)T * FF);
        }
        for (size_t i = 0; i < TC; i++) xo[i] = x2[i] + a->eout[i];
    }

    /* final norm + tied logits */
    const float *xlast = a->x + (size_t)L * TC;
    uenr_rmsnorm_fwd(a->xfinal, a->rstd_final, xlast, m->rms_final, T, C, eps);

    if (a->sampled && targets && write_grad) {
        a->sampled_last = 1;
        return sampled_softmax_loss(m, a, targets, T, V, ignore_index, valid_count);
    }
    a->sampled_last = 0;
    uenr_linear_fwd(g, a->logits, a->xfinal, m->lm_head, T, C, V);

    int count = 0;
    float loss = 0.0f;
    if (targets) {
        for (int t = 0; t < T; t++)
            if (targets[t] != ignore_index) count++;
        loss = uenr_softmax_ce_fwd(a->logits, targets, T, V, ignore_index,
                                   write_grad ? a->dlogits : NULL, write_grad, g->pool);
    }
    if (valid_count) *valid_count = count;
    return loss;
}

/* ------------------------------------------------------------------------- */
/* backward                                                                  */
/* ------------------------------------------------------------------------- */

void uenr_model_backward(UenrModel *m, UenrGemm *g, UenrActs *a) {
    if (!m || !g || !a) return;
    const int T = a->T_cur, S = a->S_cur, B = T / S;
    const int C = m->C, L = m->L, H = m->H, G = m->G, hd = m->hd, KV = m->KV;
    const int E = m->E, EF = m->EF, FF = m->FF, V = m->V, K = m->K;
    const float scale = uenr_rsqrtf((float)hd);
    const float aux_coef = m->cfg.moe.router_aux_coef;
    size_t TC = (size_t)T * C, TKV = (size_t)T * KV;
    const int *tokens = a->tokens_cur;

    float *gtok = grad_of(m, m->tok_emb);
    float *glm = grad_of(m, m->lm_head);

    if (a->sampled_last) {
        memset(a->dx, 0, TC * sizeof(float));
        const int sn = a->sn;
        for (int t = 0; t < T; t++) {
            int base = t * sn;
            if (a->sids[base] < 0) continue;
            const float *xr = a->xfinal + (size_t)t * C;
            float *dxr = a->dx + (size_t)t * C;
            for (int i = 0; i < sn; i++) {
                int id = a->sids[base + i];
                float d = a->slogits[base + i];
                const float *wr = m->lm_head + (size_t)id * C;
                for (int c = 0; c < C; c++) {
                    dxr[c] += d * wr[c];
                    glm[(size_t)id * C + c] += d * xr[c];
                }
            }
        }
        a->sampled_last = 0;
    } else {
        /* logits = xfinal @ lm_head^T */
        uenr_linear_dx(g, a->dx, a->dlogits, m->lm_head, T, C, V);
        uenr_linear_dw(g, glm, a->dlogits, a->xfinal, T, C, V, 1.0f);
    }

    /* final rmsnorm */
    {
        const float *xlast = a->x + (size_t)L * TC;
        float *grms = grad_of(m, m->rms_final);
        uenr_rmsnorm_bwd(a->tmp, grms, a->dx, xlast, m->rms_final, a->rstd_final, T, C);
        memcpy(a->dx, a->tmp, TC * sizeof(float));
    }

    for (int l = L - 1; l >= 0; l--) {
        UenrLayer *y = &m->layers[l];
        const float *xl = a->x + (size_t)l * TC;
        const float *xa = a->xa + (size_t)l * TC;
        const float *qp = a->q_pre + (size_t)l * TC;
        const float *q = a->q + (size_t)l * TC;
        const float *kp = a->k_pre + (size_t)l * TKV;
        const float *k = a->k + (size_t)l * TKV;
        const float *vv = a->v + (size_t)l * TKV;
        const float *ao = a->attn_out + (size_t)l * TC;
        const float *x2 = a->x2 + (size_t)l * TC;
        const float *xf = a->xf + (size_t)l * TC;
        const float *rp = a->rprobs + (size_t)l * T * E;
        const float *eg = a->egate + (size_t)l * T * K;
        const int   *ei = a->eidx + (size_t)l * T * K;
        const float *ry = a->routed_y + (size_t)l * T * K * C;
        int         *ec = a->ecount + (size_t)l * E;
        const float *rstd_a = a->rstd_attn + (size_t)l * T;
        const float *rstd_f = a->rstd_ffn + (size_t)l * T;
        const float *rstd_q = a->rstd_q + (size_t)l * T * H;
        const float *rstd_k = a->rstd_k + (size_t)l * T * G;
        const float *probs = a->flash ? NULL : a->probs + (size_t)l * B * H * S * S;
        const float *lse = a->lse + (size_t)l * T * H;

        /* ---- FFN / MoE ---- */
        memset(a->dxf, 0, TC * sizeof(float));
        memset(a->drouter, 0, (size_t)T * E * sizeof(float));

        if (m->cfg.moe.enabled) {
            int total = T * K;
            a->eoff[0] = 0;
            for (int e = 0; e < E; e++) a->eoff[e + 1] = a->eoff[e] + ec[e];
            memcpy(a->cursor, a->eoff, (size_t)E * sizeof(int));
            for (int qx = 0; qx < total; qx++) {
                int e = ei[qx];
                a->elist[a->cursor[e]++] = qx;
            }
            for (int e = 0; e < E; e++) {
                int start = a->eoff[e], end = a->eoff[e + 1];
                int ne = end - start;
                if (ne <= 0) continue;
                for (int i = 0; i < ne; i++) {
                    int qx = a->elist[start + i];
                    int t = qx / K;
                    memcpy(a->gxs + (size_t)i * C, xf + (size_t)t * C, (size_t)C * sizeof(float));
                    for (int c = 0; c < C; c++)
                        a->gdys[(size_t)i * C + c] = a->dx[(size_t)t * C + c] * eg[qx];
                }
                float *Wg = y->expert_wg + (size_t)e * EF * C;
                float *Wu = y->expert_wu + (size_t)e * EF * C;
                float *Wd = y->expert_wd + (size_t)e * C * EF;
                float *gWg = grad_of(m, Wg);
                float *gWu = grad_of(m, Wu);
                float *gWd = grad_of(m, Wd);
                uenr_swiglu_bwd(g, ne, C, EF, a->gdys, a->gxs, Wg, Wu, Wd,
                                a->gys, gWg, gWu, gWd, a->sscratch);
                for (int i = 0; i < ne; i++) {
                    int qx = a->elist[start + i];
                    int t = qx / K;
                    uenr_add_scaled(a->dxf + (size_t)t * C, a->gys + (size_t)i * C, 1.0f, C);
                    a->gate_dp[qx] = dotf(ry + (size_t)qx * C, a->dx + (size_t)t * C, C);
                }
            }
            if (m->cfg.moe.shared_expert && y->shared_wg) {
                float *gWg = grad_of(m, y->shared_wg);
                float *gWu = grad_of(m, y->shared_wu);
                float *gWd = grad_of(m, y->shared_wd);
                uenr_swiglu_bwd(g, T, C, FF, a->dx, xf, y->shared_wg, y->shared_wu,
                                y->shared_wd, a->gys, gWg, gWu, gWd, a->sscratch);
                uenr_add_scaled(a->dxf, a->gys, 1.0f, TC);
            }
        } else {
            float *gWg = grad_of(m, y->shared_wg);
            float *gWu = grad_of(m, y->shared_wu);
            float *gWd = grad_of(m, y->shared_wd);
            uenr_swiglu_bwd(g, T, C, FF, a->dx, xf, y->shared_wg, y->shared_wu,
                            y->shared_wd, a->gys, gWg, gWu, gWd, a->sscratch);
            uenr_add_scaled(a->dxf, a->gys, 1.0f, TC);
        }

        /* router backward (softmax over selected top-k) */
        {
            const float invT = 1.0f / (float)T;
            const float invN = 1.0f / (float)(T * K);
            for (int t = 0; t < T; t++) {
                const float *p = rp + (size_t)t * E;
                float denom = 0.0f;
                for (int kk = 0; kk < K; kk++) {
                    int qx = t * K + kk;
                    denom += eg[qx] * a->gate_dp[qx];
                }
                float fsum = 0.0f;
                if (aux_coef > 0.0f)
                    for (int e = 0; e < E; e++) fsum += ((float)ec[e] * invN) * p[e];
                for (int j = 0; j < E; j++) {
                    float sel = 0.0f;
                    for (int kk = 0; kk < K; kk++) {
                        int qx = t * K + kk;
                        if (ei[qx] == j) sel += a->gate_dp[qx];
                    }
                    float dl = p[j] * (sel - denom);
                    if (aux_coef > 0.0f)
                        dl += aux_coef * (float)E * invT * p[j] *
                              (((float)ec[j] * invN) - fsum);
                    a->drouter[(size_t)t * E + j] += dl;
                }
            }
            uenr_linear_dw(g, grad_of(m, y->router), a->drouter, xf, T, C, E, 1.0f);
            uenr_linear_dx(g, a->dxa, a->drouter, y->router, T, C, E);
            uenr_add_scaled(a->dxf, a->dxa, 1.0f, TC);
        }

        /* residual + rmsnorm_ffn backward: x2 -> xf */
        uenr_residual_rmsnorm_bwd(a->dx2, grad_of(m, y->rms_ffn), a->dxf,
                                  a->dx, x2, y->rms_ffn, rstd_f, T, C);

        /* ---- attention ---- */
        uenr_linear_dw(g, grad_of(m, y->wo), a->dx2, ao, T, C, C, 1.0f);
        uenr_linear_dx(g, a->dattn, a->dx2, y->wo, T, C, C);
        if (m->lora_on && m->lora_layers && m->lora_layers[l].has_o) {
            UenrLoraLayer *ll = &m->lora_layers[l];
            lora_bwd(g, a->dattn, m->lora_g + (ll->Ao - m->lora_p),
                     m->lora_g + (ll->Bo - m->lora_p), a->dx2, ao, ll->Ao, ll->Bo,
                     T, C, C, m->lora_rank, m->lora_scale, a->lora_h, a->lora_dh);
        }

        memset(a->dq, 0, TC * sizeof(float));
        memset(a->dk, 0, TKV * sizeof(float));
        memset(a->dv, 0, TKV * sizeof(float));
        if (a->flash)
            uenr_flash_attn_bwd(a->dq, a->dk, a->dv, a->dattn, q, k, vv, ao,
                                lse, B, S, H, G, hd, scale, g->pool);
        else
            uenr_attn_bwd(a->dq, a->dk, a->dv, a->dattn, q, k, vv, probs,
                          B, S, H, G, hd, scale, g->pool);
        uenr_rmsnorm_rope_bwd(a->dq, grad_of(m, y->q_norm), a->dq, qp,
                              y->q_norm, rstd_q, &m->rope, T, S, H, hd);
        uenr_rmsnorm_rope_bwd(a->dk, grad_of(m, y->k_norm), a->dk, kp,
                              y->k_norm, rstd_k, &m->rope, T, S, G, hd);

        memset(a->dxa, 0, TC * sizeof(float));
        uenr_linear_dw(g, grad_of(m, y->wq), a->dq, xa, T, C, C, 1.0f);
        uenr_linear_dx_acc(g, a->dxa, a->dq, y->wq, T, C, C, 1.0f);
        uenr_linear_dw(g, grad_of(m, y->wk), a->dk, xa, T, C, KV, 1.0f);
        uenr_linear_dx_acc(g, a->dxa, a->dk, y->wk, T, C, KV, 1.0f);
        uenr_linear_dw(g, grad_of(m, y->wv), a->dv, xa, T, C, KV, 1.0f);
        uenr_linear_dx_acc(g, a->dxa, a->dv, y->wv, T, C, KV, 1.0f);
        if (m->lora_on && m->lora_layers) {
            UenrLoraLayer *ll = &m->lora_layers[l];
            if (ll->has_q) lora_bwd(g, a->dxa, m->lora_g + (ll->Aq - m->lora_p),
                                    m->lora_g + (ll->Bq - m->lora_p), a->dq, xa,
                                    ll->Aq, ll->Bq, T, C, C, m->lora_rank, m->lora_scale,
                                    a->lora_h, a->lora_dh);
            if (ll->has_k) lora_bwd(g, a->dxa, m->lora_g + (ll->Ak - m->lora_p),
                                    m->lora_g + (ll->Bk - m->lora_p), a->dk, xa,
                                    ll->Ak, ll->Bk, T, C, KV, m->lora_rank, m->lora_scale,
                                    a->lora_h, a->lora_dh);
            if (ll->has_v) lora_bwd(g, a->dxa, m->lora_g + (ll->Av - m->lora_p),
                                    m->lora_g + (ll->Bv - m->lora_p), a->dv, xa,
                                    ll->Av, ll->Bv, T, C, KV, m->lora_rank, m->lora_scale,
                                    a->lora_h, a->lora_dh);
        }

        /* rmsnorm_attn backward: xl -> xa */
        uenr_rmsnorm_bwd(a->tmp, grad_of(m, y->rms_attn), a->dxa, xl,
                         y->rms_attn, rstd_a, T, C);
        memcpy(a->dx, a->dx2, TC * sizeof(float));
        uenr_add_scaled(a->dx, a->tmp, 1.0f, TC);
    }

    /* embedding scatter (tied head already accumulated into glm) */
    for (int t = 0; t < T; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= V) continue;
        uenr_add_scaled(gtok + (size_t)tok * C, a->dx + (size_t)t * C, 1.0f, C);
    }
}

/* ------------------------------------------------------------------------- */
/* optimizer                                                                 */
/* ------------------------------------------------------------------------- */

void uenr_model_zero_grad(UenrModel *m) {
    if (!m) return;
    if (m->grad) memset(m->grad, 0, m->alloc_floats * sizeof(float));
    if (m->lora_g) memset(m->lora_g, 0, m->lora_n * sizeof(float));
}

float uenr_model_grad_norm(UenrModel *m, UenrPool *pool) {
    if (!m || !m->grad) return 0.0f;
    float sum = 0.0f;
    if (!m->freeze_base) {
        for (int i = 0; i < m->n_tensors; i++) {
            const UenrTensor *t = &m->tensors[i];
            float n = uenr_l2_norm(m->grad + t->off, t->n);
            sum += n * n;
        }
    }
    if (m->lora_g) {
        float n = uenr_l2_norm_p(pool, m->lora_g, m->lora_n);
        sum += n * n;
    }
    return uenr_sqrtf(sum);
}

float uenr_model_clip_grad(UenrModel *m, float clip, UenrPool *pool) {
    float norm = uenr_model_grad_norm(m, pool);
    if (clip > 0.0f && norm > clip) {
        float s = clip * uenr_rcpf(norm);
        if (!m->freeze_base) {
            for (int i = 0; i < m->n_tensors; i++) {
                UenrTensor *t = &m->tensors[i];
                uenr_scale_p(pool, m->grad + t->off, t->n, s);
            }
        }
        if (m->lora_g) uenr_scale_p(pool, m->lora_g, m->lora_n, s);
    }
    return norm;
}

void uenr_model_adamw(UenrModel *m, int step, float lr, float b1, float b2,
                      float eps, float weight_decay, UenrPool *pool) {
    if (!m || !m->opt_ready) return;
    if (!m->freeze_base) {
        for (int i = 0; i < m->n_tensors; i++) {
            UenrTensor *t = &m->tensors[i];
            uenr_adamw_step(t->p, m->grad + t->off, m->adam_m + t->off,
                            m->adam_v + t->off, t->n, step, lr, b1, b2, eps,
                            t->no_decay ? 0.0f : weight_decay, pool);
        }
    }
    if (m->lora_on && m->lora_p)
        uenr_adamw_step(m->lora_p, m->lora_g, m->lora_m, m->lora_v, m->lora_n,
                        step, lr, b1, b2, eps, 0.0f, pool);
}

/* ------------------------------------------------------------------------- */
/* checkpoint                                                                */
/* ------------------------------------------------------------------------- */

#define UMDL_MAGIC 0x4c444d55u /* "UMDL" */

UenrResult uenr_model_save(UenrModel *m, const char *path, int step, int with_opt) {
    if (!m || !path) return UENR_ERR_INVALID;
    int has_opt = (with_opt && m->opt_ready) ? 1 : 0;
    size_t hdr = 20 * sizeof(uint32_t);
    size_t n = m->alloc_floats;
    size_t bytes = hdr + n * sizeof(float) * (1 + (has_opt ? 2 : 0));
    unsigned char *blob = (unsigned char *)malloc(bytes);
    if (!blob) return UENR_ERR_ALLOC;
    uint32_t *h = (uint32_t *)blob;
    h[0] = UMDL_MAGIC; h[1] = 1;
    h[2] = (uint32_t)m->V; h[3] = (uint32_t)m->C; h[4] = (uint32_t)m->L;
    h[5] = (uint32_t)m->H; h[6] = (uint32_t)m->G; h[7] = (uint32_t)m->hd;
    h[8] = (uint32_t)m->E; h[9] = (uint32_t)m->EF; h[10] = (uint32_t)m->FF;
    h[11] = (uint32_t)m->tie; h[12] = (uint32_t)m->K; h[13] = (uint32_t)step;
    h[14] = (uint32_t)has_opt; h[15] = 0;
    h[16] = (uint32_t)(n & 0xffffffffu);
    h[17] = (uint32_t)(n >> 32);
    h[18] = 0; h[19] = 0;
    size_t off = hdr;
    memcpy(blob + off, m->params, n * sizeof(float)); off += n * sizeof(float);
    if (has_opt) {
        memcpy(blob + off, m->adam_m, n * sizeof(float)); off += n * sizeof(float);
        memcpy(blob + off, m->adam_v, n * sizeof(float)); off += n * sizeof(float);
    }
    UenrResult r = uenr_write_file_atomic(path, blob, bytes);
    free(blob);
    return r;
}

UenrResult uenr_model_load(UenrModel *m, const char *path, int *step, int load_opt) {
    if (!m || !path) return UENR_ERR_INVALID;
    void *data = NULL;
    size_t size = 0;
    UENR_TRY(uenr_read_file(path, &data, &size));
    size_t hdr = 20 * sizeof(uint32_t);
    if (size < hdr) { free(data); return UENR_ERR_INVALID; }
    uint32_t *h = (uint32_t *)data;
    if (h[0] != UMDL_MAGIC || h[1] != 1) { free(data); return UENR_ERR_INVALID; }
    int V = (int)h[2], C = (int)h[3], L = (int)h[4], H = (int)h[5];
    int G = (int)h[6], hd = (int)h[7], E = (int)h[8], EF = (int)h[9];
    int FF = (int)h[10], tie = (int)h[11], K = (int)h[12];
    size_t n = (size_t)h[16] | ((size_t)h[17] << 32);
    int has_opt = (int)h[14];
    if (V != m->V || C != m->C || L != m->L || H != m->H || G != m->G ||
        hd != m->hd || E != m->E || EF != m->EF || FF != m->FF ||
        tie != m->tie || K != m->K || n != m->alloc_floats) {
        free(data);
        return UENR_ERR_SHAPE;
    }
    size_t need = hdr + n * sizeof(float) * (1 + (has_opt ? 2 : 0));
    if (size != need) { free(data); return UENR_ERR_INVALID; }
    size_t off = hdr;
    memcpy(m->params, (unsigned char *)data + off, n * sizeof(float)); off += n * sizeof(float);
    if (step) *step = (int)h[13];
    if (load_opt && has_opt) {
        if (uenr_model_opt_init(m, 1) != UENR_OK) { free(data); return UENR_ERR_ALLOC; }
        memcpy(m->adam_m, (unsigned char *)data + off, n * sizeof(float)); off += n * sizeof(float);
        memcpy(m->adam_v, (unsigned char *)data + off, n * sizeof(float));
    }
    free(data);
    return UENR_OK;
}
