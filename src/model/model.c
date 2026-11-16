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
    if (p) {
        memset(p, 0, n);
        uenr_madvise_huge(p, n);
    }
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
    /* Frozen base: no full-size gradient buffer exists. Every base-weight
     * gradient write lands in one bounded sink and is never read. */
    if (m->freeze_base) return m->grad_sink;
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
        if (y->has_k && b->wk)
            uenr_gemm(g, KV, C, r, y->Bk, r, 0, y->Ak, C, 0, b->wk, C, s, 1.0f);
        if (y->has_v && b->wv)
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
    uenr_huge_pages_enabled = cfg->hw.huge_pages;
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
    m->n_slots = uenr_active_kv_layers(cfg);
    for (int l = 0; l < m->L; l++) {
        UenrLayer *y = &m->layers[l];
        y->type = uenr_layer_is_sliding(cfg, l) ? 0 : 1;
        y->is_shared = uenr_layer_is_shared(cfg, l);
        y->donor = y->is_shared ? uenr_layer_donor(cfg, l) : l;
        y->slot = y->donor;
        y->rms_attn = tadd(m, (size_t)m->C, "rms_attn", 1);
        y->wq = tadd(m, (size_t)m->C * m->C, "wq", 0);
        y->wk = y->is_shared ? NULL : tadd(m, (size_t)m->KV * m->C, "wk", 0);
        y->wv = y->is_shared ? NULL : tadd(m, (size_t)m->KV * m->C, "wv", 0);
        y->wo = tadd(m, (size_t)m->C * m->C, "wo", 0);
        y->q_norm = tadd(m, (size_t)m->hd, "q_norm", 1);
        y->k_norm = y->is_shared ? NULL : tadd(m, (size_t)m->hd, "k_norm", 1);
        y->rms_ffn = tadd(m, (size_t)m->C, "rms_ffn", 1);
        y->ls = tadd(m, (size_t)m->C, "ls", 1);
        y->post_attn = tadd(m, (size_t)m->C, "post_attn", 1);
        y->post_ffn = tadd(m, (size_t)m->C, "post_ffn", 1);
        if (y->is_shared && cfg->model.kv_adapter) {
            y->ka = tadd(m, (size_t)m->KV, "ka", 1);
            y->kb = tadd(m, (size_t)m->KV, "kb", 1);
            y->va = tadd(m, (size_t)m->KV, "va", 1);
            y->vb = tadd(m, (size_t)m->KV, "vb", 1);
            if (!y->ka || !y->kb || !y->va || !y->vb) { uenr_model_free(m); return UENR_ERR_ALLOC; }
        }
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
        if (!y->rms_attn || !y->wq || (!y->is_shared && (!y->wk || !y->wv || !y->k_norm)) ||
            !y->wo || !y->q_norm || !y->rms_ffn || !y->ls || !y->post_attn ||
            !y->post_ffn || !y->router ||
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
        if (!t || !t->p) continue;
        if (t->no_decay) {
            for (size_t j = 0; j < t->n; j++) t->p[j] = 1.0f;
        } else {
            for (size_t j = 0; j < t->n; j++) t->p[j] = 0.02f * uenr_rng_normal(rng);
        }
    }
    for (int l = 0; l < m->L; l++) {
        UenrLayer *y = &m->layers[l];
        if (y->ka) { for (int i = 0; i < m->KV; i++) { y->ka[i] = 1.0f; y->va[i] = 1.0f; y->kb[i] = 0.0f; y->vb[i] = 0.0f; } }
    }
    float res_scale = 1.0f / uenr_sqrtf(2.0f * (float)m->L);
    float pni = cfg->model.post_norm_init > 0.0f ? cfg->model.post_norm_init : 0.0f;
    for (int l = 0; l < m->L; l++) {
        UenrLayer *y = &m->layers[l];
        if (y->post_attn) uenr_scale(y->post_attn, (size_t)m->C, pni);
        if (y->post_ffn) uenr_scale(y->post_ffn, (size_t)m->C, pni);
        uenr_scale(y->wo, (size_t)m->C * m->C, res_scale);
        if (y->expert_wd) uenr_scale(y->expert_wd, (size_t)m->E * m->C * m->EF, res_scale);
        if (y->shared_wd) uenr_scale(y->shared_wd, (size_t)m->C * m->FF, res_scale);
    }

    {
        float tl = cfg->model.rope_theta_local > 0.0f ? cfg->model.rope_theta_local
                                                      : cfg->model.rope_theta;
        float tg = cfg->model.rope_theta_global > 0.0f ? cfg->model.rope_theta_global
                                                       : cfg->model.rope_theta;
        float pl = cfg->model.rope_partial_local > 0.0f ? cfg->model.rope_partial_local : 1.0f;
        float pg = cfg->model.rope_partial_global > 0.0f ? cfg->model.rope_partial_global : 1.0f;
        if (uenr_rope_init_partial(&m->rope_local, cfg->model.max_seq, m->hd / 2, tl, pl,
                                   cfg->model.rope_scale, cfg->model.rope_yarn,
                                   cfg->model.rope_beta_fast, cfg->model.rope_beta_slow,
                                   cfg->model.rope_attn_factor) != UENR_OK ||
            uenr_rope_init_partial(&m->rope_global, cfg->model.max_seq, m->hd / 2, tg, pg,
                                   cfg->model.rope_scale, cfg->model.rope_yarn,
                                   cfg->model.rope_beta_fast, cfg->model.rope_beta_slow,
                                   cfg->model.rope_attn_factor) != UENR_OK) {
            uenr_model_free(m);
            return UENR_ERR_ALLOC;
        }
    }
    return UENR_OK;
}

void uenr_model_free(UenrModel *m) {
    if (!m) return;
    uenr_rope_destroy(&m->rope_local);
    uenr_rope_destroy(&m->rope_global);
    free(m->layers);
    free(m->params);
    free(m->grad);
    free(m->grad_sink);
    free(m->adam_m);
    free(m->adam_v);
    free(m->adam_m8);
    free(m->adam_ms);
    free(m->muon_m);
    free(m->ada); free(m->ada_rows); free(m->ada_cols);
    free(m->ada_roff); free(m->ada_coff);
    free(m->ema);
    uenr_model_free_q8(m);
    free(m->lora_layers);
    free(m->lora_p);
    free(m->lora_g);
    free(m->lora_m);
    free(m->lora_v);
    memset(m, 0, sizeof(*m));
}

UenrResult uenr_model_ema_init(UenrModel *m) {
    if (!m || !m->params) return UENR_ERR_INVALID;
    if (!m->ema) {
        m->ema = (float *)aligned_alloc(64, ((m->alloc_floats * sizeof(float)) + 63) & ~(size_t)63);
        if (!m->ema) return UENR_ERR_ALLOC;
    }
    memcpy(m->ema, m->params, m->alloc_floats * sizeof(float));
    return UENR_OK;
}

static void tensor_shape(const UenrModel *m, const UenrTensor *t, int *rows, int *cols);

void uenr_model_weight_qat(UenrModel *m, int bits) {
    if (!m || (bits != 4 && bits != 8)) return;
    for (int i = 0; i < m->n_tensors; i++) {
        UenrTensor *t = &m->tensors[i];
        if (t->no_decay) continue;
        int r, c;
        tensor_shape(m, t, &r, &c);
        int levels = bits == 4 ? 7 : 127;
        for (int rr = 0; rr < r; rr++) {
            float *row = t->p + (size_t)rr * c;
            float mx = 0.0f;
            for (int cc = 0; cc < c; cc++) { float a = row[cc] < 0 ? -row[cc] : row[cc]; if (a > mx) mx = a; }
            if (mx <= 0.0f) continue;
            float sc = mx / (float)levels;
            for (int cc = 0; cc < c; cc++) {
                int q = (int)(row[cc] / sc + (row[cc] >= 0 ? 0.5f : -0.5f));
                if (q > levels) q = levels;
                if (q < -levels) q = -levels;
                row[cc] = (float)q * sc;
            }
        }
    }
}

void uenr_model_ema_update(UenrModel *m, float decay) {
    if (!m || !m->ema) return;
    for (int i = 0; i < m->n_tensors; i++) {
        const UenrTensor *t = &m->tensors[i];
        for (size_t j = 0; j < t->n; j++)
            m->ema[t->off + j] = decay * m->ema[t->off + j] + (1.0f - decay) * t->p[j];
    }
}

static void tensor_shape(const UenrModel *m, const UenrTensor *t, int *rows, int *cols) {
    const int C = m->C, V = m->V, KV = m->KV, E = m->E, EF = m->EF, FF = m->FF;
    const char *n = t->name;
    if (!strcmp(n, "tok_emb") || !strcmp(n, "lm_head")) { *rows = V; *cols = C; }
    else if (!strcmp(n, "wq") || !strcmp(n, "wo")) { *rows = C; *cols = C; }
    else if (!strcmp(n, "wk") || !strcmp(n, "wv")) { *rows = KV; *cols = C; }
    else if (!strcmp(n, "router")) { *rows = E; *cols = C; }
    else if (!strcmp(n, "expert_wg") || !strcmp(n, "expert_wu")) { *rows = E * EF; *cols = C; }
    else if (!strcmp(n, "expert_wd")) { *rows = E * C; *cols = EF; }
    else if (!strcmp(n, "shared_wg") || !strcmp(n, "shared_wu")) { *rows = FF; *cols = C; }
    else if (!strcmp(n, "shared_wd")) { *rows = C; *cols = FF; }
    else { *rows = (int)t->n; *cols = 1; }
}

UenrResult uenr_model_opt_init(UenrModel *m, int with_adam) {
    if (!m || !m->params) return UENR_ERR_INVALID;
    if (m->freeze_base) {
        /* LoRA/QLoRA: base weights get no gradients and no optimizer state.
         * A tiny sink absorbs (and discards) frozen-base grad writes. */
        if (!m->grad_sink) {
            size_t maxn = 1;
            for (int i = 0; i < m->n_tensors; i++)
                if (m->tensors[i].n > maxn) maxn = m->tensors[i].n;
            m->grad_sink_n = maxn;
            m->grad_sink = alloc_f(maxn);
            if (!m->grad_sink) return UENR_ERR_ALLOC;
        }
        m->opt_ready = 1;
        return UENR_OK;
    }
    if (!m->grad) {
        m->grad = alloc_f(m->alloc_floats);
        if (!m->grad) return UENR_ERR_ALLOC;
    }
    m->optim8 = m->cfg.train.optim8 ? 1 : 0;
    m->muon = m->cfg.train.muon ? 1 : 0;
    m->adafactor = (!m->muon && m->cfg.train.adafactor) ? 1 : 0;
    if (with_adam && m->muon) {
        m->muon_m = alloc_f(m->alloc_floats);
        if (!m->muon_m) return UENR_ERR_ALLOC;
        m->opt_ready = 1;
        return UENR_OK;
    }
    if (with_adam && m->adafactor) {
        int nt = m->n_tensors;
        m->ada_rows = alloc_i(nt); m->ada_cols = alloc_i(nt);
        m->ada_roff = (size_t *)zalloc64((size_t)nt * sizeof(size_t));
        m->ada_coff = (size_t *)zalloc64((size_t)nt * sizeof(size_t));
        if (!m->ada_rows || !m->ada_cols || !m->ada_roff || !m->ada_coff)
            return UENR_ERR_ALLOC;
        size_t nr = 0, nc = 0, nv = 0;
        for (int i = 0; i < nt; i++) {
            int r, c;
            tensor_shape(m, &m->tensors[i], &r, &c);
            m->ada_rows[i] = r; m->ada_cols[i] = c;
            if (r > 1 && c > 1) { nr += r; nc += c; }
            else nv += m->tensors[i].n;
        }
        size_t nr_cur = 0, nc_cur = 0, nv_cur = 0;
        for (int i = 0; i < nt; i++) {
            int r = m->ada_rows[i], c = m->ada_cols[i];
            if (r > 1 && c > 1) {
                m->ada_roff[i] = nr_cur; m->ada_coff[i] = nr + nc_cur;
                nr_cur += (size_t)r; nc_cur += (size_t)c;
            } else {
                m->ada_roff[i] = nr + nc + nv_cur; m->ada_coff[i] = 0;
                nv_cur += m->tensors[i].n;
            }
        }
        m->ada_n = nr + nc + nv;
        m->ada = alloc_f(m->ada_n);
        if (!m->ada) return UENR_ERR_ALLOC;
        m->opt_ready = 1;
        return UENR_OK;
    }
    if (with_adam && m->optim8) {
        m->adam_nblocks = (m->alloc_floats + UENR_ADAM8_BLOCK - 1) / UENR_ADAM8_BLOCK;
        if (!m->adam_m8) {
            m->adam_m8 = (int8_t *)zalloc64(m->alloc_floats);
            m->adam_ms = alloc_f(m->adam_nblocks);
            m->adam_v = alloc_f(m->alloc_floats);
            if (!m->adam_m8 || !m->adam_ms || !m->adam_v)
                return UENR_ERR_ALLOC;
        }
        m->opt_ready = 1;
        return UENR_OK;
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
    a->recompute_attn = cfg->train.recompute_attn && !cfg->model.flash_attn;
    a->block = cfg->train.block_checkpoint && !cfg->model.flash_attn;
    a->fp16_acts = cfg->train.fp16_activations && !cfg->model.flash_attn;
    a->sampled = cfg->train.sampled_softmax;
    a->store_swiglu = !cfg->train.recompute_swiglu;
    a->sn = a->sampled ? cfg->train.sampled_negatives + 1 : 0;

    size_t T_ = (size_t)T, C = (size_t)a->C, L = (size_t)a->L, B = (size_t)a->B;
    size_t H = (size_t)a->H, G = (size_t)a->G, S_ = (size_t)S, KV = (size_t)a->KV;
    size_t E = (size_t)a->E, K = (size_t)a->K, EF = (size_t)a->EF, FF = (size_t)a->FF, V = (size_t)a->V;
    size_t TC = T_ * C, TKV = T_ * KV;
    /* Block checkpointing keeps one layer's saved activations; the forward is
     * re-run per layer during backward. K/V stays per layer because shared
     * layers borrow their donor's pre-projection tensors. */
    size_t LA = a->block ? 1 : L;

#define AF(field, count) do { a->field = alloc_f(count); if (!a->field) goto fail; } while (0)
#define AI(field, count) do { a->field = alloc_i(count); if (!a->field) goto fail; } while (0)

    AF(x, (L + 1) * TC);
    AF(xa, LA * TC);
    AF(q_pre, LA * TC);
    AF(q, LA * TC);
    AF(k_pre, LA * TKV);
    AF(k, L * TKV);
    AF(v, L * TKV);
    if (!a->flash && !a->fp16_acts)
        AF(probs, ((a->recompute_attn || a->block) ? 1 : L) * B * H * S_ * S_);
    if (!a->flash && a->fp16_acts) {
        size_t np = ((a->recompute_attn || a->block) ? 1 : L) * B * H * S_ * S_;
        a->probs16 = (uint16_t *)calloc(np, sizeof(uint16_t));
        if (!a->probs16) goto fail;
    }
    AF(lse, L * T_ * H);
    AF(attn_out, LA * TC);
    AF(x2, LA * TC);
    AF(xf, LA * TC);
    AF(rprobs, LA * T_ * E);
    AI(eidx, LA * T_ * K);
    AF(egate, LA * T_ * K);
    AF(routed_y, LA * T_ * K * C);
    AF(eout, TC);   /* single shared scratch */
    AI(ecount, LA * E);
    AI(eoff, E + 1);
    AI(elist, T_ * K + 1);
    AI(cursor, E + 1);
    AF(rstd_attn, LA * T_);
    AF(rstd_ffn, LA * T_);
    AF(rstd_q, LA * T_ * H);
    AF(rstd_k, LA * T_ * G);
    AF(rstd_v, LA * T_ * G);
    AF(v_raw, L * TKV);
    AF(rstd_final, T_);
    AF(xfinal, TC);
    AF(logits, T_ * V);
    if (cfg->model.final_softcap > 0.0f) {
        AF(dlogits, T_ * V);
    } else {
        a->dlogits = a->logits; /* loss overwrites logits in place (item 8) */
    }
    AF(dx, TC);
    AF(dxa, TC);
    AF(dattn, TC);
    AF(dq, TC);
    AF(dk, TKV);
    AF(dv, TKV);
    AF(dxf, TC);
    AF(dx2, TC);
    AF(drouter, T_ * E);
    AF(dk_acc, L * TKV);
    AF(dv_acc, L * TKV);
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
    if (a->store_swiglu) {
        AF(swg_r, LA * T_ * K * EF);
        AF(swu_r, LA * T_ * K * EF);
        AF(ls_proj, LA * TC);
    AF(ls_post, LA * TC);
    AF(ffn_out, LA * TC);
    AF(rstd_post_a, LA * T_);
    AF(rstd_post_f, LA * T_);
    AF(swg_s, LA * T_ * FF);
        AF(swu_s, L * T_ * FF);
        AF(swg_tmp, T_ * (EF > FF ? EF : FF));
        AF(swu_tmp, T_ * (EF > FF ? EF : FF));
    }
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
    free(a->k_pre); free(a->k); free(a->v); free(a->probs); free(a->probs16); free(a->lse);
    free(a->attn_out); free(a->x2); free(a->xf); free(a->rprobs);
    free(a->eidx); free(a->egate); free(a->routed_y); free(a->eout);
    free(a->ecount); free(a->eoff); free(a->elist); free(a->cursor);
    free(a->rstd_attn); free(a->rstd_ffn); free(a->rstd_q); free(a->rstd_k);
    free(a->rstd_v); free(a->v_raw);
    free(a->rstd_final); free(a->xfinal);
    if (a->dlogits != a->logits) free(a->dlogits);
    free(a->logits);
    a->dlogits = NULL;
    free(a->dx); free(a->dxa); free(a->dattn); free(a->dq); free(a->dk);
    free(a->dv); free(a->dxf); free(a->dx2); free(a->drouter); free(a->tmp);
    free(a->dk_acc); free(a->dv_acc);
    free(a->gate_dp); free(a->lora_h); free(a->lora_dh);
    free(a->slogits); free(a->sids);
    free(a->ls_proj); free(a->ls_post); free(a->ffn_out);
    free(a->rstd_post_a); free(a->rstd_post_f);
    free(a->swg_r); free(a->swu_r); free(a->swg_s); free(a->swu_s);
    free(a->swg_tmp); free(a->swu_tmp);
    free(a->gxs); free(a->gys); free(a->gdys); free(a->sscratch);
    memset(a, 0, sizeof(*a));
}

/* ------------------------------------------------------------------------- */
/* forward                                                                   */
/* ------------------------------------------------------------------------- */

static void kv_hadamard_trunc(float *v, int rows, int kv, int hd, int rank) {
    for (int t = 0; t < rows; t++) {
        float *row = v + (size_t)t * kv;
        for (int g = 0; g < kv / hd; g++) {
            float *h = row + (size_t)g * hd;
            for (int len = 1; len < hd; len <<= 1)
                for (int i = 0; i < hd; i += len << 1)
                    for (int j = i; j < i + len; j++) {
                        float a = h[j], b = h[j + len];
                        h[j] = a + b;
                        h[j + len] = a - b;
                    }
            float s = uenr_rsqrtf((float)hd);
            for (int d = 0; d < hd; d++) h[d] *= (d < rank ? s : 0.0f);
        }
    }
}

static void kv_qat_quant(float *v, int rows, int kv, int hd, int bits) {
    for (int t = 0; t < rows; t++) {
        float *row = v + (size_t)t * kv;
        for (int g = 0; g < kv / hd; g++) {
            float *h = row + (size_t)g * hd;
            float mx = 0.0f;
            for (int d = 0; d < hd; d++) {
                float a = h[d] < 0.0f ? -h[d] : h[d];
                if (a > mx) mx = a;
            }
            if (mx <= 0.0f) continue;
            if (bits >= 4) {
                float lvl = mx / 7.0f;
                for (int d = 0; d < hd; d++) {
                    float q = h[d] / lvl;
                    int iq = (int)(q + (q >= 0.0f ? 0.5f : -0.5f));
                    if (iq > 7) iq = 7;
                    if (iq < -7) iq = -7;
                    h[d] = (float)iq * lvl;
                }
            } else {
                float thr = 0.33f * mx;
                for (int d = 0; d < hd; d++) {
                    float x = h[d];
                    h[d] = (x > thr) ? mx : (x < -thr ? -mx : 0.0f);
                }
            }
        }
    }
}

static void layer_rms(float *dst, float *rstd, const float *src, const float *w,
                      int T, int D, float eps) {
    uenr_rmsnorm_fwd(dst, rstd, src, w, T, D, eps);
}

static inline void copy_row(float *dst, const float *src, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
    for (; i < n; i++) dst[i] = src[i];
}
static inline void copy_scaled_row(float *dst, const float *src, float s, int n) {
    int i = 0;
    __m256 sv = _mm256_set1_ps(s);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), sv));
    for (; i < n; i++) dst[i] = src[i] * s;
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

static void layer_forward(UenrModel *m, UenrGemm *g, UenrActs *a,
                          int l, int T, int S, int B) {
    const int C = m->C, H = m->H, G = m->G, hd = m->hd, KV = m->KV;
    const int E = m->E, EF = m->EF, FF = m->FF, K = m->K;
    const float eps = m->cfg.model.rms_eps;
    const float scale = m->cfg.model.attn_scale > 0.0f
                            ? m->cfg.model.attn_scale : uenr_rsqrtf((float)hd);
    const int SL = a->block ? 0 : l;   /* saved-activation slot */
    size_t TC = (size_t)T * C, TKV = (size_t)T * KV;

        UenrLayer *y = &m->layers[l];
        float *xl = a->x + (size_t)l * TC;
        float *xo = a->x + (size_t)(l + 1) * TC;
        float *xa = a->xa + (size_t)SL * TC;
        float *qp = a->q_pre + (size_t)SL * TC;
        float *q = a->q + (size_t)SL * TC;
        float *kp = a->k_pre + (size_t)SL * TKV;
        float *k = a->k + (size_t)y->donor * TKV;
        float *vv = a->v + (size_t)y->donor * TKV;
        if (y->is_shared && y->ka) {
            float *ko = a->k + (size_t)l * TKV;
            float *vo = a->v + (size_t)l * TKV;
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < KV; d++) {
                    ko[(size_t)t * KV + d] = y->ka[d] * k[(size_t)t * KV + d] + y->kb[d];
                    vo[(size_t)t * KV + d] = y->va[d] * vv[(size_t)t * KV + d] + y->vb[d];
                }
            }
            k = ko; vv = vo;
        }
        float *ao = a->attn_out + (size_t)SL * TC;
        float *x2 = a->x2 + (size_t)SL * TC;
        float *xf = a->xf + (size_t)SL * TC;
        float *rp = a->rprobs + (size_t)SL * T * E;
        float *eg = a->egate + (size_t)SL * T * K;
        int   *ei = a->eidx + (size_t)SL * T * K;
        float *ry = a->routed_y + (size_t)SL * T * K * C;
        int   *ec = a->ecount + (size_t)SL * E;
        float *rstd_a = a->rstd_attn + (size_t)SL * T;
        float *rstd_f = a->rstd_ffn + (size_t)SL * T;
        float *rstd_q = a->rstd_q + (size_t)SL * T * H;
        float *rstd_k = a->rstd_k + (size_t)SL * T * G;
        float *rstd_v = a->rstd_v + (size_t)SL * T * G;
        float *rstd_pa = a->rstd_post_a + (size_t)SL * T;
        float *rstd_pf = a->rstd_post_f + (size_t)SL * T;
        float *vraw = a->v_raw + (size_t)SL * TKV;
        float *probs = (a->flash || a->fp16_acts) ? NULL
                    : a->probs + (size_t)((a->recompute_attn || a->block) ? 0 : l) * B * H * S * S;
        uint16_t *probs16 = (a->flash || !a->fp16_acts) ? NULL
                    : a->probs16 + (size_t)((a->recompute_attn || a->block) ? 0 : l) * B * H * S * S;
        float *lse = a->lse + (size_t)l * T * H;

        layer_rms(xa, rstd_a, xl, y->rms_attn, T, C, eps);
        uenr_linear_fwd(g, qp, xa, y->wq, T, C, C);
        if (!y->is_shared) {
            uenr_linear_fwd(g, kp, xa, y->wk, T, C, KV);
            uenr_linear_fwd(g, vraw, xa, y->wv, T, C, KV);
        }
        if (m->lora_on && m->lora_layers) {
            UenrLoraLayer *ll = &m->lora_layers[l];
            if (ll->has_q) lora_fwd(g, qp, xa, ll->Aq, ll->Bq, T, C, C, m->lora_rank, m->lora_scale, a->lora_h);
            if (!y->is_shared && ll->has_k) lora_fwd(g, kp, xa, ll->Ak, ll->Bk, T, C, KV, m->lora_rank, m->lora_scale, a->lora_h);
            if (!y->is_shared && ll->has_v) lora_fwd(g, vraw, xa, ll->Av, ll->Bv, T, C, KV, m->lora_rank, m->lora_scale, a->lora_h);
        }
        const UenrRope *rope_l = y->type ? &m->rope_global : &m->rope_local;
        uenr_rmsnorm_rope_fwd(q, rstd_q, qp, y->q_norm, rope_l,
                              T, S, H, hd, eps);
        if (!y->is_shared) {
            uenr_rmsnorm_rope_fwd(k, rstd_k, kp, y->k_norm, rope_l,
                                  T, S, G, hd, eps);
            if (m->cfg.model.v_norm) {
                uenr_rmsnorm_noscale_fwd(vv, rstd_v, vraw, T * G, hd, eps);
            } else {
                memcpy(vv, vraw, TKV * sizeof(float));
            }
            if (m->cfg.train.kv_qat_bits > 0) {
                kv_qat_quant(k, T, KV, hd, m->cfg.train.kv_qat_bits);
                kv_qat_quant(vv, T, KV, hd, m->cfg.train.kv_qat_bits);
            }
            if (m->cfg.train.kv_lowrank > 0) {
                kv_hadamard_trunc(k, T, KV, hd, m->cfg.train.kv_lowrank);
                kv_hadamard_trunc(vv, T, KV, hd, m->cfg.train.kv_lowrank);
            }
        }

        int win = (y->type == 0) ? m->cfg.model.sliding_window : 0;
        if (a->flash)
            uenr_flash_attn_fwd(ao, lse, q, k, vv, B, S, H, G, hd, scale, win,
                                m->cfg.model.attn_softcap, g->mem_pool);
        else if (probs16)
            uenr_attn_fwd16(ao, probs16, q, k, vv, B, S, H, G, hd, scale, win,
                            m->cfg.model.attn_softcap, g->mem_pool);
        else
            uenr_attn_fwd(ao, probs, q, k, vv, B, S, H, G, hd, scale, win,
                          m->cfg.model.attn_softcap, g->mem_pool);

        uenr_linear_fwd(g, a->tmp, ao, y->wo, T, C, C);
        if (m->lora_on && m->lora_layers && m->lora_layers[l].has_o)
            lora_fwd(g, a->tmp, ao, m->lora_layers[l].Ao, m->lora_layers[l].Bo,
                     T, C, C, m->lora_rank, m->lora_scale, a->lora_h);
        {
            float *lsp = a->ls_proj + (size_t)SL * TC;
            float *lpost = a->ls_post + (size_t)SL * TC;
            memcpy(lsp, a->tmp, TC * sizeof(float));
            layer_rms(a->tmp, rstd_pa, lsp, y->post_attn, T, C, eps);
            memcpy(lpost, a->tmp, TC * sizeof(float));
            for (int t = 0; t < T; t++) {
                float *row = a->tmp + (size_t)t * C;
                float *x2r = x2 + (size_t)t * C;
                for (int c = 0; c < C; c++) x2r[c] = xl[(size_t)t * C + c] + y->ls[c] * row[c];
            }
            layer_rms(xf, rstd_f, x2, y->rms_ffn, T, C, eps);
        }

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
                    int t = (K == 1) ? qx : qx / K;
                    copy_row(a->gxs + (size_t)i * C, xf + (size_t)t * C, C);
                }
                const float *Wg = y->expert_wg + (size_t)e * EF * C;
                const float *Wu = y->expert_wu + (size_t)e * EF * C;
                const float *Wd = y->expert_wd + (size_t)e * C * EF;
                if (a->store_swiglu) {
                    uenr_swiglu_fwd_save(g, ne, C, EF, a->gxs, Wg, Wu, Wd,
                                         a->gys, a->swg_tmp, a->swu_tmp, a->sscratch);
                    for (int i = 0; i < ne; i++) {
                        int qx = a->elist[start + i];
                        memcpy(a->swg_r + ((size_t)SL * T * K + qx) * EF,
                               a->swg_tmp + (size_t)i * EF, (size_t)EF * sizeof(float));
                        memcpy(a->swu_r + ((size_t)SL * T * K + qx) * EF,
                               a->swu_tmp + (size_t)i * EF, (size_t)EF * sizeof(float));
                    }
                } else {
                    uenr_swiglu_fwd(g, ne, C, EF, a->gxs, Wg, Wu, Wd,
                                    a->gys, a->sscratch, a->sscratch + (size_t)ne * EF);
                }
                for (int i = 0; i < ne; i++) {
                    int qx = a->elist[start + i];
                    int t = (K == 1) ? qx : qx / K;
                    float gate = eg[qx];
                    uenr_add_scaled(a->eout + (size_t)t * C, a->gys + (size_t)i * C, gate, C);
                    copy_row(ry + (size_t)qx * C, a->gys + (size_t)i * C, C);
                }
            }
            if (m->cfg.moe.shared_expert && y->shared_wg) {
                if (a->store_swiglu) {
                    uenr_swiglu_fwd_save(g, T, C, FF, xf, y->shared_wg, y->shared_wu,
                                         y->shared_wd, a->gys,
                                         a->swg_s + (size_t)SL * T * FF,
                                         a->swu_s + (size_t)SL * T * FF, a->sscratch);
                } else {
                    uenr_swiglu_fwd(g, T, C, FF, xf, y->shared_wg, y->shared_wu, y->shared_wd,
                                    a->gys, a->sscratch, a->sscratch + (size_t)T * FF);
                }
                uenr_add_scaled(a->eout, a->gys, 1.0f, TC);
            }
        } else {
            if (a->store_swiglu) {
                uenr_swiglu_fwd_save(g, T, C, FF, xf, y->shared_wg, y->shared_wu,
                                     y->shared_wd, a->eout,
                                     a->swg_s + (size_t)SL * T * FF,
                                     a->swu_s + (size_t)SL * T * FF, a->sscratch);
            } else {
                uenr_swiglu_fwd(g, T, C, FF, xf, y->shared_wg, y->shared_wu, y->shared_wd,
                                a->eout, a->sscratch, a->sscratch + (size_t)T * FF);
            }
        }
        {
            float *raw = a->ffn_out + (size_t)SL * TC;
            memcpy(raw, a->eout, TC * sizeof(float));
            layer_rms(a->eout, rstd_pf, raw, y->post_ffn, T, C, eps);
            for (size_t i = 0; i < TC; i++) xo[i] = x2[i] + a->eout[i];
        }
}


float uenr_model_forward(UenrModel *m, UenrGemm *g, UenrActs *a,
                         const int *tokens, int T, int S,
                         const int *targets, int ignore_index,
                         int write_grad, int *valid_count) {
    if (!m || !g || !a || !tokens || T <= 0 || S <= 0 || T % S != 0) return 0.0f;
    const int C = m->C, L = m->L, V = m->V;
    const int B = T / S;
    const float eps = m->cfg.model.rms_eps;
    a->T_cur = T; a->S_cur = S; a->tokens_cur = tokens;

    size_t TC = (size_t)T * C;

    /* embeddings */
    for (int t = 0; t < T; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= V) tok = 0;
        memcpy(a->x + (size_t)t * C, m->tok_emb + (size_t)tok * C, (size_t)C * sizeof(float));
    }

    for (int l = 0; l < L; l++) {
        layer_forward(m, g, a, l, T, S, B);
    }

    /* final norm + tied logits */
    const float *xlast = a->x + (size_t)L * TC;
    uenr_rmsnorm_fwd(a->xfinal, a->rstd_final, xlast, m->rms_final, T, C, eps);

    if (m->cfg.model.final_softcap > 0.0f) {
        const float sc = m->cfg.model.final_softcap;
        for (size_t i = 0; i < (size_t)T * V; i++) {
            float z = a->logits[i];
            if (z > 0.1f * sc || z < -0.1f * sc)
                a->logits[i] = sc * uenr_tanhf(z / sc);
        }
    }

    if (a->sampled && targets && write_grad && m->cfg.model.final_softcap <= 0.0f) {
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
                                   write_grad ? a->dlogits : NULL, write_grad, g->mem_pool,
                                   m->cfg.train.z_loss);
        if (write_grad && m->cfg.model.final_softcap > 0.0f) {
            const float sc = m->cfg.model.final_softcap;
            size_t n = (size_t)T * V;
            for (size_t i = 0; i < n; i++) {
                float z = a->logits[i];
                if (z > 0.1f * sc || z < -0.1f * sc) {
                    float t = z / sc;
                    a->dlogits[i] *= 1.0f - t * t;
                }
            }
        }
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
    const float scale = m->cfg.model.attn_scale > 0.0f
                            ? m->cfg.model.attn_scale : uenr_rsqrtf((float)hd);
    const float aux_coef = m->cfg.moe.router_aux_coef;
    size_t TC = (size_t)T * C, TKV = (size_t)T * KV;
    const int *tokens = a->tokens_cur;

    memset(a->dk_acc, 0, (size_t)L * TKV * sizeof(float));
    memset(a->dv_acc, 0, (size_t)L * TKV * sizeof(float));
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
        const int SL = a->block ? 0 : l;   /* saved-activation slot */
        if (a->block) {
            /* Block checkpointing: no layer intermediates are persisted except
             * the residual stream. Recompute this layer's forward into the
             * shared slot immediately before its backward. */
            layer_forward(m, g, a, l, T, S, B);
        }
        const float *xl = a->x + (size_t)l * TC;
        const float *xa = a->xa + (size_t)SL * TC;
        const float *qp = a->q_pre + (size_t)SL * TC;
        const float *q = a->q + (size_t)SL * TC;
        const float *kp = a->k_pre + (size_t)SL * TKV;
        const float *k = (y->is_shared && y->ka)
                             ? a->k + (size_t)l * TKV
                             : a->k + (size_t)y->donor * TKV;
        const float *vv = (y->is_shared && y->ka)
                              ? a->v + (size_t)l * TKV
                              : a->v + (size_t)y->donor * TKV;
        const float *ao = a->attn_out + (size_t)SL * TC;
        const float *x2 = a->x2 + (size_t)SL * TC;
        const float *xf = a->xf + (size_t)SL * TC;
        const float *rp = a->rprobs + (size_t)SL * T * E;
        const float *eg = a->egate + (size_t)SL * T * K;
        const int   *ei = a->eidx + (size_t)SL * T * K;
        const float *ry = a->routed_y + (size_t)SL * T * K * C;
        int         *ec = a->ecount + (size_t)SL * E;
        const float *rstd_a = a->rstd_attn + (size_t)SL * T;
        const float *rstd_f = a->rstd_ffn + (size_t)SL * T;
        const float *rstd_q = a->rstd_q + (size_t)SL * T * H;
        const float *rstd_k = a->rstd_k + (size_t)SL * T * G;
        const float *rstd_v = a->rstd_v + (size_t)SL * T * G;
        const float *rstd_pa = a->rstd_post_a + (size_t)SL * T;
        const float *rstd_pf = a->rstd_post_f + (size_t)SL * T;
        const float *lpost = a->ls_post + (size_t)SL * TC;
        const float *vraw = a->v_raw + (size_t)SL * TKV;
        const float *probs = (a->flash || a->fp16_acts) ? NULL
                    : a->probs + (size_t)((a->recompute_attn || a->block) ? 0 : l) * B * H * S * S;
        const uint16_t *probs16 = (a->flash || !a->fp16_acts) ? NULL
                    : a->probs16 + (size_t)((a->recompute_attn || a->block) ? 0 : l) * B * H * S * S;
        const float *lse = a->lse + (size_t)l * T * H;

        /* ---- FFN / MoE ---- */
        float *df = a->dattn;
        uenr_rmsnorm_bwd(df, grad_of(m, y->post_ffn), a->dx,
                         a->ffn_out + (size_t)SL * TC, y->post_ffn, rstd_pf, T, C);
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
                    int t = (K == 1) ? qx : qx / K;
                    copy_row(a->gxs + (size_t)i * C, xf + (size_t)t * C, C);
                    for (int c = 0; c < C; c++)
                        a->gdys[(size_t)i * C + c] = df[(size_t)t * C + c] * eg[qx];
                }
                float *Wg = y->expert_wg + (size_t)e * EF * C;
                float *Wu = y->expert_wu + (size_t)e * EF * C;
                float *Wd = y->expert_wd + (size_t)e * C * EF;
                float *gWg = grad_of(m, Wg);
                float *gWu = grad_of(m, Wu);
                float *gWd = grad_of(m, Wd);
                if (a->store_swiglu) {
                    for (int i = 0; i < ne; i++) {
                        int qx = a->elist[start + i];
                        memcpy(a->swg_tmp + (size_t)i * EF,
                               a->swg_r + ((size_t)SL * T * K + qx) * EF,
                               (size_t)EF * sizeof(float));
                        memcpy(a->swu_tmp + (size_t)i * EF,
                               a->swu_r + ((size_t)SL * T * K + qx) * EF,
                               (size_t)EF * sizeof(float));
                    }
                    uenr_swiglu_bwd_saved(g, ne, C, EF, a->gdys, a->gxs, Wg, Wu, Wd,
                                          a->swg_tmp, a->swu_tmp,
                                          a->gys, gWg, gWu, gWd, a->sscratch);
                } else {
                    uenr_swiglu_bwd(g, ne, C, EF, a->gdys, a->gxs, Wg, Wu, Wd,
                                    a->gys, gWg, gWu, gWd, a->sscratch);
                }
                for (int i = 0; i < ne; i++) {
                    int qx = a->elist[start + i];
                    int t = (K == 1) ? qx : qx / K;
                    uenr_add_scaled(a->dxf + (size_t)t * C, a->gys + (size_t)i * C, 1.0f, C);
                    a->gate_dp[qx] = dotf(ry + (size_t)qx * C, df + (size_t)t * C, C);
                }
            }
            if (m->cfg.moe.shared_expert && y->shared_wg) {
                float *gWg = grad_of(m, y->shared_wg);
                float *gWu = grad_of(m, y->shared_wu);
                float *gWd = grad_of(m, y->shared_wd);
                if (a->store_swiglu) {
                    uenr_swiglu_bwd_saved(g, T, C, FF, df, xf, y->shared_wg,
                                          y->shared_wu, y->shared_wd,
                                          a->swg_s + (size_t)SL * T * FF,
                                          a->swu_s + (size_t)SL * T * FF,
                                          a->gys, gWg, gWu, gWd, a->sscratch);
                } else {
                    uenr_swiglu_bwd(g, T, C, FF, df, xf, y->shared_wg, y->shared_wu,
                                    y->shared_wd, a->gys, gWg, gWu, gWd, a->sscratch);
                }
                uenr_add_scaled(a->dxf, a->gys, 1.0f, TC);
            }
        } else {
            float *gWg = grad_of(m, y->shared_wg);
            float *gWu = grad_of(m, y->shared_wu);
            float *gWd = grad_of(m, y->shared_wd);
            if (a->store_swiglu) {
                uenr_swiglu_bwd_saved(g, T, C, FF, df, xf, y->shared_wg,
                                      y->shared_wu, y->shared_wd,
                                      a->swg_s + (size_t)SL * T * FF,
                                      a->swu_s + (size_t)SL * T * FF,
                                      a->gys, gWg, gWu, gWd, a->sscratch);
            } else {
                uenr_swiglu_bwd(g, T, C, FF, df, xf, y->shared_wg, y->shared_wu,
                                y->shared_wd, a->gys, gWg, gWu, gWd, a->sscratch);
            }
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
        const int win = (y->type == 0) ? m->cfg.model.sliding_window : 0;
        {
            float *gls = grad_of(m, y->ls);
            const float *raw = a->ls_proj + (size_t)SL * TC;
            for (int t = 0; t < T; t++) {
                const float *src = a->dx2 + (size_t)t * C;
                const float *pa = lpost + (size_t)t * C;
                float *dst = a->tmp + (size_t)t * C;
                for (int c = 0; c < C; c++) {
                    gls[c] += src[c] * pa[c];
                    dst[c] = src[c] * y->ls[c];
                }
            }
            uenr_rmsnorm_bwd(a->dattn, grad_of(m, y->post_attn), a->tmp,
                             raw, y->post_attn, rstd_pa, T, C);
        }
        uenr_linear_dw(g, grad_of(m, y->wo), a->dattn, ao, T, C, C, 1.0f);
        uenr_linear_dx(g, a->tmp, a->dattn, y->wo, T, C, C);
        if (m->lora_on && m->lora_layers && m->lora_layers[l].has_o) {
            UenrLoraLayer *ll = &m->lora_layers[l];
            lora_bwd(g, a->tmp, m->lora_g + (ll->Ao - m->lora_p),
                     m->lora_g + (ll->Bo - m->lora_p), a->dattn, ao, ll->Ao, ll->Bo,
                     T, C, C, m->lora_rank, m->lora_scale, a->lora_h, a->lora_dh);
        }

        if (a->recompute_attn && !a->block) {
            /* Recompute this layer's attention probabilities into the single
             * shared buffer; q/k/v are still saved per layer. */
            if (a->fp16_acts)
                uenr_attn_fwd16(a->eout, a->probs16, q, k, vv, B, S, H, G, hd,
                                scale, win, m->cfg.model.attn_softcap, g->mem_pool);
            else
                uenr_attn_fwd(a->eout, a->probs, q, k, vv, B, S, H, G, hd, scale,
                              win, m->cfg.model.attn_softcap, g->mem_pool);
        }
        memset(a->dq, 0, TC * sizeof(float));
        memset(a->dk, 0, TKV * sizeof(float));
        memset(a->dv, 0, TKV * sizeof(float));
        if (a->flash)
            uenr_flash_attn_bwd(a->dq, a->dk, a->dv, a->tmp, q, k, vv, ao,
                                lse, B, S, H, G, hd, scale, win,
                                m->cfg.model.attn_softcap, g->mem_pool);
        else if (probs16)
            uenr_attn_bwd16(a->dq, a->dk, a->dv, a->tmp, q, k, vv, probs16,
                            B, S, H, G, hd, scale, win,
                            m->cfg.model.attn_softcap, g->mem_pool);
        else
            uenr_attn_bwd(a->dq, a->dk, a->dv, a->tmp, q, k, vv, probs,
                          B, S, H, G, hd, scale, win,
                          m->cfg.model.attn_softcap, g->mem_pool);
        const UenrRope *rope_l = y->type ? &m->rope_global : &m->rope_local;
        uenr_rmsnorm_rope_bwd(a->dq, grad_of(m, y->q_norm), a->dq, qp,
                              y->q_norm, rstd_q, rope_l, T, S, H, hd);
        if (y->is_shared) {
            /* Donor gradient accumulation: attn_bwd produced dK/dV for the
             * borrowed (possibly adapted) tensors. */
            if (y->ka) {
                float *gka = grad_of(m, y->ka), *gkb = grad_of(m, y->kb);
                float *gva = grad_of(m, y->va), *gvb = grad_of(m, y->vb);
                const float *dk0 = a->k + (size_t)y->donor * TKV;
                const float *dv0 = a->v + (size_t)y->donor * TKV;
                for (int t = 0; t < T; t++) {
                    for (int d = 0; d < KV; d++) {
                        size_t i = (size_t)t * KV + d;
                        gka[d] += a->dk[i] * dk0[i];
                        gkb[d] += a->dk[i];
                        gva[d] += a->dv[i] * dv0[i];
                        gvb[d] += a->dv[i];
                        a->dk[i] *= y->ka[d];
                        a->dv[i] *= y->va[d];
                    }
                }
            }
            float *ak = a->dk_acc + (size_t)y->donor * TKV;
            float *av = a->dv_acc + (size_t)y->donor * TKV;
            for (size_t i = 0; i < TKV; i++) { ak[i] += a->dk[i]; av[i] += a->dv[i]; }
        } else {
            const float *ak = a->dk_acc + (size_t)l * TKV;
            const float *av = a->dv_acc + (size_t)l * TKV;
            for (size_t i = 0; i < TKV; i++) { a->dk[i] += ak[i]; a->dv[i] += av[i]; }
            uenr_rmsnorm_rope_bwd(a->dk, grad_of(m, y->k_norm), a->dk, kp,
                                  y->k_norm, rstd_k, rope_l, T, S, G, hd);
            if (m->cfg.model.v_norm)
                uenr_rmsnorm_noscale_bwd(a->dv, a->dv, vraw, rstd_v, T * G, hd);
        }

        memset(a->dxa, 0, TC * sizeof(float));
        uenr_linear_dw(g, grad_of(m, y->wq), a->dq, xa, T, C, C, 1.0f);
        uenr_linear_dx_acc(g, a->dxa, a->dq, y->wq, T, C, C, 1.0f);
        if (!y->is_shared) {
            uenr_linear_dw(g, grad_of(m, y->wk), a->dk, xa, T, C, KV, 1.0f);
            uenr_linear_dx_acc(g, a->dxa, a->dk, y->wk, T, C, KV, 1.0f);
            uenr_linear_dw(g, grad_of(m, y->wv), a->dv, xa, T, C, KV, 1.0f);
            uenr_linear_dx_acc(g, a->dxa, a->dv, y->wv, T, C, KV, 1.0f);
        }
        if (m->lora_on && m->lora_layers) {
            UenrLoraLayer *ll = &m->lora_layers[l];
            if (ll->has_q) lora_bwd(g, a->dxa, m->lora_g + (ll->Aq - m->lora_p),
                                    m->lora_g + (ll->Bq - m->lora_p), a->dq, xa,
                                    ll->Aq, ll->Bq, T, C, C, m->lora_rank, m->lora_scale,
                                    a->lora_h, a->lora_dh);
            if (!y->is_shared && ll->has_k) lora_bwd(g, a->dxa, m->lora_g + (ll->Ak - m->lora_p),
                                    m->lora_g + (ll->Bk - m->lora_p), a->dk, xa,
                                    ll->Ak, ll->Bk, T, C, KV, m->lora_rank, m->lora_scale,
                                    a->lora_h, a->lora_dh);
            if (!y->is_shared && ll->has_v) lora_bwd(g, a->dxa, m->lora_g + (ll->Av - m->lora_p),
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
    if (m->grad_sink) memset(m->grad_sink, 0, m->grad_sink_n * sizeof(float));
    if (m->lora_g) memset(m->lora_g, 0, m->lora_n * sizeof(float));
}

float uenr_model_grad_norm(UenrModel *m, UenrPool *pool) {
    if (!m) return 0.0f;
    float sum = 0.0f;
    if (!m->freeze_base && m->grad) {
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
        if (!m->freeze_base && m->grad)
            uenr_scale_p(pool, m->grad, m->alloc_floats, s);
        if (m->lora_g) uenr_scale_p(pool, m->lora_g, m->lora_n, s);
    }
    return norm;
}

static void adafactor_tensor(UenrModel *m, UenrTensor *t, int i, int step,
                             float lr, float b2, float eps) {
    int r = m->ada_rows[i], c = m->ada_cols[i];
    float *p = t->p, *g = m->grad + t->off;
    if (r <= 1 || c <= 1) {
        float *v = m->ada + m->ada_roff[i];
        float bc2 = 1.0f - uenr_expf((float)step * uenr_logf(b2));
        for (size_t k = 0; k < t->n; k++) {
            v[k] = b2 * v[k] + (1.0f - b2) * g[k] * g[k];
            p[k] -= lr * g[k] / (uenr_sqrtf(v[k] / bc2) + eps);
        }
        return;
    }
    float *rv = m->ada + m->ada_roff[i];
    float *colv = m->ada + m->ada_coff[i];
    float bc2 = 1.0f - uenr_expf((float)step * uenr_logf(b2));
    for (int rr = 0; rr < r; rr++) {
        const float *gr = g + (size_t)rr * c;
        float s = 0.0f;
        for (int cc = 0; cc < c; cc++) s += gr[cc] * gr[cc];
        rv[rr] = b2 * rv[rr] + (1.0f - b2) * s / (float)c;
    }
    for (int cc = 0; cc < c; cc++) {
        float s = 0.0f;
        for (int rr = 0; rr < r; rr++) { float x = g[(size_t)rr * c + cc]; s += x * x; }
        colv[cc] = b2 * colv[cc] + (1.0f - b2) * s / (float)r;
    }
    float mr = 0.0f;
    for (int rr = 0; rr < r; rr++) mr += rv[rr];
    mr = mr / (float)r + 1e-30f;
    for (int rr = 0; rr < r; rr++) {
        for (int cc = 0; cc < c; cc++) {
            size_t k = (size_t)rr * c + cc;
            float vh = rv[rr] * colv[cc] / mr;
            float upd = g[k] / (uenr_sqrtf(vh / bc2) + eps);
            p[k] -= lr * upd;
        }
    }
}

static void muon_tensor(UenrModel *m, UenrTensor *t, int i, int step, float lr) {
    int r = 1, c = 1;
    tensor_shape(m, t, &r, &c);
    float *p = t->p, *g = m->grad + t->off, *mm = m->muon_m + t->off;
    const float mom = 0.9f;
    for (size_t k = 0; k < t->n; k++) mm[k] = mom * mm[k] + g[k];
    if (r <= 1 || c <= 1) {
        float ss = 0.0f;
        for (size_t k = 0; k < t->n; k++) ss += mm[k] * mm[k];
        float inv = uenr_rcpf(uenr_sqrtf(ss / (float)t->n) + 1e-6f);
        for (size_t k = 0; k < t->n; k++) p[k] -= lr * mm[k] * inv;
    } else {
        for (int rr = 0; rr < r; rr++) {
            float *row = mm + (size_t)rr * c;
            float ss = 0.0f;
            for (int cc = 0; cc < c; cc++) ss += row[cc] * row[cc];
            float inv = uenr_rcpf(uenr_sqrtf(ss) + 1e-6f);
            for (int cc = 0; cc < c; cc++) p[(size_t)rr * c + cc] -= lr * row[cc] * inv;
        }
    }
}

void uenr_model_adamw(UenrModel *m, int step, float lr, float b1, float b2,
                      float eps, float weight_decay, UenrPool *pool) {
    if (!m || !m->opt_ready) return;
    if (!m->freeze_base && m->muon) {
        for (int i = 0; i < m->n_tensors; i++)
            muon_tensor(m, &m->tensors[i], i, step, lr);
    } else if (!m->freeze_base && m->adafactor) {
        for (int i = 0; i < m->n_tensors; i++)
            adafactor_tensor(m, &m->tensors[i], i, step, lr, b2, eps);
    } else if (!m->freeze_base && m->optim8) {
        if (weight_decay > 0.0f) {
            for (int i = 0; i < m->n_tensors; i++) {
                UenrTensor *t = &m->tensors[i];
                if (!t->no_decay) uenr_scale(t->p, t->n, 1.0f - lr * weight_decay);
            }
        }
        uenr_adamw_step_m8(m->params, m->grad, m->adam_m8, m->adam_ms,
                           m->adam_v, m->alloc_floats, step, lr, b1, b2,
                           eps, 0.0f);
    } else if (!m->freeze_base) {
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
    size_t hdr = 20 * sizeof(uint32_t);
    size_t n = m->alloc_floats;
    /* v2 checkpoints persist whichever optimizer state is in use, so an
     * interrupted run resumes with warm moments, not just warm weights. */
    int kind = 0;
    size_t extra = 0;
    if (with_opt && m->opt_ready && !m->freeze_base) {
        if (m->optim8 && m->adam_m8 && m->adam_ms && m->adam_v) {
            kind = 4;
            extra = n + m->adam_nblocks * sizeof(float) + n * sizeof(float);
        } else if (m->adafactor && m->ada) {
            kind = 2;
            extra = m->ada_n * sizeof(float);
        } else if (m->muon && m->muon_m) {
            kind = 3;
            extra = n * sizeof(float);
        } else if (m->adam_m && m->adam_v) {
            kind = 1;
            extra = 2 * n * sizeof(float);
        }
    }
    size_t bytes = hdr + n * sizeof(float) + extra;
    unsigned char *blob = (unsigned char *)malloc(bytes);
    if (!blob) return UENR_ERR_ALLOC;
    uint32_t *h = (uint32_t *)blob;
    h[0] = UMDL_MAGIC; h[1] = 2;
    h[2] = (uint32_t)m->V; h[3] = (uint32_t)m->C; h[4] = (uint32_t)m->L;
    h[5] = (uint32_t)m->H; h[6] = (uint32_t)m->G; h[7] = (uint32_t)m->hd;
    h[8] = (uint32_t)m->E; h[9] = (uint32_t)m->EF; h[10] = (uint32_t)m->FF;
    h[11] = (uint32_t)m->tie; h[12] = (uint32_t)m->K; h[13] = (uint32_t)step;
    h[14] = (uint32_t)(kind ? 1 : 0); h[15] = (uint32_t)kind;
    h[16] = (uint32_t)(n & 0xffffffffu);
    h[17] = (uint32_t)(n >> 32);
    h[18] = (uint32_t)(extra & 0xffffffffu);
    h[19] = (uint32_t)(extra >> 32);
    size_t off = hdr;
    memcpy(blob + off, m->params, n * sizeof(float)); off += n * sizeof(float);
    if (kind == 1) {
        memcpy(blob + off, m->adam_m, n * sizeof(float)); off += n * sizeof(float);
        memcpy(blob + off, m->adam_v, n * sizeof(float)); off += n * sizeof(float);
    } else if (kind == 2) {
        memcpy(blob + off, m->ada, m->ada_n * sizeof(float));
    } else if (kind == 3) {
        memcpy(blob + off, m->muon_m, n * sizeof(float));
    } else if (kind == 4) {
        memcpy(blob + off, m->adam_m8, n); off += n;
        memcpy(blob + off, m->adam_ms, m->adam_nblocks * sizeof(float));
        off += m->adam_nblocks * sizeof(float);
        memcpy(blob + off, m->adam_v, n * sizeof(float));
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
    if (h[0] != UMDL_MAGIC || (h[1] != 1 && h[1] != 2)) { free(data); return UENR_ERR_INVALID; }
    int V = (int)h[2], C = (int)h[3], L = (int)h[4], H = (int)h[5];
    int G = (int)h[6], hd = (int)h[7], E = (int)h[8], EF = (int)h[9];
    int FF = (int)h[10], tie = (int)h[11], K = (int)h[12];
    size_t n = (size_t)h[16] | ((size_t)h[17] << 32);
    int kind = (h[1] == 1) ? ((int)h[14] ? 1 : 0) : (int)h[15];
    size_t extra = (h[1] == 1)
        ? (size_t)(h[14] ? 2 * n * sizeof(float) : 0)
        : ((size_t)h[18] | ((size_t)h[19] << 32));
    if (V != m->V || C != m->C || L != m->L || H != m->H || G != m->G ||
        hd != m->hd || E != m->E || EF != m->EF || FF != m->FF ||
        tie != m->tie || K != m->K || n != m->alloc_floats) {
        free(data);
        return UENR_ERR_SHAPE;
    }
    size_t need = hdr + n * sizeof(float) + extra;
    if (size != need) { free(data); return UENR_ERR_INVALID; }
    size_t off = hdr;
    memcpy(m->params, (unsigned char *)data + off, n * sizeof(float)); off += n * sizeof(float);
    if (step) *step = (int)h[13];
    if (load_opt && kind) {
        if (uenr_model_opt_init(m, 1) != UENR_OK) { free(data); return UENR_ERR_ALLOC; }
        if (kind == 1 && m->adam_m && m->adam_v) {
            memcpy(m->adam_m, (unsigned char *)data + off, n * sizeof(float));
            off += n * sizeof(float);
            memcpy(m->adam_v, (unsigned char *)data + off, n * sizeof(float));
        } else if (kind == 2 && m->adafactor && m->ada &&
                   extra == m->ada_n * sizeof(float)) {
            memcpy(m->ada, (unsigned char *)data + off, extra);
        } else if (kind == 3 && m->muon_m && extra == n * sizeof(float)) {
            memcpy(m->muon_m, (unsigned char *)data + off, extra);
        } else if (kind == 4 && m->adam_m8 && m->adam_ms && m->adam_v &&
                   extra == n + m->adam_nblocks * sizeof(float) + n * sizeof(float)) {
            memcpy(m->adam_m8, (unsigned char *)data + off, n);
            off += n;
            memcpy(m->adam_ms, (unsigned char *)data + off, m->adam_nblocks * sizeof(float));
            off += m->adam_nblocks * sizeof(float);
            memcpy(m->adam_v, (unsigned char *)data + off, n * sizeof(float));
        }
        /* A mismatched optimizer kind (config changed) loads weights only. */
    }
    free(data);
    return UENR_OK;
}
