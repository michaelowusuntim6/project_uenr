#include "uenr/model.h"
#include "uenr/math.h"
#include "uenr/ops.h"

#include <stdlib.h>
#include <string.h>

static float *ac(size_t n) {
    size_t bytes = ((n * sizeof(float)) + 63) & ~(size_t)63;
    return (float *)aligned_alloc(64, bytes ? bytes : 64);
}
static int *ai(size_t n) {
    size_t bytes = ((n * sizeof(int)) + 63) & ~(size_t)63;
    return (int *)aligned_alloc(64, bytes ? bytes : 64);
}

UenrResult uenr_kv_cache_init(UenrKVCache *c, const UenrConfig *cfg, int cap) {
    if (!c || !cfg || cap <= 0) return UENR_ERR_INVALID;
    memset(c, 0, sizeof(*c));
    int L = cfg->model.layers;
    int G = cfg->model.kv_heads;
    int hd = cfg->model.head_dim;
    int KV = G * hd;
    int C = cfg->model.hidden;
    int F = cfg->model.ffn;
    int EF = cfg->moe.expert_ffn;
    int E = cfg->moe.num_experts;
    int K = cfg->moe.experts_per_tok;
    int maxF = (F > EF) ? F : EF;
    c->L = L; c->KV = KV; c->cap = cap; c->len = 0;
    c->int8 = cfg->infer.kv_int8;
    c->G = G; c->hd = hd;
    c->n_slots = uenr_active_kv_layers(cfg);
    c->type = ai(L); c->donor = ai(L); c->shared = ai(L);
    if (!c->type || !c->donor || !c->shared) { uenr_kv_cache_free(c); return UENR_ERR_ALLOC; }
    for (int l = 0; l < L; l++) {
        c->type[l] = uenr_layer_is_sliding(cfg, l) ? 0 : 1;
        c->shared[l] = uenr_layer_is_shared(cfg, l);
        c->donor[l] = c->shared[l] ? uenr_layer_donor(cfg, l) : l;
    }
    if (c->int8) {
        c->k8 = (int8_t *)aligned_alloc(64, (((size_t)L * cap * KV) + 63) & ~(size_t)63);
        c->v8 = (int8_t *)aligned_alloc(64, (((size_t)L * cap * KV) + 63) & ~(size_t)63);
        c->ks = ac((size_t)L * G);
        c->vs = ac((size_t)L * G);
        if (c->k8) memset(c->k8, 0, (size_t)L * cap * KV);
        if (c->v8) memset(c->v8, 0, (size_t)L * cap * KV);
    } else {
        c->k = ac((size_t)L * cap * KV);
        c->v = ac((size_t)L * cap * KV);
    }
    c->x = ac(C); c->xa = ac(C); c->q = ac(C); c->attn = ac(C);
    c->proj = ac(C); c->x2 = ac(C); c->xf = ac(C); c->eout = ac(C); c->ys = ac(C);
    c->krow = ac(KV); c->vrow = ac(KV);
    c->gbuf = ac(maxF); c->ubuf = ac(maxF);
    c->router_logits = ac(E);
    c->xfinal = ac(C); c->logits = ac(cfg->model.vocab_size);
    c->topk_idx = ai(K); c->topk_val = ac(K);
    if ((!c->int8 && (!c->k || !c->v)) ||
        (c->int8 && (!c->k8 || !c->v8 || !c->ks || !c->vs)) ||
        !c->x || !c->xa || !c->q || !c->attn || !c->proj ||
        !c->x2 || !c->xf || !c->eout || !c->ys || !c->krow || !c->vrow || !c->gbuf ||
        !c->ubuf || !c->router_logits || !c->xfinal || !c->logits ||
        !c->topk_idx || !c->topk_val) {
        uenr_kv_cache_free(c);
        return UENR_ERR_ALLOC;
    }
    return UENR_OK;
}

void uenr_kv_cache_free(UenrKVCache *c) {
    if (!c) return;
    free(c->k); free(c->v); free(c->k8); free(c->v8); free(c->ks); free(c->vs); free(c->x); free(c->xa); free(c->q); free(c->attn);
    free(c->proj); free(c->x2); free(c->xf); free(c->eout); free(c->ys); free(c->krow); free(c->vrow);
    free(c->gbuf); free(c->ubuf); free(c->router_logits); free(c->xfinal); free(c->logits);
    free(c->topk_idx); free(c->topk_val);
    free(c->type); free(c->donor); free(c->shared);
    memset(c, 0, sizeof(*c));
}

void uenr_kv_cache_reset(UenrKVCache *c) { if (c) c->len = 0; }

static inline float dotv(const float *a, const float *b, int n) {
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

static void norm_rope_head(float *v, const float *w, const UenrRope *r, int pos,
                           int hd, float eps) {
    float ss = 0.0f;
    for (int d = 0; d < hd; d++) ss += v[d] * v[d];
    float rr = uenr_rsqrtf(ss / (float)hd + eps);
    for (int d = 0; d < hd; d++) v[d] *= rr * w[d];
    int half = hd / 2;
    const float *ct = r->cos_tab + (size_t)pos * r->half;
    const float *st = r->sin_tab + (size_t)pos * r->half;
    if (half == r->half) {
        for (int j = 0; j < half; j++) {
            float a = v[j], b = v[j + half];
            float cc = ct[j], ss2 = st[j];
            v[j] = a * cc - b * ss2;
            v[j + half] = a * ss2 + b * cc;
        }
    }
}

static void norm_v_head(float *v, int hd, float eps) {
    float ss = 0.0f;
    for (int d = 0; d < hd; d++) ss += v[d] * v[d];
    float r = uenr_rsqrtf(ss / (float)hd + eps);
    for (int d = 0; d < hd; d++) v[d] *= r;
}

UenrResult uenr_model_forward_step(UenrModel *m, UenrGemm *g, UenrKVCache *c,
                                   int token, int pos, float *logits_out) {
    if (!m || !g || !c || !logits_out || token < 0 || token >= m->V)
        return UENR_ERR_INVALID;
    if (pos < 0 || pos >= c->cap) return UENR_ERR_BOUNDS;
    const int C = m->C, H = m->H, G = m->G, hd = m->hd, KV = m->KV;
    const int E = m->E, EF = m->EF, FF = m->FF, K = m->K;
    const float eps = m->cfg.model.rms_eps;
    const float scale = m->cfg.model.attn_scale > 0.0f
                            ? m->cfg.model.attn_scale : uenr_rsqrtf((float)hd);
    const int group = H / G;
    float *x = c->x, *xa = c->xa, *q = c->q, *krow = c->krow, *vrow = c->vrow;
    float *attn = c->attn, *proj = c->proj, *x2 = c->x2;
    float *xf = c->xf, *eout = c->eout, *ys = c->ys;

    memcpy(x, m->tok_emb + (size_t)token * C, (size_t)C * sizeof(float));
    const int q8 = m->q8_on;

    for (int l = 0; l < m->L; l++) {
        UenrLayer *y = &m->layers[l];
        UenrQLayer *ql = q8 ? &m->q_layers[l] : NULL;
        const int donor = c->donor[l];
        const int shared = c->shared[l];
        const int local = (c->type[l] == 0);
        const int W = m->cfg.model.sliding_window;
        const int widx = (local && W > 0) ? (pos % W) : pos;
        const int astart = (local && W > 0 && pos >= W) ? pos - W + 1 : 0;
        uenr_rmsnorm_fwd(xa, &c->rstd, x, y->rms_attn, 1, C, eps);
        if (q8) {
            uenr_gemv_q8(q, xa, ql->wq.q, ql->wq.scale, C, C);
            if (!shared) {
                uenr_gemv_q8(krow, xa, ql->wk.q, ql->wk.scale, KV, C);
                uenr_gemv_q8(vrow, xa, ql->wv.q, ql->wv.scale, KV, C);
            }
        } else {
            uenr_linear_fwd(g, q, xa, y->wq, 1, C, C);
            if (!shared) {
                uenr_linear_fwd(g, krow, xa, y->wk, 1, C, KV);
                uenr_linear_fwd(g, vrow, xa, y->wv, 1, C, KV);
            }
        }

        const UenrRope *rp = y->type ? &m->rope_global : &m->rope_local;
        for (int h = 0; h < H; h++)
            norm_rope_head(q + (size_t)h * hd, y->q_norm, rp, pos, hd, eps);
        if (!shared) {
            for (int gg = 0; gg < G; gg++)
                norm_rope_head(krow + (size_t)gg * hd, y->k_norm, rp, pos, hd, eps);
            if (m->cfg.model.v_norm) {
                for (int gg = 0; gg < G; gg++)
                    norm_v_head(vrow + (size_t)gg * hd, hd, eps);
            }
        }

        if (!shared && c->int8) {
            int8_t *kc8 = c->k8 + ((size_t)l * c->cap + widx) * KV;
            int8_t *vc8 = c->v8 + ((size_t)l * c->cap + widx) * KV;
            for (int gg = 0; gg < G; gg++) {
                const float *kh = krow + (size_t)gg * hd;
                const float *vh = vrow + (size_t)gg * hd;
                float kmax = 0.0f, vmax = 0.0f;
                for (int d = 0; d < hd; d++) {
                    float a = kh[d] < 0 ? -kh[d] : kh[d];
                    float b = vh[d] < 0 ? -vh[d] : vh[d];
                    if (a > kmax) kmax = a;
                    if (b > vmax) vmax = b;
                }
                float ks = kmax > 0 ? kmax / 127.0f : 1.0f;
                float vs = vmax > 0 ? vmax / 127.0f : 1.0f;
                c->ks[(size_t)l * G + gg] = ks;
                c->vs[(size_t)l * G + gg] = vs;
                for (int d = 0; d < hd; d++) {
                    float qk = kh[d] / ks;
                    float qv = vh[d] / vs;
                    int ik = (int)(qk + (qk >= 0 ? 0.5f : -0.5f));
                    int iv = (int)(qv + (qv >= 0 ? 0.5f : -0.5f));
                    if (ik > 127) ik = 127;
                    if (ik < -127) ik = -127;
                    if (iv > 127) iv = 127;
                    if (iv < -127) iv = -127;
                    kc8[(size_t)gg * hd + d] = (int8_t)ik;
                    vc8[(size_t)gg * hd + d] = (int8_t)iv;
                }
            }
        } else if (!shared) {
            float *kc = c->k + ((size_t)l * c->cap + widx) * KV;
            float *vc = c->v + ((size_t)l * c->cap + widx) * KV;
            memcpy(kc, krow, (size_t)KV * sizeof(float));
            memcpy(vc, vrow, (size_t)KV * sizeof(float));
        }

        for (int h = 0; h < H; h++) {
            int gg = h / group;
            const float *qh = q + (size_t)h * hd;
            float *oh = attn + (size_t)h * hd;
            float mx = -3.402823466e+38f;
            float sc[4096];
            const size_t rowstride = (size_t)c->cap * KV;
            const float ksc = c->int8 ? c->ks[(size_t)donor * G + gg] : 1.0f;
            const float vsc = c->int8 ? c->vs[(size_t)donor * G + gg] : 1.0f;
            for (int j = astart; j <= pos; j++) {
                float acc = 0.0f;
                int jj = (local && W > 0) ? (j % W) : j;
                if (c->int8) {
                    const int8_t *kj = c->k8 + (size_t)donor * rowstride + (size_t)jj * KV + (size_t)gg * hd;
                    for (int d = 0; d < hd; d++) acc += qh[d] * ((float)kj[d] * ksc);
                } else {
                    const float *kj = c->k + (size_t)donor * rowstride + (size_t)jj * KV + (size_t)gg * hd;
                    acc = dotv(qh, kj, hd);
                }
                sc[j] = acc * scale;
                if (sc[j] > mx) mx = sc[j];
            }
            float sum = 0.0f;
            for (int j = astart; j <= pos; j++) { sc[j] = uenr_expf(sc[j] - mx); sum += sc[j]; }
            float inv = uenr_rcpf(sum);
            for (int d = 0; d < hd; d++) oh[d] = 0.0f;
            for (int j = astart; j <= pos; j++) {
                float p = sc[j] * inv;
                int jj = (local && W > 0) ? (j % W) : j;
                if (c->int8) {
                    const int8_t *vj = c->v8 + (size_t)donor * rowstride + (size_t)jj * KV + (size_t)gg * hd;
                    for (int d = 0; d < hd; d++) oh[d] += p * ((float)vj[d] * vsc);
                } else {
                    const float *vj = c->v + (size_t)donor * rowstride + (size_t)jj * KV + (size_t)gg * hd;
                    for (int d = 0; d < hd; d++) oh[d] += p * vj[d];
                }
            }
        }

        if (q8) uenr_gemv_q8(proj, attn, ql->wo.q, ql->wo.scale, C, C);
        else uenr_linear_fwd(g, proj, attn, y->wo, 1, C, C);
        uenr_rmsnorm_fwd(proj, &c->rstd, proj, y->post_attn, 1, C, eps);
        for (int d = 0; d < C; d++) x2[d] = x[d] + y->ls[d] * proj[d];

        uenr_rmsnorm_fwd(xf, &c->rstd, x2, y->rms_ffn, 1, C, eps);
        if (q8) uenr_gemv_q8(c->router_logits, xf, ql->router.q, ql->router.scale, E, C);
        else uenr_linear_fwd(g, c->router_logits, xf, y->router, 1, C, E);
        float mx = c->router_logits[0];
        for (int i = 1; i < E; i++) if (c->router_logits[i] > mx) mx = c->router_logits[i];
        float sum = 0.0f;
        for (int i = 0; i < E; i++) { c->router_logits[i] = uenr_expf(c->router_logits[i] - mx); sum += c->router_logits[i]; }
        float inv = uenr_rcpf(sum);
        for (int i = 0; i < E; i++) c->router_logits[i] *= inv;
        for (int s = 0; s < K; s++) {
            int best = -1;
            float bv = -1.0f;
            for (int i = 0; i < E; i++) {
                int taken = 0;
                for (int j = 0; j < s; j++) if (c->topk_idx[j] == i) { taken = 1; break; }
                if (!taken && c->router_logits[i] > bv) { bv = c->router_logits[i]; best = i; }
            }
            c->topk_idx[s] = best;
            c->topk_val[s] = bv;
        }

        for (int d = 0; d < C; d++) eout[d] = 0.0f;
        if (m->cfg.moe.enabled) {
            for (int s = 0; s < K; s++) {
                int e = c->topk_idx[s];
                if (q8) {
                    UenrQMat qg = { ql->expert_wg.q + (size_t)e * EF * C,
                                    ql->expert_wg.scale + (size_t)e * EF, EF, C };
                    UenrQMat qu = { ql->expert_wu.q + (size_t)e * EF * C,
                                    ql->expert_wu.scale + (size_t)e * EF, EF, C };
                    UenrQMat qd = { ql->expert_wd.q + (size_t)e * C * EF,
                                    ql->expert_wd.scale + (size_t)e * C, C, EF };
                    uenr_swiglu_fwd_q8_1(ys, xf, &qg, &qu, &qd, c->gbuf, c->ubuf);
                } else {
                    const float *Wg = y->expert_wg + (size_t)e * EF * C;
                    const float *Wu = y->expert_wu + (size_t)e * EF * C;
                    const float *Wd = y->expert_wd + (size_t)e * C * EF;
                    uenr_swiglu_fwd(g, 1, C, EF, xf, Wg, Wu, Wd, ys, c->gbuf, c->ubuf);
                }
                float gate = c->topk_val[s];
                for (int d = 0; d < C; d++) eout[d] += gate * ys[d];
            }
            if (m->cfg.moe.shared_expert && y->shared_wg) {
                if (q8) uenr_swiglu_fwd_q8_1(ys, xf, &ql->shared_wg, &ql->shared_wu,
                                             &ql->shared_wd, c->gbuf, c->ubuf);
                else uenr_swiglu_fwd(g, 1, C, FF, xf, y->shared_wg, y->shared_wu,
                                     y->shared_wd, ys, c->gbuf, c->ubuf);
                for (int d = 0; d < C; d++) eout[d] += ys[d];
            }
        } else {
            if (q8) uenr_swiglu_fwd_q8_1(ys, xf, &ql->shared_wg, &ql->shared_wu,
                                         &ql->shared_wd, c->gbuf, c->ubuf);
            else uenr_swiglu_fwd(g, 1, C, FF, xf, y->shared_wg, y->shared_wu,
                                 y->shared_wd, ys, c->gbuf, c->ubuf);
            for (int d = 0; d < C; d++) eout[d] += ys[d];
        }
        uenr_rmsnorm_fwd(eout, &c->rstd, eout, y->post_ffn, 1, C, eps);
        for (int d = 0; d < C; d++) x[d] = x2[d] + eout[d];
    }

    uenr_rmsnorm_fwd(c->xfinal, &c->rstd, x, m->rms_final, 1, C, eps);
    if (q8) uenr_gemv_q8(logits_out, c->xfinal, m->q_lm_head.q,
                         m->q_lm_head.scale, m->V, C);
    else uenr_linear_fwd(g, logits_out, c->xfinal, m->lm_head, 1, C, m->V);
    c->len = pos + 1;
    return UENR_OK;
}
