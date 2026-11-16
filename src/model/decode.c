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
    int KV = cfg->model.kv_heads * cfg->model.head_dim;
    int C = cfg->model.hidden;
    int F = cfg->model.ffn;
    int EF = cfg->moe.expert_ffn;
    int E = cfg->moe.num_experts;
    int K = cfg->moe.experts_per_tok;
    int maxF = (F > EF) ? F : EF;
    c->L = L; c->KV = KV; c->cap = cap; c->len = 0;
    c->k = ac((size_t)L * cap * KV);
    c->v = ac((size_t)L * cap * KV);
    c->x = ac(C); c->xa = ac(C); c->q = ac(C); c->attn = ac(C);
    c->proj = ac(C); c->xf = ac(C); c->eout = ac(C); c->ys = ac(C);
    c->krow = ac(KV); c->vrow = ac(KV);
    c->gbuf = ac(maxF); c->ubuf = ac(maxF);
    c->router_logits = ac(E);
    c->xfinal = ac(C); c->logits = ac(cfg->model.vocab_size);
    c->topk_idx = ai(K); c->topk_val = ac(K);
    if (!c->k || !c->v || !c->x || !c->xa || !c->q || !c->attn || !c->proj ||
        !c->xf || !c->eout || !c->ys || !c->krow || !c->vrow || !c->gbuf ||
        !c->ubuf || !c->router_logits || !c->xfinal || !c->logits ||
        !c->topk_idx || !c->topk_val) {
        uenr_kv_cache_free(c);
        return UENR_ERR_ALLOC;
    }
    return UENR_OK;
}

void uenr_kv_cache_free(UenrKVCache *c) {
    if (!c) return;
    free(c->k); free(c->v); free(c->x); free(c->xa); free(c->q); free(c->attn);
    free(c->proj); free(c->xf); free(c->eout); free(c->ys); free(c->krow); free(c->vrow);
    free(c->gbuf); free(c->ubuf); free(c->router_logits); free(c->xfinal); free(c->logits);
    free(c->topk_idx); free(c->topk_val);
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
            float a = v[2 * j], b = v[2 * j + 1];
            float cc = ct[j], ss2 = st[j];
            v[2 * j] = a * cc - b * ss2;
            v[2 * j + 1] = a * ss2 + b * cc;
        }
    }
}

UenrResult uenr_model_forward_step(UenrModel *m, UenrGemm *g, UenrKVCache *c,
                                   int token, int pos, float *logits_out) {
    if (!m || !g || !c || !logits_out || token < 0 || token >= m->V)
        return UENR_ERR_INVALID;
    if (pos < 0 || pos >= c->cap) return UENR_ERR_BOUNDS;
    const int C = m->C, H = m->H, G = m->G, hd = m->hd, KV = m->KV;
    const int E = m->E, EF = m->EF, FF = m->FF, K = m->K;
    const float eps = m->cfg.model.rms_eps;
    const float scale = uenr_rsqrtf((float)hd);
    const int group = H / G;
    float *x = c->x, *xa = c->xa, *q = c->q, *krow = c->krow, *vrow = c->vrow;
    float *attn = c->attn, *proj = c->proj, *xf = c->xf, *eout = c->eout, *ys = c->ys;

    memcpy(x, m->tok_emb + (size_t)token * C, (size_t)C * sizeof(float));

    for (int l = 0; l < m->L; l++) {
        UenrLayer *y = &m->layers[l];
        uenr_rmsnorm_fwd(xa, &c->rstd, x, y->rms_attn, 1, C, eps);
        uenr_linear_fwd(g, q, xa, y->wq, 1, C, C);
        uenr_linear_fwd(g, krow, xa, y->wk, 1, C, KV);
        uenr_linear_fwd(g, vrow, xa, y->wv, 1, C, KV);

        for (int h = 0; h < H; h++)
            norm_rope_head(q + (size_t)h * hd, y->q_norm, &m->rope, pos, hd, eps);
        for (int gg = 0; gg < G; gg++)
            norm_rope_head(krow + (size_t)gg * hd, y->k_norm, &m->rope, pos, hd, eps);

        float *kc = c->k + ((size_t)l * c->cap + pos) * KV;
        float *vc = c->v + ((size_t)l * c->cap + pos) * KV;
        memcpy(kc, krow, (size_t)KV * sizeof(float));
        memcpy(vc, vrow, (size_t)KV * sizeof(float));

        for (int h = 0; h < H; h++) {
            int gg = h / group;
            const float *qh = q + (size_t)h * hd;
            float *oh = attn + (size_t)h * hd;
            float mx = -3.402823466e+38f;
            float sc[4096];
            for (int j = 0; j <= pos; j++) {
                const float *kj = c->k + ((size_t)l * c->cap + j) * KV + (size_t)gg * hd;
                sc[j] = dotv(qh, kj, hd) * scale;
                if (sc[j] > mx) mx = sc[j];
            }
            float sum = 0.0f;
            for (int j = 0; j <= pos; j++) { sc[j] = uenr_expf(sc[j] - mx); sum += sc[j]; }
            float inv = uenr_rcpf(sum);
            for (int d = 0; d < hd; d++) oh[d] = 0.0f;
            for (int j = 0; j <= pos; j++) {
                const float *vj = c->v + ((size_t)l * c->cap + j) * KV + (size_t)gg * hd;
                float p = sc[j] * inv;
                for (int d = 0; d < hd; d++) oh[d] += p * vj[d];
            }
        }

        uenr_linear_fwd(g, proj, attn, y->wo, 1, C, C);
        for (int d = 0; d < C; d++) x[d] += proj[d];

        uenr_rmsnorm_fwd(xf, &c->rstd, x, y->rms_ffn, 1, C, eps);
        uenr_linear_fwd(g, c->router_logits, xf, y->router, 1, C, E);
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
                const float *Wg = y->expert_wg + (size_t)e * EF * C;
                const float *Wu = y->expert_wu + (size_t)e * EF * C;
                const float *Wd = y->expert_wd + (size_t)e * C * EF;
                uenr_swiglu_fwd(g, 1, C, EF, xf, Wg, Wu, Wd, ys, c->gbuf, c->ubuf);
                float gate = c->topk_val[s];
                for (int d = 0; d < C; d++) eout[d] += gate * ys[d];
            }
            if (m->cfg.moe.shared_expert && y->shared_wg) {
                uenr_swiglu_fwd(g, 1, C, FF, xf, y->shared_wg, y->shared_wu,
                                y->shared_wd, ys, c->gbuf, c->ubuf);
                for (int d = 0; d < C; d++) eout[d] += ys[d];
            }
        } else {
            uenr_swiglu_fwd(g, 1, C, FF, xf, y->shared_wg, y->shared_wu,
                            y->shared_wd, ys, c->gbuf, c->ubuf);
            for (int d = 0; d < C; d++) eout[d] += ys[d];
        }
        for (int d = 0; d < C; d++) x[d] += eout[d];
    }

    uenr_rmsnorm_fwd(c->xfinal, &c->rstd, x, m->rms_final, 1, C, eps);
    uenr_linear_fwd(g, logits_out, c->xfinal, m->lm_head, 1, C, m->V);
    c->len = pos + 1;
    return UENR_OK;
}
