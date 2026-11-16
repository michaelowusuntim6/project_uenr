#include "uenr/ops.h"
#include "uenr/math.h"
#include "uenr/simd.h"
#include "uenr/thread.h"

static inline float dot_f(const float *a, const float *b, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8)));
        a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16)));
        a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24)));
    }
    float s = uenr_v_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ------------------------------------------------------------------------- */
/* Forward: per query online softmax; stores only log-sum-exp.                */
/* ------------------------------------------------------------------------- */
typedef struct {
    int B, S, H, G, hd;
    float scale;
    int window;
    const float *q, *k, *v;
    float *out, *lse;
} FwdCtx;

static void flash_fwd_bh(FwdCtx *c, int b, int h) {
    const int S = c->S, H = c->H, G = c->G, hd = c->hd;
    int g = h / (H / G);
    for (int s = 0; s < S; s++) {
        int t = b * S + s;
        const float *qr = c->q + ((size_t)t * H + h) * hd;
        float *orr = c->out + ((size_t)t * H + h) * hd;
        for (int d = 0; d < hd; d++) orr[d] = 0.0f;
        float m = -3.402823466e+38f, l = 0.0f;

        int j0 = (c->window > 0 && s >= c->window) ? s - c->window + 1 : 0;
        for (int j = j0; j <= s; j++) {
            const float *kr = c->k + ((size_t)(b * S + j) * G + g) * hd;
            const float *vr = c->v + ((size_t)(b * S + j) * G + g) * hd;
            float sc = dot_f(qr, kr, hd) * c->scale;
            if (sc > m) {
                float alpha = uenr_expf(m - sc);
                l = l * alpha + 1.0f;
                for (int d = 0; d < hd; d++) orr[d] = orr[d] * alpha + vr[d];
                m = sc;
            } else {
                float p = uenr_expf(sc - m);
                l += p;
                for (int d = 0; d < hd; d++) orr[d] += p * vr[d];
            }
        }
        float inv = uenr_rcpf(l);
        for (int d = 0; d < hd; d++) orr[d] *= inv;
        c->lse[((size_t)b * H + h) * S + s] = m + uenr_logf(l);
    }
}

static void flash_fwd_range(void *ctx, int start, int end, int worker) {
    (void)worker;
    FwdCtx *c = (FwdCtx *)ctx;
    for (int idx = start; idx < end; idx++)
        flash_fwd_bh(c, idx / c->H, idx % c->H);
}

void uenr_flash_attn_fwd(float *out, float *lse,
                         const float *q, const float *k, const float *v,
                         int B, int S, int H, int G, int hd, float scale,
                         int window, UenrPool *pool) {
    FwdCtx c = { B, S, H, G, hd, scale, window, q, k, v, out, lse };
    uenr_pool_range(pool, flash_fwd_range, &c, B * H);
}

/* ------------------------------------------------------------------------- */
/* Backward: recompute p = exp(score - lse); no probability buffer.          */
/* Parallel over (batch, KV group) to keep dk/dv race-free.                  */
/* ------------------------------------------------------------------------- */
typedef struct {
    int B, S, H, G, hd;
    float scale;
    int window;
    const float *q, *k, *v, *dout, *out, *lse;
    float *dq, *dk, *dv;
} BwdCtx;

static void flash_bwd_bg(BwdCtx *c, int b, int g) {
    const int S = c->S, H = c->H, G = c->G, hd = c->hd;
    const int group = H / G;
    for (int s = 0; s < S; s++) {
        int t = b * S + s;
        for (int h = g * group; h < (g + 1) * group; h++) {
            const float *qr = c->q + ((size_t)t * H + h) * hd;
            const float *dor = c->dout + ((size_t)t * H + h) * hd;
            const float *orr = c->out + ((size_t)t * H + h) * hd;
            float *dqr = c->dq + ((size_t)t * H + h) * hd;
            float lse = c->lse[((size_t)b * H + h) * S + s];
            float delta = dot_f(dor, orr, hd);
            for (int d = 0; d < hd; d++) dqr[d] = 0.0f;

            int j0 = (c->window > 0 && s >= c->window) ? s - c->window + 1 : 0;
            for (int j = j0; j <= s; j++) {
                const float *kr = c->k + ((size_t)(b * S + j) * G + g) * hd;
                const float *vr = c->v + ((size_t)(b * S + j) * G + g) * hd;
                float sc = dot_f(qr, kr, hd) * c->scale;
                float p = uenr_expf(sc - lse);
                float dp = dot_f(dor, vr, hd);
                float ds = p * (dp - delta);
                float dsc = ds * c->scale;
                float *dkr = c->dk + ((size_t)(b * S + j) * G + g) * hd;
                float *dvr = c->dv + ((size_t)(b * S + j) * G + g) * hd;
                for (int d = 0; d < hd; d++) {
                    dqr[d] += dsc * kr[d];
                    dkr[d] += dsc * qr[d];
                    dvr[d] += p * dor[d];
                }
            }
        }
    }
}

static void flash_bwd_range(void *ctx, int start, int end, int worker) {
    (void)worker;
    BwdCtx *c = (BwdCtx *)ctx;
    for (int idx = start; idx < end; idx++)
        flash_bwd_bg(c, idx / c->G, idx % c->G);
}

void uenr_flash_attn_bwd(float *dq, float *dk, float *dv,
                         const float *dout, const float *q, const float *k,
                         const float *v, const float *out, const float *lse,
                         int B, int S, int H, int G, int hd, float scale,
                         int window, UenrPool *pool) {
    BwdCtx c = { B, S, H, G, hd, scale, window, q, k, v, dout, out, lse, dq, dk, dv };
    uenr_pool_range(pool, flash_bwd_range, &c, B * G);
}
