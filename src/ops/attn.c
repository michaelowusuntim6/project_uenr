#include "uenr/ops.h"
#include "uenr/math.h"
#include "uenr/simd.h"
#include "uenr/thread.h"

#include <stdlib.h>

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
/* Forward: parallel over (batch, query head).                               */
/* ------------------------------------------------------------------------- */
typedef struct {
    int B, S, H, G, hd;
    float scale;
    const float *q, *k, *v;
    float *out, *probs;
} FwdCtx;

static void attn_fwd_bh(FwdCtx *c, int b, int h) {
    const int S = c->S, H = c->H, G = c->G, hd = c->hd;
    const int group = H / G;
    int g = h / group;
    for (int s = 0; s < S; s++) {
        int t = b * S + s;
        const float *qr = c->q + ((size_t)t * H + h) * hd;
        float *prow = c->probs + (((size_t)b * H + h) * S + s) * S;
        float maxv = -3.402823466e+38f;
        for (int j = 0; j <= s; j++) {
            const float *kr = c->k + ((size_t)(b * S + j) * G + g) * hd;
            float sc = dot_f(qr, kr, hd) * c->scale;
            prow[j] = sc;
            if (sc > maxv) maxv = sc;
        }
        float sum = 0.0f;
        for (int j = 0; j <= s; j++) {
            float e = uenr_expf(prow[j] - maxv);
            prow[j] = e;
            sum += e;
        }
        float inv = uenr_rcpf(sum);
        for (int j = 0; j <= s; j++) prow[j] *= inv;
        for (int j = s + 1; j < S; j++) prow[j] = 0.0f;

        float *orr = c->out + ((size_t)t * H + h) * hd;
        for (int d = 0; d < hd; d++) orr[d] = 0.0f;
        for (int j = 0; j <= s; j++) {
            const float *vr = c->v + ((size_t)(b * S + j) * G + g) * hd;
            float p = prow[j];
            for (int d = 0; d < hd; d++) orr[d] += p * vr[d];
        }
    }
}

static void attn_fwd_range(void *ctx, int start, int end, int worker) {
    (void)worker;
    FwdCtx *c = (FwdCtx *)ctx;
    for (int idx = start; idx < end; idx++) {
        int b = idx / c->H;
        int h = idx % c->H;
        attn_fwd_bh(c, b, h);
    }
}

void uenr_attn_fwd(float *out, float *probs,
                   const float *q, const float *k, const float *v,
                   int B, int S, int H, int G, int hd, float scale,
                   UenrPool *pool) {
    FwdCtx c = { B, S, H, G, hd, scale, q, k, v, out, probs };
    uenr_pool_range(pool, attn_fwd_range, &c, B * H);
}

/* ------------------------------------------------------------------------- */
/* Backward: parallel over (batch, KV head group); all query heads in a      */
/* group accumulate into the same dk/dv, so they must stay in one worker.    */
/* ------------------------------------------------------------------------- */
typedef struct {
    int B, S, H, G, hd;
    float scale;
    const float *q, *k, *v, *probs, *dout;
    float *dq, *dk, *dv;
    float *dprob; /* [nworkers * S] */
} BwdCtx;

static void attn_bwd_bg(BwdCtx *c, int b, int g, int worker) {
    const int S = c->S, H = c->H, G = c->G, hd = c->hd;
    const int group = H / G;
    float *dprob = c->dprob + (size_t)worker * S;

    for (int s = 0; s < S; s++) {
        int t = b * S + s;
        for (int h = g * group; h < (g + 1) * group; h++) {
            const float *qr = c->q + ((size_t)t * H + h) * hd;
            const float *dor = c->dout + ((size_t)t * H + h) * hd;
            float *dqr = c->dq + ((size_t)t * H + h) * hd;
            const float *prow = c->probs + (((size_t)b * H + h) * S + s) * S;

            float dotp = 0.0f;
            for (int j = 0; j <= s; j++) {
                const float *vr = c->v + ((size_t)(b * S + j) * G + g) * hd;
                float dp = dot_f(dor, vr, hd);
                dprob[j] = dp;
                dotp += prow[j] * dp;
            }
            for (int j = 0; j <= s; j++) {
                const float *kr = c->k + ((size_t)(b * S + j) * G + g) * hd;
                float ds = prow[j] * (dprob[j] - dotp);
                float dsc = ds * c->scale;
                for (int d = 0; d < hd; d++) {
                    dqr[d] += dsc * kr[d];
                    c->dk[((size_t)(b * S + j) * G + g) * hd + d] += dsc * qr[d];
                }
                float p = prow[j];
                for (int d = 0; d < hd; d++)
                    c->dv[((size_t)(b * S + j) * G + g) * hd + d] += p * dor[d];
            }
        }
    }
}

static void attn_bwd_range(void *ctx, int start, int end, int worker) {
    BwdCtx *c = (BwdCtx *)ctx;
    for (int idx = start; idx < end; idx++) {
        int b = idx / c->G;
        int g = idx % c->G;
        attn_bwd_bg(c, b, g, worker);
    }
}

void uenr_attn_bwd(float *dq, float *dk, float *dv,
                   const float *dout, const float *q, const float *k,
                   const float *v, const float *probs,
                   int B, int S, int H, int G, int hd, float scale,
                   UenrPool *pool) {
    int nw = pool ? uenr_pool_threads(pool) : 1;
    float *dprob = (float *)malloc((size_t)nw * S * sizeof(float));
    if (!dprob) return;
    BwdCtx c = { B, S, H, G, hd, scale, q, k, v, probs, dout, dq, dk, dv, dprob };
    uenr_pool_range(pool, attn_bwd_range, &c, B * G);
    free(dprob);
}
