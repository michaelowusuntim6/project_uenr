#include "impl_suffix.h"
#define uenr_attn_fwd UENR_OP(attn_fwd)
#define uenr_attn_bwd UENR_OP(attn_bwd)
#define uenr_attn_fwd16 UENR_OP(attn_fwd16)
#define uenr_attn_bwd16 UENR_OP(attn_bwd16)
#include "uenr/ops.h"
#include "uenr/math.h"
#include "uenr/simd.h"
#include "uenr/thread.h"
#if defined(__AVX2__)
#include "avx2_math.h"
#endif

#include <stdlib.h>

#if defined(__AVX512F__)
static inline float dot_f(const float *a, const float *b, int n) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16),
                             _mm512_loadu_ps(b + i + 16), a1);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    for (; i < n; i++) s = uenr_fmaf(a[i], b[i], s);
    return s;
}

static inline void axpy_f(float *dst, const float *src, float s, int n) {
    __m512 sv = _mm512_set1_ps(s);
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(sv, _mm512_loadu_ps(src + i),
                                                  _mm512_loadu_ps(dst + i)));
    for (; i < n; i++) dst[i] = uenr_fmaf(s, src[i], dst[i]);
}
#else
static inline float dot_f(const float *a, const float *b, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = uenr_v_fmadd(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
        a1 = uenr_v_fmadd(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
        a2 = uenr_v_fmadd(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), a2);
        a3 = uenr_v_fmadd(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), a3);
    }
    float s = uenr_v_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    for (; i < n; i++) s = uenr_fmaf(a[i], b[i], s);
    return s;
}

/* dst += s * src.  FMA in the AVX2 translation unit, mul+add in AVX1. */
static inline void axpy_f(float *dst, const float *src, float s, int n) {
    __m256 sv = _mm256_set1_ps(s);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, uenr_v_fmadd(sv, _mm256_loadu_ps(src + i),
                                               _mm256_loadu_ps(dst + i)));
    for (; i < n; i++) dst[i] = uenr_fmaf(s, src[i], dst[i]);
}
#endif

/* ------------------------------------------------------------------------- */
/* Forward: parallel over (batch, query head).                               */
/* ------------------------------------------------------------------------- */
typedef struct {
    int B, S, H, G, hd;
    float scale;
    int window;
    float softcap;
    const float *q, *k, *v;
    float *out, *probs;
    const int *doc_start;
} FwdCtx;

static void attn_fwd_bh(FwdCtx *c, int b, int h) {
    const int S = c->S, H = c->H, G = c->G, hd = c->hd;
    const int group = H / G;
    int g = h / group;
    for (int s = 0; s < S; s++) {
        int t = b * S + s;
        const float *qr = c->q + ((size_t)t * H + h) * hd;
        float *prow = c->probs + (((size_t)b * H + h) * S + s) * S;
        float *orr0 = c->out + ((size_t)t * H + h) * hd;
        int ds = c->doc_start ? c->doc_start[t] : 0;
        if (ds < 0) {
            for (int j = 0; j < S; j++) prow[j] = 0.0f;
            for (int d = 0; d < hd; d++) orr0[d] = 0.0f;
            continue;
        }
        int j0 = (c->window > 0 && s >= c->window) ? s - c->window + 1 : 0;
        if (j0 < ds) j0 = ds;
        for (int j = 0; j < j0; j++) prow[j] = 0.0f;
        for (int j = j0; j <= s; j++) {
            const float *kr = c->k + ((size_t)(b * S + j) * G + g) * hd;
            float sc = dot_f(qr, kr, hd) * c->scale;
            if (c->softcap > 0.0f && (sc > 0.1f * c->softcap || sc < -0.1f * c->softcap))
                sc = c->softcap * uenr_tanhf(sc / c->softcap);
            prow[j] = sc;
        }
        /* Row max: exact comparison, so the reduction order does not matter. */
        float maxv = -3.402823466e+38f;
        int j = j0;
#if defined(__AVX2__)
        {
            __m256 vm = _mm256_set1_ps(-3.402823466e+38f);
            for (; j + 8 <= s + 1; j += 8)
                vm = _mm256_max_ps(vm, _mm256_loadu_ps(prow + j));
            maxv = uenr_v_hmax(vm);
        }
#endif
        for (; j <= s; j++) if (prow[j] > maxv) maxv = prow[j];
        float sum = 0.0f;
        j = j0;
#if defined(__AVX2__)
        {
            __m256 vsum = _mm256_setzero_ps();
            __m256 vmax = _mm256_set1_ps(maxv);
            for (; j + 8 <= s + 1; j += 8) {
                __m256 e = uenr_avx2_exp8(
                    _mm256_sub_ps(_mm256_loadu_ps(prow + j), vmax));
                _mm256_storeu_ps(prow + j, e);
                vsum = _mm256_add_ps(vsum, e);
            }
            sum = uenr_v_hsum(vsum);
        }
#endif
        for (; j <= s; j++) {
            float e = uenr_expf(prow[j] - maxv);
            prow[j] = e;
            sum += e;
        }
        float inv = uenr_rcpf(sum);
        j = j0;
#if defined(__AVX2__)
        {
            __m256 vinv = _mm256_set1_ps(inv);
            for (; j + 8 <= s + 1; j += 8)
                _mm256_storeu_ps(prow + j,
                                 _mm256_mul_ps(_mm256_loadu_ps(prow + j), vinv));
        }
#endif
        for (; j <= s; j++) prow[j] *= inv;
        for (int j = s + 1; j < S; j++) prow[j] = 0.0f;

        float *orr = c->out + ((size_t)t * H + h) * hd;
        for (int d = 0; d < hd; d++) orr[d] = 0.0f;
        for (int j = j0; j <= s; j++) {
            const float *vr = c->v + ((size_t)(b * S + j) * G + g) * hd;
            axpy_f(orr, vr, prow[j], hd);
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
                   int window, float softcap, const int *doc_start,
                   UenrPool *pool) {
    FwdCtx c = { B, S, H, G, hd, scale, window, softcap, q, k, v, out, probs,
                 doc_start };
    uenr_pool_range(pool, attn_fwd_range, &c, B * H);
}

/* ------------------------------------------------------------------------- */
/* Backward: parallel over (batch, KV head group); all query heads in a      */
/* group accumulate into the same dk/dv, so they must stay in one worker.    */
/* ------------------------------------------------------------------------- */
typedef struct {
    int B, S, H, G, hd;
    float scale;
    int window;
    float softcap;
    const float *q, *k, *v, *probs, *dout;
    float *dq, *dk, *dv;
    float *dprob; /* [nworkers * S] */
    const int *doc_start;
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
            int ds = c->doc_start ? c->doc_start[t] : 0;
            if (ds < 0) continue;
            int j0 = (c->window > 0 && s >= c->window) ? s - c->window + 1 : 0;
            if (j0 < ds) j0 = ds;

            float dotp = 0.0f;
            for (int j = j0; j <= s; j++) {
                const float *vr = c->v + ((size_t)(b * S + j) * G + g) * hd;
                float dp = dot_f(dor, vr, hd);
                dprob[j] = dp;
                dotp += prow[j] * dp;
            }
            for (int j = j0; j <= s; j++) {
                const float *kr = c->k + ((size_t)(b * S + j) * G + g) * hd;
                float ds = prow[j] * (dprob[j] - dotp);
                const float *kraw = c->k + ((size_t)(b * S + j) * G + g) * hd;
                float raw = dot_f(qr, kraw, hd) * c->scale;
                float cap_deriv = 1.0f;
                if (c->softcap > 0.0f &&
                    (raw > 0.1f * c->softcap || raw < -0.1f * c->softcap)) {
                    float t = uenr_tanhf(raw / c->softcap);
                    cap_deriv = 1.0f - t * t;
                }
                float dsc = ds * cap_deriv * c->scale;
                float *dkr = c->dk + ((size_t)(b * S + j) * G + g) * hd;
                float *dvr = c->dv + ((size_t)(b * S + j) * G + g) * hd;
                axpy_f(dqr, kr, dsc, hd);
                axpy_f(dkr, qr, dsc, hd);
                axpy_f(dvr, dor, prow[j], hd);
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

UenrResult uenr_attn_bwd(float *dq, float *dk, float *dv,
                         const float *dout, const float *q, const float *k,
                         const float *v, const float *probs,
                         int B, int S, int H, int G, int hd, float scale,
                         int window, float softcap, const int *doc_start,
                         float *scratch, UenrPool *pool) {
    int nw = pool ? uenr_pool_threads(pool) : 1;
    float *dprob = scratch;
    if (!dprob) {
        dprob = (float *)malloc((size_t)nw * S * sizeof(float));
        if (!dprob) return UENR_ERR_ALLOC;
    }
    BwdCtx c = { B, S, H, G, hd, scale, window, softcap, q, k, v, probs, dout,
                 dq, dk, dv, dprob, doc_start };
    uenr_pool_range(pool, attn_bwd_range, &c, B * G);
    if (!scratch) free(dprob);
    return UENR_OK;
}

/* ------------------------------------------------------------------------- */
/* FP16 probability storage: same math, half the saved-activation RAM.       */
/* Forward stores normalized probabilities as IEEE half; backward converts   */
/* each consumed row element on the fly.                                     */
/* ------------------------------------------------------------------------- */
static inline uint16_t f2h16(float x) {
    __m128 v = _mm_set_ss(x);
    __m128i hcode = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return (uint16_t)_mm_extract_epi16(hcode, 0);
}

static inline float h2f16(uint16_t h) {
    __m128i hcode = _mm_cvtsi32_si128((int)h);
    return _mm_cvtss_f32(_mm_cvtph_ps(hcode));
}

typedef struct {
    int B, S, H, G, hd;
    float scale;
    int window;
    float softcap;
    const float *q, *k, *v;
    float *out;
    uint16_t *probs;
    const int *doc_start;
} Fwd16Ctx;

static void attn_fwd16_bh(Fwd16Ctx *c, int b, int h) {
    const int S = c->S, H = c->H, G = c->G, hd = c->hd;
    const int group = H / G;
    int g = h / group;
    for (int s = 0; s < S; s++) {
        int t = b * S + s;
        const float *qr = c->q + ((size_t)t * H + h) * hd;
        uint16_t *prow = c->probs + (((size_t)b * H + h) * S + s) * S;
        float *orr0 = c->out + ((size_t)t * H + h) * hd;
        int ds = c->doc_start ? c->doc_start[t] : 0;
        if (ds < 0) {
            for (int j = 0; j < S; j++) prow[j] = 0;
            for (int d = 0; d < hd; d++) orr0[d] = 0.0f;
            continue;
        }
        int j0 = (c->window > 0 && s >= c->window) ? s - c->window + 1 : 0;
        if (j0 < ds) j0 = ds;
        for (int j = 0; j < j0; j++) prow[j] = 0;
        float maxv = -3.402823466e+38f;
        /* Pass 1: raw capped scores, stored as FP16 (softcap bounds them). */
        for (int j = j0; j <= s; j++) {
            const float *kr = c->k + ((size_t)(b * S + j) * G + g) * hd;
            float sc = dot_f(qr, kr, hd) * c->scale;
            if (c->softcap > 0.0f && (sc > 0.1f * c->softcap || sc < -0.1f * c->softcap))
                sc = c->softcap * uenr_tanhf(sc / c->softcap);
            prow[j] = f2h16(sc);
            if (sc > maxv) maxv = sc;
        }
        /* Pass 2: exponentiate (stored FP16), accumulate denominator. */
        float sum = 0.0f;
        for (int j = j0; j <= s; j++) {
            float e = uenr_expf(h2f16(prow[j]) - maxv);
            sum += e;
            prow[j] = f2h16(e);
        }
        float inv = uenr_rcpf(sum);
        /* Pass 3: normalize. */
        for (int j = j0; j <= s; j++)
            prow[j] = f2h16(h2f16(prow[j]) * inv);
        for (int j = s + 1; j < S; j++) prow[j] = 0;
        float *orr = c->out + ((size_t)t * H + h) * hd;
        for (int d = 0; d < hd; d++) orr[d] = 0.0f;
        for (int j = j0; j <= s; j++) {
            const float *vr = c->v + ((size_t)(b * S + j) * G + g) * hd;
            axpy_f(orr, vr, h2f16(prow[j]), hd);
        }
    }
}

static void attn_fwd16_range(void *ctx, int start, int end, int worker) {
    (void)worker;
    Fwd16Ctx *c = (Fwd16Ctx *)ctx;
    for (int idx = start; idx < end; idx++) {
        int b = idx / c->H;
        int h = idx % c->H;
        attn_fwd16_bh(c, b, h);
    }
}

void uenr_attn_fwd16(float *out, uint16_t *probs,
                     const float *q, const float *k, const float *v,
                     int B, int S, int H, int G, int hd, float scale,
                     int window, float softcap, const int *doc_start,
                     UenrPool *pool) {
    Fwd16Ctx c = { B, S, H, G, hd, scale, window, softcap, q, k, v, out, probs,
                   doc_start };
    uenr_pool_range(pool, attn_fwd16_range, &c, B * H);
}

typedef struct {
    int B, S, H, G, hd;
    float scale;
    int window;
    float softcap;
    const float *q, *k, *v, *dout;
    const uint16_t *probs;
    float *dq, *dk, *dv;
    float *dprob;
    const int *doc_start;
} Bwd16Ctx;

static void attn_bwd16_bg(Bwd16Ctx *c, int b, int g, int worker) {
    const int S = c->S, H = c->H, G = c->G, hd = c->hd;
    const int group = H / G;
    float *dprob = c->dprob + (size_t)worker * S;

    for (int s = 0; s < S; s++) {
        int t = b * S + s;
        for (int h = g * group; h < (g + 1) * group; h++) {
            const float *qr = c->q + ((size_t)t * H + h) * hd;
            const float *dor = c->dout + ((size_t)t * H + h) * hd;
            float *dqr = c->dq + ((size_t)t * H + h) * hd;
            const uint16_t *prow = c->probs + (((size_t)b * H + h) * S + s) * S;
            int ds = c->doc_start ? c->doc_start[t] : 0;
            if (ds < 0) continue;
            int j0 = (c->window > 0 && s >= c->window) ? s - c->window + 1 : 0;
            if (j0 < ds) j0 = ds;

            float dotp = 0.0f;
            for (int j = j0; j <= s; j++) {
                const float *vr = c->v + ((size_t)(b * S + j) * G + g) * hd;
                float dp = dot_f(dor, vr, hd);
                dprob[j] = dp;
                dotp += h2f16(prow[j]) * dp;
            }
            for (int j = j0; j <= s; j++) {
                const float *kr = c->k + ((size_t)(b * S + j) * G + g) * hd;
                float p = h2f16(prow[j]);
                float ds = p * (dprob[j] - dotp);
                float raw = dot_f(qr, kr, hd) * c->scale;
                float cap_deriv = 1.0f;
                if (c->softcap > 0.0f &&
                    (raw > 0.1f * c->softcap || raw < -0.1f * c->softcap)) {
                    float tt = uenr_tanhf(raw / c->softcap);
                    cap_deriv = 1.0f - tt * tt;
                }
                float dsc = ds * cap_deriv * c->scale;
                float *dkr = c->dk + ((size_t)(b * S + j) * G + g) * hd;
                float *dvr = c->dv + ((size_t)(b * S + j) * G + g) * hd;
                axpy_f(dqr, kr, dsc, hd);
                axpy_f(dkr, qr, dsc, hd);
                axpy_f(dvr, dor, p, hd);
            }
        }
    }
}

static void attn_bwd16_range(void *ctx, int start, int end, int worker) {
    Bwd16Ctx *c = (Bwd16Ctx *)ctx;
    for (int idx = start; idx < end; idx++) {
        int b = idx / c->G;
        int g = idx % c->G;
        attn_bwd16_bg(c, b, g, worker);
    }
}

UenrResult uenr_attn_bwd16(float *dq, float *dk, float *dv,
                           const float *dout, const float *q, const float *k,
                           const float *v, const uint16_t *probs,
                           int B, int S, int H, int G, int hd, float scale,
                           int window, float softcap, const int *doc_start,
                           float *scratch, UenrPool *pool) {
    int nw = pool ? uenr_pool_threads(pool) : 1;
    float *dprob = scratch;
    if (!dprob) {
        dprob = (float *)malloc((size_t)nw * S * sizeof(float));
        if (!dprob) return UENR_ERR_ALLOC;
    }
    Bwd16Ctx c = { B, S, H, G, hd, scale, window, softcap, q, k, v, dout,
                   probs, dq, dk, dv, dprob, doc_start };
    uenr_pool_range(pool, attn_bwd16_range, &c, B * G);
    if (!scratch) free(dprob);
    return UENR_OK;
}
