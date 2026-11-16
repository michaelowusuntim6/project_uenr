#include "uenr/ops.h"
#include "uenr/math.h"
#include "uenr/simd.h"
#include "uenr/thread.h"

#include <stddef.h>

void uenr_zero(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = 0.0f;
}

typedef struct { float *x; size_t n; float s; } ScaleCtx;

static void scale_range(void *ctx, int start, int end, int worker) {
    (void)worker;
    ScaleCtx *c = (ScaleCtx *)ctx;
    size_t i = (size_t)start;
    size_t n = (size_t)end;
    __m256 sv = _mm256_set1_ps(c->s);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(c->x + i, _mm256_mul_ps(_mm256_loadu_ps(c->x + i), sv));
    for (; i < n; i++) c->x[i] *= c->s;
}

void uenr_scale_p(UenrPool *pool, float *x, size_t n, float s) {
    ScaleCtx c = { x, n, s };
    uenr_pool_range(pool, scale_range, &c, (int)n);
}

void uenr_scale(float *x, size_t n, float s) { uenr_scale_p(NULL, x, n, s); }

void uenr_copy(float *dst, const float *src, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = src[i];
}

void uenr_add_scaled(float *dst, const float *src, float s, size_t n) {
    size_t i = 0;
    __m256 sv = _mm256_set1_ps(s);
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(dst + i);
        __m256 b = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(a, _mm256_mul_ps(b, sv)));
    }
    for (; i < n; i++) dst[i] += src[i] * s;
}

typedef struct { const float *x; size_t n; float *parts; } NormCtx;

static void norm_range(void *ctx, int start, int end, int worker) {
    NormCtx *c = (NormCtx *)ctx;
    size_t i = (size_t)start, e = (size_t)end;
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (; i + 32 <= e; i += 32) {
        __m256 v0 = _mm256_loadu_ps(c->x + i);
        __m256 v1 = _mm256_loadu_ps(c->x + i + 8);
        __m256 v2 = _mm256_loadu_ps(c->x + i + 16);
        __m256 v3 = _mm256_loadu_ps(c->x + i + 24);
        a0 = _mm256_add_ps(a0, _mm256_mul_ps(v0, v0));
        a1 = _mm256_add_ps(a1, _mm256_mul_ps(v1, v1));
        a2 = _mm256_add_ps(a2, _mm256_mul_ps(v2, v2));
        a3 = _mm256_add_ps(a3, _mm256_mul_ps(v3, v3));
    }
    float s = uenr_v_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    for (; i < e; i++) s += c->x[i] * c->x[i];
    c->parts[worker] += s;
}

float uenr_l2_norm_p(UenrPool *pool, const float *x, size_t n) {
    int nw = pool ? uenr_pool_threads(pool) : 1;
    if (nw > 64) nw = 64;
    float parts[64];
    for (int i = 0; i < 64; i++) parts[i] = 0.0f;
    NormCtx c = { x, n, parts };
    uenr_pool_range(pool, norm_range, &c, (int)n);
    double s = 0.0;
    for (int i = 0; i < nw; i++) s += parts[i];
    return uenr_sqrtf((float)s);
}

float uenr_l2_norm(const float *x, size_t n) { return uenr_l2_norm_p(NULL, x, n); }

typedef struct {
    float *p;
    const float *grad;
    float *m, *v;
    size_t n;
    float c1, c2, b1, b2, om1, om2, vwd;
} AdamCtx;

static void adam_range(void *ctx, int start, int end, int worker) {
    (void)worker;
    AdamCtx *c = (AdamCtx *)ctx;
    size_t i = (size_t)start, e = (size_t)end;
    __m256 vb1 = _mm256_set1_ps(c->b1);
    __m256 vb2 = _mm256_set1_ps(c->b2);
    __m256 om1 = _mm256_set1_ps(c->om1);
    __m256 om2 = _mm256_set1_ps(c->om2);
    __m256 vc1 = _mm256_set1_ps(c->c1);
    __m256 vc2 = _mm256_set1_ps(c->c2);
    __m256 vwd = _mm256_set1_ps(c->vwd);
    for (; i + 8 <= e; i += 8) {
        __m256 g = _mm256_loadu_ps(c->grad + i);
        __m256 mm = _mm256_add_ps(_mm256_mul_ps(vb1, _mm256_loadu_ps(c->m + i)),
                                  _mm256_mul_ps(om1, g));
        __m256 vv = _mm256_add_ps(_mm256_mul_ps(vb2, _mm256_loadu_ps(c->v + i)),
                                  _mm256_mul_ps(om2, _mm256_mul_ps(g, g)));
        _mm256_storeu_ps(c->m + i, mm);
        _mm256_storeu_ps(c->v + i, vv);
#if defined(__AVX__)
        __m256 sqrt_v = _mm256_sqrt_ps(vv);
#else
        __m256 sqrt_v = _mm_sqrt_ps(vv);
#endif
        __m256 denom = _mm256_add_ps(sqrt_v, vc2);
        __m256 upd = _mm256_mul_ps(vc1, _mm256_mul_ps(mm, uenr_v_rcp(denom)));
        __m256 pp = _mm256_loadu_ps(c->p + i);
        pp = _mm256_sub_ps(pp, _mm256_mul_ps(vwd, pp));
        pp = _mm256_sub_ps(pp, upd);
        _mm256_storeu_ps(c->p + i, pp);
    }
    for (; i < e; i++) {
        float g = c->grad[i];
        c->m[i] = c->b1 * c->m[i] + c->om1 * g;
        c->v[i] = c->b2 * c->v[i] + c->om2 * g * g;
        float upd = c->c1 * c->m[i] * uenr_rcpf(uenr_sqrtf(c->v[i]) + c->c2);
        c->p[i] -= c->vwd * c->p[i] + upd;
    }
}

void uenr_adamw_step(float *p, const float *grad, float *m, float *v, size_t n,
                     int step, float lr, float b1, float b2, float eps,
                     float weight_decay, UenrPool *pool) {
    float bc1 = 1.0f - uenr_expf((float)step * uenr_logf(b1));
    float bc2 = 1.0f - uenr_expf((float)step * uenr_logf(b2));
    float sqrt_bc2 = uenr_sqrtf(bc2);
    AdamCtx c = { p, grad, m, v, n,
                  lr * sqrt_bc2 / bc1, eps * sqrt_bc2,
                  b1, b2, 1.0f - b1, 1.0f - b2, lr * weight_decay };
    uenr_pool_range(pool, adam_range, &c, (int)n);
}
