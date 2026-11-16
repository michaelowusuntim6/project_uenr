#include "uenr/ops.h"
#include "uenr/math.h"
#include "uenr/simd.h"

void uenr_zero(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = 0.0f;
}

void uenr_scale(float *x, size_t n, float s) {
    size_t i = 0;
    __m256 sv = _mm256_set1_ps(s);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), sv));
    for (; i < n; i++) x[i] *= s;
}

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

float uenr_l2_norm(const float *x, size_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256 v0 = _mm256_loadu_ps(x + i);
        __m256 v1 = _mm256_loadu_ps(x + i + 8);
        __m256 v2 = _mm256_loadu_ps(x + i + 16);
        __m256 v3 = _mm256_loadu_ps(x + i + 24);
        a0 = _mm256_add_ps(a0, _mm256_mul_ps(v0, v0));
        a1 = _mm256_add_ps(a1, _mm256_mul_ps(v1, v1));
        a2 = _mm256_add_ps(a2, _mm256_mul_ps(v2, v2));
        a3 = _mm256_add_ps(a3, _mm256_mul_ps(v3, v3));
    }
    float s = uenr_v_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    for (; i < n; i++) s += x[i] * x[i];
    return uenr_sqrtf(s);
}

void uenr_adamw_step(float *p, const float *grad, float *m, float *v, size_t n,
                     int step, float lr, float b1, float b2, float eps,
                     float weight_decay) {
    float bc1 = 1.0f - uenr_expf((float)step * uenr_logf(b1));
    float bc2 = 1.0f - uenr_expf((float)step * uenr_logf(b2));
    float sqrt_bc2 = uenr_sqrtf(bc2);
    float c1 = lr * sqrt_bc2 / bc1;
    float c2 = eps * sqrt_bc2;

    size_t i = 0;
    __m256 vb1 = _mm256_set1_ps(b1);
    __m256 vb2 = _mm256_set1_ps(b2);
    __m256 om1 = _mm256_set1_ps(1.0f - b1);
    __m256 om2 = _mm256_set1_ps(1.0f - b2);
    __m256 vc1 = _mm256_set1_ps(c1);
    __m256 vc2 = _mm256_set1_ps(c2);
    __m256 vwd = _mm256_set1_ps(lr * weight_decay);
    __m256 zero = _mm256_setzero_ps();

    for (; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(grad + i);
        __m256 mm = _mm256_add_ps(_mm256_mul_ps(vb1, _mm256_loadu_ps(m + i)),
                                  _mm256_mul_ps(om1, g));
        __m256 vv = _mm256_add_ps(_mm256_mul_ps(vb2, _mm256_loadu_ps(v + i)),
                                  _mm256_mul_ps(om2, _mm256_mul_ps(g, g)));
        _mm256_storeu_ps(m + i, mm);
        _mm256_storeu_ps(v + i, vv);
#if defined(__AVX__)
        __m256 sqrt_v = _mm256_sqrt_ps(vv); /* safe at v=0; v*rsqrt(v) is NaN */
#else
        __m256 sqrt_v = _mm_sqrt_ps(vv);
#endif
        __m256 denom = _mm256_add_ps(sqrt_v, vc2);
        __m256 upd = _mm256_mul_ps(vc1, _mm256_mul_ps(mm, uenr_v_rcp(denom)));
        /* decoupled weight decay */
        __m256 pp = _mm256_loadu_ps(p + i);
        pp = _mm256_sub_ps(pp, _mm256_mul_ps(vwd, pp));
        pp = _mm256_sub_ps(pp, upd);
        _mm256_storeu_ps(p + i, pp);
        (void)zero;
    }
    for (; i < n; i++) {
        float g = grad[i];
        m[i] = b1 * m[i] + (1.0f - b1) * g;
        v[i] = b2 * v[i] + (1.0f - b2) * g * g;
        float upd = c1 * m[i] * uenr_rcpf(uenr_sqrtf(v[i]) + c2);
        p[i] -= lr * weight_decay * p[i] + upd;
    }
}
