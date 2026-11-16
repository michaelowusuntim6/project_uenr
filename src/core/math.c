#include "uenr/math.h"

#include <stdint.h>

/* ------------------------------------------------------------------------- */
/* Scalar building blocks                                                    */
/* ------------------------------------------------------------------------- */

static inline float uenr_pow2i(int k) {
    if (k < -126) return 0.0f;
    if (k > 127) return 3.402823466e+38f; /* FLT_MAX */
    union { float f; uint32_t u; } v;
    v.u = (uint32_t)(k + 127) << 23;
    return v.f;
}

static inline float uenr_round_nearest(float x) {
    return (float)(int)(x + (x >= 0.0f ? 0.5f : -0.5f));
}

/* exp(x) = 2^k * exp(r), x = k*ln2 + r, |r| <= ln2/2.
 * Degree-6 Horner; relative error < 2e-7 over the reduced range. */
float uenr_expf(float x) {
    const float ln2 = 0.6931471805599453f;
    const float inv_ln2 = 1.4426950408889634f;

    if (x > 88.0f)  return 3.402823466e+38f;
    if (x < -88.0f) return 0.0f;

    int k = (int)uenr_round_nearest(x * inv_ln2);
    float r = x - (float)k * ln2;

    float p = 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;

    return p * uenr_pow2i(k);
}

/* log(x): split exponent/mantissa, reduce m to [sqrt(1/2), sqrt(2)),
 * log(m) = 2*atanh(s), s = (m-1)/(m+1), |s| <= 0.1716.
 * Odd series through s^9. */
float uenr_logf(float x) {
    if (x <= 0.0f) return -3.402823466e+38f;

    union { float f; uint32_t u; } v;
    v.f = x;
    int e = (int)((v.u >> 23) & 0xffu) - 127;
    v.u = (v.u & 0x007fffffu) | 0x3f800000u; /* m in [1,2) */
    float m = v.f;

    if (m > 1.41421356237f) {
        m *= 0.5f;
        e += 1;
    }

    float s = (m - 1.0f) / (m + 1.0f);
    float s2 = s * s;

    float p = 1.0f / 11.0f;
    p = p * s2 + 1.0f / 9.0f;
    p = p * s2 + 1.0f / 7.0f;
    p = p * s2 + 1.0f / 5.0f;
    p = p * s2 + 1.0f / 3.0f;
    p = p * s2 + 1.0f;

    float logm = 2.0f * s * p;
    return (float)e * 0.6931471805599453f + logm;
}

/* Quadrant-reduced Taylor polynomials on |r| <= pi/4. */
float uenr_sinf(float x) {
    const float two_over_pi = 0.6366197723675813f;
    const float pio2 = 1.5707963267948966f;

    int k = (int)uenr_round_nearest(x * two_over_pi);
    float r = x - (float)k * pio2;
    int q = k & 3;

    float r2 = r * r;
    float sp = -1.0f / 362880.0f;
    sp = sp * r2 + 1.0f / 5040.0f;
    sp = sp * r2 - 1.0f / 120.0f;
    sp = sp * r2 + 1.0f / 6.0f;
    sp = sp * r2 - 1.0f;
    float sr = r * -sp; /* r - r^3/6 + r^5/120 - r^7/5040 + r^9/362880 */

    float cp = 1.0f / 40320.0f;
    cp = cp * r2 - 1.0f / 720.0f;
    cp = cp * r2 + 1.0f / 24.0f;
    cp = cp * r2 - 0.5f;
    cp = cp * r2 + 1.0f;
    float cr = cp; /* 1 - r^2/2 + r^4/24 - r^6/720 + r^8/40320 */

    switch (q) {
    case 0: return sr;
    case 1: return cr;
    case 2: return -sr;
    default: return -cr;
    }
}

float uenr_cosf(float x) {
    return uenr_sinf(x + 1.5707963267948966f);
}

float uenr_rsqrtf(float x) {
    __m128 xv = _mm_set_ss(x);
    __m128 y = _mm_rsqrt_ss(xv);
    __m128 half_x = _mm_mul_ss(_mm_set_ss(0.5f), xv);
    __m128 t = _mm_sub_ss(_mm_set_ss(1.5f), _mm_mul_ss(_mm_mul_ss(half_x, y), y));
    y = _mm_mul_ss(y, t);
    return _mm_cvtss_f32(y);
}

float uenr_rcpf(float x) {
    __m128 xv = _mm_set_ss(x);
    __m128 y = _mm_rcp_ss(xv);
    __m128 t = _mm_sub_ss(_mm_set_ss(2.0f), _mm_mul_ss(xv, y));
    y = _mm_mul_ss(y, t);
    return _mm_cvtss_f32(y);
}

float uenr_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;
    return x * uenr_rsqrtf(x);
}

float uenr_sigmoidf(float x) {
    return uenr_rcpf(1.0f + uenr_expf(-x));
}

float uenr_siluf(float x) {
    return x * uenr_sigmoidf(x);
}

float uenr_tanhf(float x) {
    return 2.0f * uenr_sigmoidf(2.0f * x) - 1.0f;
}

/* GELU tanh approximation: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))). */
float uenr_geluf(float x) {
    const float k = 0.7978845608028654f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + uenr_tanhf(k * (x + 0.044715f * x3)));
}

/* ------------------------------------------------------------------------- */
/* Vector versions                                                           */
/* ------------------------------------------------------------------------- */

uenr_v8f uenr_v_exp(uenr_v8f x) {
    const uenr_v8f ln2 = uenr_v_set1(0.6931471805599453f);
    const uenr_v8f inv_ln2 = uenr_v_set1(1.4426950408889634f);
    const uenr_v8f hi = uenr_v_set1(88.0f);
    const uenr_v8f lo = uenr_v_set1(-88.0f);

    x = uenr_v_max(uenr_v_min(x, hi), lo);

#if defined(__AVX__)
    uenr_v8f kf = _mm256_round_ps(_mm256_mul_ps(x, inv_ln2),
                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
    uenr_v8f kf = _mm_round_ps(_mm_mul_ps(x, inv_ln2),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#endif
    uenr_v8f r = uenr_v_sub(x, uenr_v_mul(kf, ln2));

    uenr_v8f p = uenr_v_set1(1.0f / 720.0f);
    p = uenr_v_muladd(p, r, uenr_v_set1(1.0f / 120.0f));
    p = uenr_v_muladd(p, r, uenr_v_set1(1.0f / 24.0f));
    p = uenr_v_muladd(p, r, uenr_v_set1(1.0f / 6.0f));
    p = uenr_v_muladd(p, r, uenr_v_set1(0.5f));
    p = uenr_v_muladd(p, r, uenr_v_set1(1.0f));
    p = uenr_v_muladd(p, r, uenr_v_set1(1.0f));

    return uenr_v_mul(p, uenr_v_pow2i(kf));
}

uenr_v8f uenr_v_sigmoid(uenr_v8f x) {
    uenr_v8f e = uenr_v_exp(uenr_v_sub(uenr_v_zero(), x));
    return uenr_v_rcp(uenr_v_add(uenr_v_set1(1.0f), e));
}

uenr_v8f uenr_v_silu(uenr_v8f x) {
    return uenr_v_mul(x, uenr_v_sigmoid(x));
}

uenr_v8f uenr_v_tanh(uenr_v8f x) {
    uenr_v8f s = uenr_v_sigmoid(uenr_v_add(x, x));
    return uenr_v_sub(uenr_v_add(s, s), uenr_v_set1(1.0f));
}

uenr_v8f uenr_v_gelu(uenr_v8f x) {
    const uenr_v8f k = uenr_v_set1(0.7978845608028654f);
    const uenr_v8f c = uenr_v_set1(0.044715f);
    uenr_v8f x3 = uenr_v_mul(uenr_v_mul(x, x), x);
    uenr_v8f inner = uenr_v_mul(k, uenr_v_add(x, uenr_v_mul(c, x3)));
    uenr_v8f t = uenr_v_tanh(inner);
    return uenr_v_mul(uenr_v_set1(0.5f), uenr_v_mul(x, uenr_v_add(uenr_v_set1(1.0f), t)));
}
