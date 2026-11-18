/* AVX2-only vector exp used by the softmax kernels.
 * The polynomial is mul+add (uenr_v_muladd) by design: only linear-algebra
 * kernels use FMA, so this matches the scalar/AVX1 exp polynomial. */
#ifndef UENR_OPS_AVX2_MATH_H
#define UENR_OPS_AVX2_MATH_H

#include <immintrin.h>
#include "uenr/simd.h"

static inline __m256 uenr_avx2_exp8(__m256 x) {
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 inv_ln2 = _mm256_set1_ps(1.4426950408889634f);
    const __m256 hi = _mm256_set1_ps(88.0f);
    const __m256 lo = _mm256_set1_ps(-88.0f);
    const __m256 zero = _mm256_set1_ps(0.0f);
    __m256 hi_mask = _mm256_cmp_ps(x, hi, _CMP_GT_OQ);
    __m256 lo_mask = _mm256_cmp_ps(x, lo, _CMP_LT_OQ);

    x = _mm256_max_ps(_mm256_min_ps(x, hi), lo);
    __m256 kf = _mm256_round_ps(_mm256_mul_ps(x, inv_ln2),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(kf, ln2));

    __m256 p = _mm256_set1_ps(1.0f / 720.0f);
    p = uenr_v_muladd(p, r, _mm256_set1_ps(1.0f / 120.0f));
    p = uenr_v_muladd(p, r, _mm256_set1_ps(1.0f / 24.0f));
    p = uenr_v_muladd(p, r, _mm256_set1_ps(1.0f / 6.0f));
    p = uenr_v_muladd(p, r, _mm256_set1_ps(0.5f));
    p = uenr_v_muladd(p, r, _mm256_set1_ps(1.0f));
    p = uenr_v_muladd(p, r, _mm256_set1_ps(1.0f));

    __m256 out = _mm256_mul_ps(p, uenr_v_pow2i(kf));
    out = _mm256_blendv_ps(out, _mm256_set1_ps(3.402823466e+38f), hi_mask);
    return _mm256_blendv_ps(out, zero, lo_mask);
}

#endif
