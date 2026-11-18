/* AVX2 final-logit softcap.  The tanh/sigmoid polynomial is the shared
 * non-FMA vector polynomial, so results match the scalar path; only the
 * 8-wide application changes.  Compiled with -mavx2 -mfma but uses no FMA. */
#include <immintrin.h>
#include <stddef.h>

#include "uenr/math.h"
#include "uenr/simd.h"

void uenr_softcap_fwd_avx2(float *x, size_t n, float sc) {
    const __m256 vsc = _mm256_set1_ps(sc);
    const __m256 thr = _mm256_set1_ps(0.1f * sc);
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 z = _mm256_loadu_ps(x + i);
        __m256 az = _mm256_and_ps(z, absmask);
        __m256 mask = _mm256_cmp_ps(az, thr, _CMP_GT_OQ);
        __m256 t = uenr_v_tanh(_mm256_div_ps(z, vsc));
        __m256 capped = _mm256_mul_ps(vsc, t);
        _mm256_storeu_ps(x + i, _mm256_blendv_ps(z, capped, mask));
    }
    for (; i < n; i++) {
        float z = x[i];
        if (z > 0.1f * sc || z < -0.1f * sc)
            x[i] = sc * uenr_tanhf(z / sc);
    }
}

void uenr_softcap_grad_avx2(float *d, const float *y, size_t n, float sc) {
    const __m256 vsc = _mm256_set1_ps(sc);
    const __m256 thr = _mm256_set1_ps(0.1f * sc);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 z = _mm256_loadu_ps(y + i);
        __m256 az = _mm256_and_ps(z, absmask);
        __m256 mask = _mm256_cmp_ps(az, thr, _CMP_GT_OQ);
        __m256 t = _mm256_div_ps(z, vsc);
        __m256 f = _mm256_sub_ps(one, _mm256_mul_ps(t, t));
        __m256 dv = _mm256_loadu_ps(d + i);
        _mm256_storeu_ps(d + i, _mm256_blendv_ps(dv, _mm256_mul_ps(dv, f), mask));
    }
    for (; i < n; i++) {
        float z = y[i];
        if (z > 0.1f * sc || z < -0.1f * sc) {
            float t = z / sc;
            d[i] *= 1.0f - t * t;
        }
    }
}
