/* AVX-512 4x16 GEMM microkernel, opt-in only.
 * Zen 4/4c double-pumps 512-bit FP, so this is not expected to double AVX2;
 * it exists to measure whether fewer instructions and 16 lanes win.  Called
 * only when uenr_simd_level() == UENR_SIMD_AVX512. */
#include <immintrin.h>

static inline void store16(float *c, __m512 v, float alpha, float beta) {
    if (alpha != 1.0f) v = _mm512_mul_ps(v, _mm512_set1_ps(alpha));
    if (beta == 0.0f) {
        _mm512_storeu_ps(c, v);
    } else {
        __m512 old = _mm512_loadu_ps(c);
        _mm512_storeu_ps(c, _mm512_fmadd_ps(_mm512_set1_ps(beta), old, v));
    }
}

void uenr_gemm_avx512_4x16(int K,
                           const float *a0, const float *a1,
                           const float *a2, const float *a3, int a_kstride,
                           const float *bp, int b_kstride,
                           float *c0, float *c1, float *c2, float *c3,
                           float alpha, float beta) {
    __m512 q0 = _mm512_setzero_ps(), q1 = _mm512_setzero_ps();
    __m512 q2 = _mm512_setzero_ps(), q3 = _mm512_setzero_ps();
    const float *p0 = a0, *p1 = a1, *p2 = a2, *p3 = a3, *pb = bp;
    for (int k = 0; k < K; k++) {
        __m512 b = _mm512_loadu_ps(pb);
        q0 = _mm512_fmadd_ps(_mm512_set1_ps(p0[0]), b, q0);
        q1 = _mm512_fmadd_ps(_mm512_set1_ps(p1[0]), b, q1);
        q2 = _mm512_fmadd_ps(_mm512_set1_ps(p2[0]), b, q2);
        q3 = _mm512_fmadd_ps(_mm512_set1_ps(p3[0]), b, q3);
        p0 += a_kstride; p1 += a_kstride; p2 += a_kstride; p3 += a_kstride;
        pb += b_kstride;
    }
    store16(c0, q0, alpha, beta);
    store16(c1, q1, alpha, beta);
    store16(c2, q2, alpha, beta);
    store16(c3, q3, alpha, beta);
}
