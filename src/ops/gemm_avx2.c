/* AVX2 + FMA GEMM/GEMV microkernels.
 *
 * This translation unit is compiled with -march=x86-64-v3 -mavx2 -mfma.  It
 * is never called unless uenr_simd_level() == UENR_SIMD_AVX2, so the Ivy
 * Bridge build still runs the original AVX1 kernels in gemm.c.
 *
 * The 4x16 kernel uses eight 256-bit accumulators (four rows x two column
 * vectors).  Per k step it issues six memory operations (four broadcasts,
 * two B loads) and eight FMAs; Zen 4/3 can retire two FMAs per cycle, so the
 * FMA pipes are the bottleneck rather than the load ports.  The 8x8 variant
 * (eight row accumulators, one column vector) is kept for benchmark A/B.
 */
#include <immintrin.h>

static inline int imin_a(int a, int b) { return a < b ? a : b; }

static inline void store8(float *c, __m256 v, float alpha, float beta) {
    if (alpha != 1.0f) v = _mm256_mul_ps(v, _mm256_set1_ps(alpha));
    if (beta == 0.0f) {
        _mm256_storeu_ps(c, v);
    } else {
        __m256 old = _mm256_loadu_ps(c);
        _mm256_storeu_ps(c, _mm256_fmadd_ps(_mm256_set1_ps(beta), old, v));
    }
}

void uenr_gemm_avx2_4x16(int K,
                         const float *a0, const float *a1,
                         const float *a2, const float *a3, int a_kstride,
                         const float *bp, int b_kstride,
                         float *c0, float *c1, float *c2, float *c3, int ldc,
                         float alpha, float beta) {
    __m256 l0 = _mm256_setzero_ps(), l1 = _mm256_setzero_ps();
    __m256 l2 = _mm256_setzero_ps(), l3 = _mm256_setzero_ps();
    __m256 h0 = _mm256_setzero_ps(), h1 = _mm256_setzero_ps();
    __m256 h2 = _mm256_setzero_ps(), h3 = _mm256_setzero_ps();
    const float *p0 = a0, *p1 = a1, *p2 = a2, *p3 = a3, *pb = bp;
    for (int k = 0; k < K; k++) {
        __m256 b0 = _mm256_loadu_ps(pb);
        __m256 b1 = _mm256_loadu_ps(pb + 8);
        __m256 x0 = _mm256_broadcast_ss(p0);
        __m256 x1 = _mm256_broadcast_ss(p1);
        __m256 x2 = _mm256_broadcast_ss(p2);
        __m256 x3 = _mm256_broadcast_ss(p3);
        l0 = _mm256_fmadd_ps(x0, b0, l0);
        l1 = _mm256_fmadd_ps(x1, b0, l1);
        l2 = _mm256_fmadd_ps(x2, b0, l2);
        l3 = _mm256_fmadd_ps(x3, b0, l3);
        h0 = _mm256_fmadd_ps(x0, b1, h0);
        h1 = _mm256_fmadd_ps(x1, b1, h1);
        h2 = _mm256_fmadd_ps(x2, b1, h2);
        h3 = _mm256_fmadd_ps(x3, b1, h3);
        p0 += a_kstride; p1 += a_kstride; p2 += a_kstride; p3 += a_kstride;
        pb += b_kstride;
    }
    store8(c0, l0, alpha, beta); store8(c0 + 8, h0, alpha, beta);
    store8(c1, l1, alpha, beta); store8(c1 + 8, h1, alpha, beta);
    store8(c2, l2, alpha, beta); store8(c2 + 8, h2, alpha, beta);
    store8(c3, l3, alpha, beta); store8(c3 + 8, h3, alpha, beta);
    (void)ldc;
}

void uenr_gemm_avx2_4x8(int K,
                        const float *a0, const float *a1,
                        const float *a2, const float *a3, int a_kstride,
                        const float *bp, int b_kstride,
                        float *c0, float *c1, float *c2, float *c3,
                        float alpha, float beta) {
    __m256 q0 = _mm256_setzero_ps(), q1 = _mm256_setzero_ps();
    __m256 q2 = _mm256_setzero_ps(), q3 = _mm256_setzero_ps();
    const float *p0 = a0, *p1 = a1, *p2 = a2, *p3 = a3, *pb = bp;
    for (int k = 0; k < K; k++) {
        __m256 b = _mm256_loadu_ps(pb);
        q0 = _mm256_fmadd_ps(_mm256_broadcast_ss(p0), b, q0);
        q1 = _mm256_fmadd_ps(_mm256_broadcast_ss(p1), b, q1);
        q2 = _mm256_fmadd_ps(_mm256_broadcast_ss(p2), b, q2);
        q3 = _mm256_fmadd_ps(_mm256_broadcast_ss(p3), b, q3);
        p0 += a_kstride; p1 += a_kstride; p2 += a_kstride; p3 += a_kstride;
        pb += b_kstride;
    }
    store8(c0, q0, alpha, beta); store8(c1, q1, alpha, beta);
    store8(c2, q2, alpha, beta); store8(c3, q3, alpha, beta);
}

void uenr_gemm_avx2_8x8(int K, const float *a, int a_rstride, int a_kstride,
                        const float *bp, int b_kstride,
                        float *c, int ldc, float alpha, float beta) {
    __m256 acc[8];
    for (int i = 0; i < 8; i++) acc[i] = _mm256_setzero_ps();
    const float *ar = a, *pb = bp;
    for (int k = 0; k < K; k++) {
        __m256 b = _mm256_loadu_ps(pb);
        for (int i = 0; i < 8; i++)
            acc[i] = _mm256_fmadd_ps(_mm256_broadcast_ss(ar + (size_t)i * a_rstride), b, acc[i]);
        ar += a_kstride; pb += b_kstride;
    }
    for (int i = 0; i < 8; i++) store8(c + (size_t)i * ldc, acc[i], alpha, beta);
}

void uenr_gemm_avx2_1x8(int K, const float *a0, int a_kstride,
                        const float *bp, int b_kstride,
                        float *c0, float alpha, float beta) {
    __m256 acc = _mm256_setzero_ps();
    const float *p0 = a0, *pb = bp;
    for (int k = 0; k < K; k++) {
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(p0), _mm256_loadu_ps(pb), acc);
        p0 += a_kstride; pb += b_kstride;
    }
    store8(c0, acc, alpha, beta);
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

void uenr_gemm_avx2_gemv(int N, int K,
                         const float *x, const float *B, int ldb, int transB,
                         float *y, float alpha, float beta) {
    if (!transB) {
        for (int n0 = 0; n0 < N; n0 += 8) {
            int nb = imin_a(8, N - n0);
            __m256 acc = _mm256_setzero_ps();
            if (nb == 8) {
                for (int k = 0; k < K; k++) {
                    __m256 bv = _mm256_loadu_ps(B + (size_t)k * ldb + n0);
                    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(x + k), bv, acc);
                }
                float out[8];
                _mm256_storeu_ps(out, acc);
                for (int j = 0; j < 8; j++)
                    y[n0 + j] = alpha * out[j] + beta * y[n0 + j];
            } else {
                for (int j = 0; j < nb; j++) {
                    const float *row = B + (size_t)n0 + j;
                    float s = 0.0f;
                    for (int k = 0; k < K; k++)
                        s = __builtin_fmaf(x[k], B[(size_t)k * ldb + n0 + j], s);
                    (void)row;
                    y[n0 + j] = alpha * s + beta * y[n0 + j];
                }
            }
        }
        return;
    }
    for (int n = 0; n < N; n++) {
        const float *row = B + (size_t)n * ldb;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        int k = 0;
        for (; k + 32 <= K; k += 32) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k), _mm256_loadu_ps(row + k), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 8), _mm256_loadu_ps(row + k + 8), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 16), _mm256_loadu_ps(row + k + 16), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 24), _mm256_loadu_ps(row + k + 24), a3);
        }
        float s = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        for (; k < K; k++) s = __builtin_fmaf(x[k], row[k], s);
        y[n] = alpha * s + beta * y[n];
    }
}
