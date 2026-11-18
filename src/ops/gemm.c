#include "uenr/gemm.h"
#include "uenr/io.h"
#include "uenr/simd.h"
#include "uenr/simd_dispatch.h"

#include <stdlib.h>
#include <string.h>

/* AVX2 microkernels (src/ops/gemm_avx2.c, compiled with -mavx2 -mfma). */
void uenr_gemm_avx2_4x16(int K, const float *a0, const float *a1,
                         const float *a2, const float *a3, int a_kstride,
                         const float *bp, int b_kstride,
                         float *c0, float *c1, float *c2, float *c3, int ldc,
                         float alpha, float beta);
void uenr_gemm_avx2_4x8(int K, const float *a0, const float *a1,
                        const float *a2, const float *a3, int a_kstride,
                        const float *bp, int b_kstride,
                        float *c0, float *c1, float *c2, float *c3,
                        float alpha, float beta);
void uenr_gemm_avx512_4x16(int K, const float *a0, const float *a1,
                           const float *a2, const float *a3, int a_kstride,
                           const float *bp, int b_kstride,
                           float *c0, float *c1, float *c2, float *c3,
                           float alpha, float beta);
void uenr_gemm_avx2_6x16(int K, const float *a0, const float *a1,
                         const float *a2, const float *a3,
                         const float *a4, const float *a5, int a_kstride,
                         const float *bp, int b_kstride,
                         float *c0, float *c1, float *c2,
                         float *c3, float *c4, float *c5, int ldc,
                         float alpha, float beta);
void uenr_gemm_avx2_2x16(int K, const float *a0, const float *a1,
                         int a_kstride, const float *bp, int b_kstride,
                         float *c0, float *c1, float alpha, float beta);
void uenr_gemm_avx2_8x8(int K, const float *a, int a_rstride, int a_kstride,
                        const float *bp, int b_kstride,
                        float *c, int ldc, float alpha, float beta);
void uenr_gemm_avx2_1x8(int K, const float *a0, int a_kstride,
                        const float *bp, int b_kstride,
                        float *c0, float alpha, float beta);
void uenr_gemm_avx2_gemv(int N, int K, const float *x, const float *B,
                         int ldb, int transB, float *y, float alpha, float beta);

typedef struct {
    UenrGemm *g;
    int M, N, K;
    const float *A; int lda, transA;
    const float *B; int ldb, transB;
    float *C; int ldc;
    float alpha, beta;

    /* Packed B panels for the transB==1 path. */
    const float *all_panels;
} GemmJob;

static inline int imin(int a, int b) { return a < b ? a : b; }

/* ------------------------------------------------------------------------- */
/* Microkernels. Ivy Bridge: no FMA, so each FMA-equivalent is mul+add.      */
/* ------------------------------------------------------------------------- */

static inline void micro_8x8(int K, const float *a, int a_rstride, int a_kstride,
                             const float *bp, int b_kstride,
                             float *c, int ldc,
                             float alpha, float beta) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
    const float *p0 = a, *pb = bp;
    for (int k = 0; k < K; k++) {
        __m256 b = _mm256_loadu_ps(pb);
        _mm_prefetch((const char *)(pb + b_kstride), _MM_HINT_T0);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_broadcast_ss(p0), b));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_broadcast_ss(p0 + a_rstride), b));
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_broadcast_ss(p0 + 2 * a_rstride), b));
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_broadcast_ss(p0 + 3 * a_rstride), b));
        acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(_mm256_broadcast_ss(p0 + 4 * a_rstride), b));
        acc5 = _mm256_add_ps(acc5, _mm256_mul_ps(_mm256_broadcast_ss(p0 + 5 * a_rstride), b));
        acc6 = _mm256_add_ps(acc6, _mm256_mul_ps(_mm256_broadcast_ss(p0 + 6 * a_rstride), b));
        acc7 = _mm256_add_ps(acc7, _mm256_mul_ps(_mm256_broadcast_ss(p0 + 7 * a_rstride), b));
        p0 += a_kstride; pb += b_kstride;
    }
    if (alpha != 1.0f) {
        __m256 al = _mm256_set1_ps(alpha);
        acc0 = _mm256_mul_ps(acc0, al); acc1 = _mm256_mul_ps(acc1, al);
        acc2 = _mm256_mul_ps(acc2, al); acc3 = _mm256_mul_ps(acc3, al);
        acc4 = _mm256_mul_ps(acc4, al); acc5 = _mm256_mul_ps(acc5, al);
        acc6 = _mm256_mul_ps(acc6, al); acc7 = _mm256_mul_ps(acc7, al);
    }
    __m256 acc[8] = { acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 };
    if (beta == 0.0f) {
        for (int i = 0; i < 8; i++) _mm256_storeu_ps(c + (size_t)i * ldc, acc[i]);
    } else {
        __m256 be = _mm256_set1_ps(beta);
        for (int i = 0; i < 8; i++) {
            float *ci = c + (size_t)i * ldc;
            _mm256_storeu_ps(ci, _mm256_add_ps(_mm256_mul_ps(be, _mm256_loadu_ps(ci)), acc[i]));
        }
    }
}

#ifdef UENR_6X16
static inline void micro_2x8(int K, const float *a0, const float *a1, int a_kstride,
                             const float *bp, int b_kstride,
                             float *c0, float *c1, int ldc, float alpha, float beta) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    const float *p0 = a0, *p1 = a1, *pb = bp;
    for (int k = 0; k < K; k++) {
        __m256 b = _mm256_loadu_ps(pb);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_broadcast_ss(p0), b));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_broadcast_ss(p1), b));
        p0 += a_kstride; p1 += a_kstride; pb += b_kstride;
    }
    if (alpha != 1.0f) {
        __m256 al = _mm256_set1_ps(alpha);
        acc0 = _mm256_mul_ps(acc0, al); acc1 = _mm256_mul_ps(acc1, al);
    }
    if (beta == 0.0f) {
        _mm256_storeu_ps(c0, acc0); _mm256_storeu_ps(c1, acc1);
    } else {
        __m256 be = _mm256_set1_ps(beta);
        _mm256_storeu_ps(c0, _mm256_add_ps(_mm256_mul_ps(be, _mm256_loadu_ps(c0)), acc0));
        _mm256_storeu_ps(c1, _mm256_add_ps(_mm256_mul_ps(be, _mm256_loadu_ps(c1)), acc1));
    }
}

static inline void micro_6x16(int K, const float *a, int a_rstride, int a_kstride,
                              const float *bp, int b_kstride,
                              float *c, int ldc, float alpha, float beta) {
    __m256 a0l = _mm256_setzero_ps(), a0h = _mm256_setzero_ps();
    __m256 a1l = _mm256_setzero_ps(), a1h = _mm256_setzero_ps();
    __m256 a2l = _mm256_setzero_ps(), a2h = _mm256_setzero_ps();
    __m256 a3l = _mm256_setzero_ps(), a3h = _mm256_setzero_ps();
    __m256 a4l = _mm256_setzero_ps(), a4h = _mm256_setzero_ps();
    __m256 a5l = _mm256_setzero_ps(), a5h = _mm256_setzero_ps();
    const float *p0 = a, *pb = bp;
    for (int k = 0; k < K; k++) {
        __m256 bl = _mm256_loadu_ps(pb);
        __m256 bh = _mm256_loadu_ps(pb + 8);
        __m256 x;
        x = _mm256_broadcast_ss(p0);
        a0l = _mm256_add_ps(a0l, _mm256_mul_ps(x, bl));
        a0h = _mm256_add_ps(a0h, _mm256_mul_ps(x, bh));
        x = _mm256_broadcast_ss(p0 + a_rstride);
        a1l = _mm256_add_ps(a1l, _mm256_mul_ps(x, bl));
        a1h = _mm256_add_ps(a1h, _mm256_mul_ps(x, bh));
        x = _mm256_broadcast_ss(p0 + 2 * a_rstride);
        a2l = _mm256_add_ps(a2l, _mm256_mul_ps(x, bl));
        a2h = _mm256_add_ps(a2h, _mm256_mul_ps(x, bh));
        x = _mm256_broadcast_ss(p0 + 3 * a_rstride);
        a3l = _mm256_add_ps(a3l, _mm256_mul_ps(x, bl));
        a3h = _mm256_add_ps(a3h, _mm256_mul_ps(x, bh));
        x = _mm256_broadcast_ss(p0 + 4 * a_rstride);
        a4l = _mm256_add_ps(a4l, _mm256_mul_ps(x, bl));
        a4h = _mm256_add_ps(a4h, _mm256_mul_ps(x, bh));
        x = _mm256_broadcast_ss(p0 + 5 * a_rstride);
        a5l = _mm256_add_ps(a5l, _mm256_mul_ps(x, bl));
        a5h = _mm256_add_ps(a5h, _mm256_mul_ps(x, bh));
        p0 += a_kstride; pb += b_kstride;
    }
    __m256* al[6] = { &a0l, &a1l, &a2l, &a3l, &a4l, &a5l };
    __m256* ah[6] = { &a0h, &a1h, &a2h, &a3h, &a4h, &a5h };
    __m256 alf = _mm256_set1_ps(alpha), be = _mm256_set1_ps(beta);
    for (int i = 0; i < 6; i++) {
        float *ci = c + (size_t)i * ldc;
        *al[i] = _mm256_mul_ps(*al[i], alf);
        *ah[i] = _mm256_mul_ps(*ah[i], alf);
        if (beta == 0.0f) {
            _mm256_storeu_ps(ci, *al[i]);
            _mm256_storeu_ps(ci + 8, *ah[i]);
        } else {
            _mm256_storeu_ps(ci, _mm256_add_ps(_mm256_mul_ps(be, _mm256_loadu_ps(ci)), *al[i]));
            _mm256_storeu_ps(ci + 8, _mm256_add_ps(_mm256_mul_ps(be, _mm256_loadu_ps(ci + 8)), *ah[i]));
        }
    }
}

#endif /* UENR_6X16 */

static inline void micro_4x8(int K,
                             const float *a0, const float *a1,
                             const float *a2, const float *a3,
                             int a_kstride,
                             const float *bp, int b_kstride,
                             float *c0, float *c1, float *c2, float *c3, int ldc,
                             float alpha, float beta) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    const float *p0 = a0, *p1 = a1, *p2 = a2, *p3 = a3, *pb = bp;
    for (int k = 0; k < K; k++) {
        __m256 b = _mm256_loadu_ps(pb);
        _mm_prefetch((const char *)(pb + b_kstride), _MM_HINT_T0);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_broadcast_ss(p0), b));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_broadcast_ss(p1), b));
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_broadcast_ss(p2), b));
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_broadcast_ss(p3), b));
        p0 += a_kstride; p1 += a_kstride; p2 += a_kstride; p3 += a_kstride;
        pb += b_kstride;
    }
    if (alpha != 1.0f) {
        __m256 al = _mm256_set1_ps(alpha);
        acc0 = _mm256_mul_ps(acc0, al); acc1 = _mm256_mul_ps(acc1, al);
        acc2 = _mm256_mul_ps(acc2, al); acc3 = _mm256_mul_ps(acc3, al);
    }
    if (beta == 0.0f) {
        _mm256_storeu_ps(c0, acc0); _mm256_storeu_ps(c1, acc1);
        _mm256_storeu_ps(c2, acc2); _mm256_storeu_ps(c3, acc3);
    } else {
        __m256 be = _mm256_set1_ps(beta);
        _mm256_storeu_ps(c0, _mm256_add_ps(_mm256_mul_ps(be, _mm256_loadu_ps(c0)), acc0));
        _mm256_storeu_ps(c1, _mm256_add_ps(_mm256_mul_ps(be, _mm256_loadu_ps(c1)), acc1));
        _mm256_storeu_ps(c2, _mm256_add_ps(_mm256_mul_ps(be, _mm256_loadu_ps(c2)), acc2));
        _mm256_storeu_ps(c3, _mm256_add_ps(_mm256_mul_ps(be, _mm256_loadu_ps(c3)), acc3));
    }
    (void)ldc;
}

static inline void micro_1x8(int K, const float *a0, int a_kstride,
                             const float *bp, int b_kstride,
                             float *c0, float alpha, float beta) {
    __m256 acc = _mm256_setzero_ps();
    const float *p0 = a0, *pb = bp;
    for (int k = 0; k < K; k++) {
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_broadcast_ss(p0), _mm256_loadu_ps(pb)));
        p0 += a_kstride; pb += b_kstride;
    }
    if (alpha != 1.0f) acc = _mm256_mul_ps(acc, _mm256_set1_ps(alpha));
    if (beta == 0.0f) {
        _mm256_storeu_ps(c0, acc);
    } else {
        __m256 be = _mm256_set1_ps(beta);
        _mm256_storeu_ps(c0, _mm256_add_ps(_mm256_mul_ps(be, _mm256_loadu_ps(c0)), acc));
    }
}

static inline float dot_scalar(int K, const float *a, int a_kstride,
                               const float *b, int b_kstride) {
    float s = 0.0f;
    for (int k = 0; k < K; k++) s += a[k * a_kstride] * b[k * b_kstride];
    return s;
}

static void pack_b_block(const float *B, int ldb, int n0, int nb, int K, float *panel) {
    for (int k = 0; k < K; k++) {
        float *dst = panel + (size_t)k * nb;
        for (int j = 0; j < nb; j++) dst[j] = B[(size_t)(n0 + j) * ldb + k];
    }
}

static void pack_b_all(const float *B, int ldb, int N, int K, float *packed) {
    for (int n0 = 0; n0 < N; n0 += UENR_GEMM_NB) {
        int nb = imin(UENR_GEMM_NB, N - n0);
        pack_b_block(B, ldb, n0, nb, K, packed + (size_t)n0 * K);
    }
}

/* One N-block for rows [m_start, m_end). panel==NULL means B is [K,N] row-major. */
static void block_kernel(const GemmJob *j, int n0, int nb, const float *panel,
                         int m_start, int m_end) {
    const int K = j->K;
    const int a_kstride = j->transA ? j->lda : 1;
    const int a_rstride = j->transA ? 1 : j->lda;
    const int bk = panel ? nb : j->ldb;

    const int avx2 = j->g->avx2;
    const int avx512 = j->g->avx512;
    const int tile8 = j->g->avx2_tile == 1;
    const int tile6 = j->g->avx2_tile == 2;
    for (int m = m_start; m < m_end; ) {
        const int mr = imin(8, m_end - m);
        const float *a0 = j->A + (size_t)m * a_rstride;

        int jcol = 0;
#ifdef UENR_6X16
        if (mr == 8 && j->M >= 32) {
            for (; jcol + 16 <= nb; jcol += 16) {
                const float *bp = panel ? (panel + jcol) : (j->B + (size_t)n0 + jcol);
                float *c0 = j->C + (size_t)m * j->ldc + n0 + jcol;
                micro_6x16(K, a0, a_rstride, a_kstride, bp, bk, c0, j->ldc,
                           j->alpha, j->beta);
                const float *a6 = a0 + 6 * a_rstride;
                micro_2x8(K, a6, a6 + a_rstride, a_kstride, bp, bk,
                          c0 + 6 * j->ldc, c0 + 7 * j->ldc, j->ldc,
                          j->alpha, j->beta);
                micro_2x8(K, a6, a6 + a_rstride, a_kstride, bp + 8, bk,
                          c0 + 6 * j->ldc + 8, c0 + 7 * j->ldc + 8, j->ldc,
                          j->alpha, j->beta);
            }
        }
#endif
        if (avx2 && tile6 && mr >= 6) {
            for (; jcol + 16 <= nb; jcol += 16) {
                const float *bp = panel ? (panel + jcol) : (j->B + (size_t)n0 + jcol);
                float *c0 = j->C + (size_t)m * j->ldc + n0 + jcol;
                uenr_gemm_avx2_6x16(K, a0, a0 + a_rstride,
                                    a0 + 2 * a_rstride, a0 + 3 * a_rstride,
                                    a0 + 4 * a_rstride, a0 + 5 * a_rstride,
                                    a_kstride, bp, bk, c0, c0 + j->ldc,
                                    c0 + 2 * j->ldc, c0 + 3 * j->ldc,
                                    c0 + 4 * j->ldc, c0 + 5 * j->ldc, j->ldc,
                                    j->alpha, j->beta);
                if (mr >= 8) {
                    uenr_gemm_avx2_2x16(K, a0 + 6 * a_rstride,
                                        a0 + 7 * a_rstride, a_kstride, bp, bk,
                                        c0 + 6 * j->ldc, c0 + 7 * j->ldc,
                                        j->alpha, j->beta);
                } else {
                    for (int i = 6; i < mr; i++) {
                        const float *ai = a0 + (size_t)i * a_rstride;
                        float *ci = c0 + (size_t)i * j->ldc;
                        uenr_gemm_avx2_1x8(K, ai, a_kstride, bp, bk, ci,
                                           j->alpha, j->beta);
                        uenr_gemm_avx2_1x8(K, ai, a_kstride, bp + 8, bk,
                                           ci + 8, j->alpha, j->beta);
                    }
                }
            }
        } else if (avx2 && !tile8 && !tile6 && mr >= 4) {
            /* 4x16: compute two 4-row groups per 16-column block.  A partial
             * tail block (mr 4..7) fills rows 4..mr-1 with 1x8 kernels for
             * both halves so the 8-column loop below never revisits them. */
            for (; jcol + 16 <= nb; jcol += 16) {
                const float *bp = panel ? (panel + jcol) : (j->B + (size_t)n0 + jcol);
                float *c0 = j->C + (size_t)m * j->ldc + n0 + jcol;
                if (avx512) {
                    uenr_gemm_avx512_4x16(K, a0, a0 + a_rstride,
                                          a0 + 2 * a_rstride, a0 + 3 * a_rstride,
                                          a_kstride, bp, bk, c0, c0 + j->ldc,
                                          c0 + 2 * j->ldc, c0 + 3 * j->ldc,
                                          j->alpha, j->beta);
                } else {
                    uenr_gemm_avx2_4x16(K, a0, a0 + a_rstride,
                                        a0 + 2 * a_rstride, a0 + 3 * a_rstride,
                                        a_kstride, bp, bk, c0, c0 + j->ldc,
                                        c0 + 2 * j->ldc, c0 + 3 * j->ldc,
                                        j->ldc, j->alpha, j->beta);
                }
                if (mr >= 8) {
                    if (avx512)
                        uenr_gemm_avx512_4x16(K, a0 + 4 * a_rstride,
                                              a0 + 5 * a_rstride,
                                              a0 + 6 * a_rstride,
                                              a0 + 7 * a_rstride,
                                              a_kstride, bp, bk,
                                              c0 + 4 * j->ldc, c0 + 5 * j->ldc,
                                              c0 + 6 * j->ldc, c0 + 7 * j->ldc,
                                              j->alpha, j->beta);
                    else
                        uenr_gemm_avx2_4x16(K, a0 + 4 * a_rstride,
                                            a0 + 5 * a_rstride,
                                            a0 + 6 * a_rstride,
                                            a0 + 7 * a_rstride,
                                            a_kstride, bp, bk,
                                            c0 + 4 * j->ldc, c0 + 5 * j->ldc,
                                            c0 + 6 * j->ldc, c0 + 7 * j->ldc,
                                            j->ldc, j->alpha, j->beta);
                } else {
                    for (int i = 4; i < mr; i++) {
                        const float *ai = a0 + (size_t)i * a_rstride;
                        float *ci = c0 + (size_t)i * j->ldc;
                        uenr_gemm_avx2_1x8(K, ai, a_kstride, bp, bk, ci,
                                           j->alpha, j->beta);
                        uenr_gemm_avx2_1x8(K, ai, a_kstride, bp + 8, bk,
                                           ci + 8, j->alpha, j->beta);
                    }
                }
            }
        }
        for (; jcol + 8 <= nb; jcol += 8) {
            const float *bp = panel ? (panel + jcol) : (j->B + (size_t)n0 + jcol);
            int done = 0;
            if (mr == 8) {
                float *c0 = j->C + (size_t)m * j->ldc + n0 + jcol;
                if (avx2) uenr_gemm_avx2_8x8(K, a0, a_rstride, a_kstride, bp,
                                             bk, c0, j->ldc, j->alpha, j->beta);
                else micro_8x8(K, a0, a_rstride, a_kstride, bp, bk, c0, j->ldc,
                               j->alpha, j->beta);
                done = 8;
            } else if (mr >= 4) {
                float *c0 = j->C + (size_t)m * j->ldc + n0 + jcol;
                if (avx2)
                    uenr_gemm_avx2_4x8(K, a0, a0 + a_rstride,
                                       a0 + 2 * a_rstride, a0 + 3 * a_rstride,
                                       a_kstride, bp, bk, c0, c0 + j->ldc,
                                       c0 + 2 * j->ldc, c0 + 3 * j->ldc,
                                       j->alpha, j->beta);
                else
                    micro_4x8(K, a0, a0 + a_rstride, a0 + 2 * a_rstride,
                              a0 + 3 * a_rstride, a_kstride, bp, bk,
                              c0, c0 + j->ldc, c0 + 2 * j->ldc, c0 + 3 * j->ldc,
                              j->ldc, j->alpha, j->beta);
                done = 4;
            }
            for (int i = done; i < mr; i++) {
                const float *ai = a0 + (size_t)i * a_rstride;
                float *ci = j->C + (size_t)(m + i) * j->ldc + n0 + jcol;
                if (avx2) uenr_gemm_avx2_1x8(K, ai, a_kstride, bp, bk, ci,
                                             j->alpha, j->beta);
                else micro_1x8(K, ai, a_kstride, bp, bk, ci, j->alpha, j->beta);
            }
        }
        for (int jj = jcol; jj < nb; jj++) {
            int n = n0 + jj;
            for (int i = 0; i < mr; i++) {
                const float *ai = a0 + (size_t)i * a_rstride;
                float v = panel ? dot_scalar(K, ai, a_kstride, panel + jj, nb)
                                : dot_scalar(K, ai, a_kstride, j->B + (size_t)n, j->ldb);
                float *ci = j->C + (size_t)(m + i) * j->ldc + n;
                *ci = j->alpha * v + j->beta * (*ci);
            }
        }
        m += mr;
    }
}

static void range_kernel(const GemmJob *j, int m_start, int m_end) {
    for (int n0 = 0; n0 < j->N; n0 += UENR_GEMM_NB) {
        int nb = imin(UENR_GEMM_NB, j->N - n0);
        block_kernel(j, n0, nb, NULL, m_start, m_end);
    }
}

static void worker_packed_all(void *ctx, int worker, int nworkers) {
    GemmJob *j = (GemmJob *)ctx;
    int m_per = (j->M + nworkers - 1) / nworkers;
    m_per = ((m_per + 7) / 8) * 8;
    int start = worker * m_per;
    int end = imin(start + m_per, j->M);
    if (start >= end) return;
    for (int n0 = 0; n0 < j->N; n0 += UENR_GEMM_NB) {
        int nb = imin(UENR_GEMM_NB, j->N - n0);
        block_kernel(j, n0, nb, j->all_panels + (size_t)n0 * j->K, start, end);
    }
}

static void worker_range(void *ctx, int worker, int nworkers) {
    GemmJob *j = (GemmJob *)ctx;
    int m_per = (j->M + nworkers - 1) / nworkers;
    m_per = ((m_per + 7) / 8) * 8;
    int start = worker * m_per;
    int end = imin(start + m_per, j->M);
    if (start < end) range_kernel(j, start, end);
}

/* Allocation-free transposed-B path.  Slower than the packed panels but it
 * always produces a complete, correct C, so an allocation failure degrades
 * performance instead of silently leaving stale output. */
static void direct_kernel_transB(const GemmJob *j, int m0, int m1) {
    const int a_kstride = j->transA ? j->lda : 1;
    const int a_rstride = j->transA ? 1 : j->lda;
    for (int n = 0; n < j->N; n++) {
        const float *b = j->B + (size_t)n * j->ldb;
        for (int m = m0; m < m1; m++) {
            const float *a = j->A + (size_t)m * a_rstride;
            float s = dot_scalar(j->K, a, a_kstride, b, 1);
            float *c = j->C + (size_t)m * j->ldc + n;
            *c = j->alpha * s + j->beta * (*c);
        }
    }
}

static void worker_direct(void *ctx, int worker, int nworkers) {
    GemmJob *j = (GemmJob *)ctx;
    int m_per = (j->M + nworkers - 1) / nworkers;
    m_per = ((m_per + 7) / 8) * 8;
    int start = worker * m_per;
    int end = imin(start + m_per, j->M);
    if (start < end) direct_kernel_transB(j, start, end);
}

/* ------------------------------------------------------------------------- */
/* M == 1 decode GEMV.                                                        */
/* ------------------------------------------------------------------------- */
static void gemv(UenrGemm *g, int N, int K,
                 const float *x, const float *B, int ldb, int transB,
                 float *y, float alpha, float beta) {
    if (g->avx2) {
        uenr_gemm_avx2_gemv(N, K, x, B, ldb, transB, y, alpha, beta);
        return;
    }
    if (!transB) {
        for (int n0 = 0; n0 < N; n0 += 8) {
            int nb = imin(8, N - n0);
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 bv;
                if (nb == 8) {
                    bv = _mm256_loadu_ps(B + (size_t)k * ldb + n0);
                } else {
                    float tmp[8] = {0};
                    for (int j = 0; j < nb; j++) tmp[j] = B[(size_t)k * ldb + n0 + j];
                    bv = _mm256_loadu_ps(tmp);
                }
                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_broadcast_ss(x + k), bv));
            }
            float out[8];
            _mm256_storeu_ps(out, acc);
            for (int j = 0; j < nb; j++)
                y[n0 + j] = alpha * out[j] + beta * y[n0 + j];
        }
    } else {
        for (int n = 0; n < N; n++) {
            const float *row = B + (size_t)n * ldb;
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
            int k = 0;
            for (; k + 32 <= K; k += 32) {
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(x + k), _mm256_loadu_ps(row + k)));
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(x + k + 8), _mm256_loadu_ps(row + k + 8)));
                acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_loadu_ps(x + k + 16), _mm256_loadu_ps(row + k + 16)));
                acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_loadu_ps(x + k + 24), _mm256_loadu_ps(row + k + 24)));
            }
            float s = uenr_v_hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
            for (; k < K; k++) s += x[k] * row[k];
            y[n] = alpha * s + beta * y[n];
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Public API                                                                */
/* ------------------------------------------------------------------------- */

UenrResult uenr_gemm_init(UenrGemm *g, UenrPool *pool) {
    if (!g) return UENR_ERR_INVALID;
    memset(g, 0, sizeof(*g));
    g->pool = pool;
    g->mem_pool = pool;
    g->nthreads = pool ? uenr_pool_threads(pool) : 1;
    g->avx2 = uenr_simd_level() >= UENR_SIMD_AVX2;
    g->avx512 = uenr_simd_level() == UENR_SIMD_AVX512;
    g->avx2_tile = 0;
    size_t floats = (size_t)UENR_GEMM_NB * 4096;
    g->scratch = (float *)aligned_alloc(64, (floats * sizeof(float) + 63) & ~(size_t)63);
    if (!g->scratch) return UENR_ERR_ALLOC;
    uenr_madvise_huge(g->scratch, floats * sizeof(float));
    g->scratch_floats = floats;
    return UENR_OK;
}

void uenr_gemm_set_avx2_tile(UenrGemm *g, int tile) {
    if (g) g->avx2_tile = (tile == 1 || tile == 2) ? tile : 0;
}

void uenr_gemm_destroy(UenrGemm *g) {
    if (!g) return;
    free(g->scratch);
    g->scratch = NULL;
    g->scratch_floats = 0;
}

void uenr_gemm(UenrGemm *g,
               int M, int N, int K,
               const float *A, int lda, int transA,
               const float *B, int ldb, int transB,
               float *C, int ldc,
               float alpha, float beta) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    if (M == 1) {
        gemv(g, N, K, A, B, ldb, transB, C, alpha, beta);
        return;
    }

    GemmJob job = { g, M, N, K, A, lda, transA, B, ldb, transB, C, ldc,
                    alpha, beta, NULL };

    long long work = (long long)M * N * K;
    int threaded = g->pool && g->nthreads > 1 && work >= (1LL << 20) && M >= 8 && N >= 8;

    if (!transB) {
        if (threaded) uenr_pool_run(g->pool, worker_range, &job);
        else range_kernel(&job, 0, M);
        return;
    }

    size_t need = (size_t)N * (size_t)K;
    if (!g->force_fallback && need > g->scratch_floats) {
        float *p = (float *)aligned_alloc(64, (need * sizeof(float) + 63) & ~(size_t)63);
        if (p) {
            free(g->scratch);
            g->scratch = p;
            g->scratch_floats = need;
        }
        /* On allocation failure fall through to the direct path below. */
    }
    if (g->force_fallback || need > g->scratch_floats) {
        if (threaded) uenr_pool_run(g->pool, worker_direct, &job);
        else direct_kernel_transB(&job, 0, M);
        return;
    }
    pack_b_all(B, ldb, N, K, g->scratch);
    job.all_panels = g->scratch;
    if (threaded) {
        uenr_pool_run(g->pool, worker_packed_all, &job);
    } else {
        for (int n0 = 0; n0 < N; n0 += UENR_GEMM_NB) {
            int nb = imin(UENR_GEMM_NB, N - n0);
            block_kernel(&job, n0, nb, g->scratch + (size_t)n0 * K, 0, M);
        }
    }
}
