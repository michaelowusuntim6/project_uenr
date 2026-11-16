#include "uenr/gemm.h"
#include "uenr/simd.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    UenrGemm *g;
    int M, N, K;
    const float *A; int lda, transA;
    const float *B; int ldb, transB;
    float *C; int ldc;
    float alpha, beta;
} GemmJob;

static inline int imin(int a, int b) { return a < b ? a : b; }

/* ------------------------------------------------------------------------- */
/* 4x8 microkernel body.  transB==1 reads a packed panel [K][pack_nb].       */
/* ------------------------------------------------------------------------- */

static inline void micro_4x8(int K,
                             const float *a0, const float *a1,
                             const float *a2, const float *a3,
                             int a_kstride,
                             const float *bp, int b_kstride,
                             float *c0, float *c1, float *c2, float *c3, int ldc,
                             float alpha, float beta) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    const float *p0 = a0, *p1 = a1, *p2 = a2, *p3 = a3, *pb = bp;
    for (int k = 0; k < K; k++) {
        __m256 b = _mm256_loadu_ps(pb);
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

/* Scalar fallback element (also covers non-multiple-of-8 column remainders). */
static inline float dot_scalar(int K,
                               const float *a, int a_kstride,
                               const float *b, int b_kstride) {
    float s = 0.0f;
    for (int k = 0; k < K; k++) s += a[k * a_kstride] * b[k * b_kstride];
    return s;
}

/* ------------------------------------------------------------------------- */
/* Pack B[n0..n0+nb)[K] into panel[k*nb + j].                                */
/* ------------------------------------------------------------------------- */
static void pack_b(const float *B, int ldb, int n0, int nb, int K, float *panel) {
    for (int k = 0; k < K; k++) {
        float *dst = panel + (size_t)k * nb;
        for (int j = 0; j < nb; j++)
            dst[j] = B[(size_t)(n0 + j) * ldb + k];
    }
}

/* ------------------------------------------------------------------------- */
/* Core range worker                                                         */
/* ------------------------------------------------------------------------- */

static void gemm_range(const GemmJob *j, int m_start, int m_end, float *scratch) {
    const int K = j->K;
    const int a_kstride = j->transA ? j->lda : 1;
    const int a_rstride = j->transA ? 1 : j->lda;

    for (int n0 = 0; n0 < j->N; n0 += UENR_GEMM_NB) {
        const int nb = imin(UENR_GEMM_NB, j->N - n0);
        const float *panel = NULL;
        int b_kstride;
        if (j->transB) {
            panel = scratch;
            pack_b(j->B, j->ldb, n0, nb, K, scratch);
            b_kstride = nb;
        } else {
            b_kstride = j->ldb;
        }

        for (int m = m_start; m < m_end; m += 4) {
            const int mr = imin(4, m_end - m);
            const float *a0 = j->A + (size_t)m * a_rstride;
            const float *a1 = a0 + a_rstride;
            const float *a2 = a1 + a_rstride;
            const float *a3 = a2 + a_rstride;

            int jcol = 0;
            for (; jcol + 8 <= nb; jcol += 8) {
                float *c0 = j->C + (size_t)m * j->ldc + n0 + jcol;
                const float *bp = panel ? (panel + jcol) : (j->B + (size_t)n0 + jcol);
                const int bk = panel ? b_kstride : b_kstride;
                if (mr == 4) {
                    micro_4x8(K, a0, a1, a2, a3, a_kstride, bp, bk,
                              c0, c0 + j->ldc, c0 + 2 * j->ldc, c0 + 3 * j->ldc,
                              j->ldc, j->alpha, j->beta);
                } else {
                    for (int i = 0; i < mr; i++) {
                        const float *ai = j->A + (size_t)(m + i) * a_rstride;
                        float *ci = j->C + (size_t)(m + i) * j->ldc + n0 + jcol;
                        if (j->transB) {
                            micro_1x8(K, ai, a_kstride, panel + jcol, nb, ci, j->alpha, j->beta);
                        } else {
                            micro_1x8(K, ai, a_kstride, j->B + (size_t)(n0 + jcol), j->ldb,
                                      ci, j->alpha, j->beta);
                        }
                    }
                }
            }

            /* column remainder: scalar dot products */
            for (int jj = jcol; jj < nb; jj++) {
                int n = n0 + jj;
                for (int i = 0; i < mr; i++) {
                    const float *ai = j->A + (size_t)(m + i) * a_rstride;
                    float v;
                    if (j->transB) {
                        v = dot_scalar(K, ai, a_kstride, j->B + (size_t)n * j->ldb, 1);
                    } else {
                        v = dot_scalar(K, ai, a_kstride, j->B + (size_t)n, j->ldb);
                    }
                    float *ci = j->C + (size_t)(m + i) * j->ldc + n;
                    *ci = j->alpha * v + j->beta * (*ci);
                }
            }
        }
    }
}

static void gemm_worker(void *ctx, int worker, int nworkers) {
    GemmJob *j = (GemmJob *)ctx;
    UenrGemm *g = j->g;
    int m_per = (j->M + nworkers - 1) / nworkers;
    m_per = ((m_per + 3) / 4) * 4;
    int start = worker * m_per;
    int end = imin(start + m_per, j->M);
    if (start >= end) return;
    float *scratch = g->scratch + (size_t)worker * UENR_GEMM_NB * (size_t)j->K;
    gemm_range(j, start, end, scratch);
}

/* ------------------------------------------------------------------------- */
/* M == 1 (decode): memory-bound GEMV, vectorized over the contiguous axis.  */
/* ------------------------------------------------------------------------- */
static void gemv(UenrGemm *g, int N, int K,
                 const float *x, const float *B, int ldb, int transB,
                 float *y, float alpha, float beta) {
    (void)g;
    if (!transB) {
        /* y[n] = sum_k x[k]*B[k,n]; vectorize over n. */
        for (int n0 = 0; n0 < N; n0 += 8) {
            int nb = imin(8, N - n0);
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 bv = (nb == 8) ? _mm256_loadu_ps(B + (size_t)k * ldb + n0)
                                      : _mm256_setzero_ps();
                if (nb != 8) {
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
        /* B is [N,K]; each row is a dot product with x. */
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
            __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            float s = uenr_v_hsum(acc);
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
    g->nthreads = pool ? uenr_pool_threads(pool) : 1;
    size_t floats = (size_t)g->nthreads * UENR_GEMM_NB * 4096;
    g->scratch = (float *)aligned_alloc(64, (floats * sizeof(float) + 63) & ~(size_t)63);
    if (!g->scratch) return UENR_ERR_ALLOC;
    g->scratch_floats = floats;
    return UENR_OK;
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

    /* Ensure per-worker pack space (K * NB floats). */
    size_t need = (size_t)g->nthreads * UENR_GEMM_NB * (size_t)K;
    if (need > g->scratch_floats) {
        float *p = (float *)aligned_alloc(64, (need * sizeof(float) + 63) & ~(size_t)63);
        if (!p) return; /* caller has no error channel; keep old scratch */
        free(g->scratch);
        g->scratch = p;
        g->scratch_floats = need;
    }

    GemmJob job = { g, M, N, K, A, lda, transA, B, ldb, transB, C, ldc, alpha, beta };

    long long work = (long long)M * N * K;
    if (!g->pool || g->nthreads <= 1 || work < (1LL << 20) || M < 8 || N < 8) {
        gemm_range(&job, 0, M, g->scratch);
    } else {
        uenr_pool_run(g->pool, gemm_worker, &job);
    }
}
