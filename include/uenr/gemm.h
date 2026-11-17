#ifndef UENR_GEMM_H
#define UENR_GEMM_H

#include <stddef.h>
#include "uenr/result.h"
#include "uenr/thread.h"

/*
 * C[M,N] = alpha * op(A)[M,K] * op(B)[K,N] + beta * C[M,N]
 *
 * transA/transB are 0 (row-major [M,K] / [K,N]) or 1 ([K,M] / [N,K]).
 * A 4x8 AVX microkernel with four independent accumulator chains is used for
 * the body; transposed-B operands are packed into K-major panels once per
 * N-block. Work is split across the pool over rows of C.
 *
 * The trainer uses this for every projection:
 *   forward          Y = X W^T          (transB)
 *   input grad       dX = dY W          (plain)
 *   weight grad      dW = dY^T X        (transA)
 */

#define UENR_GEMM_NB 96

typedef struct {
    UenrPool *pool;      /* GEMM workers (physical cores) */
    UenrPool *mem_pool;  /* memory-bound phases (may use SMT siblings) */
    float    *scratch;
    size_t    scratch_floats;
    int       nthreads;
    /* Diagnostic/fault-injection switch: force the allocation-free transB
     * fallback path.  Also used by tests to compare it with the packed path. */
    int       force_fallback;
} UenrGemm;

UenrResult uenr_gemm_init(UenrGemm *g, UenrPool *pool);
void       uenr_gemm_destroy(UenrGemm *g);

void uenr_gemm(UenrGemm *g,
               int M, int N, int K,
               const float *A, int lda, int transA,
               const float *B, int ldb, int transB,
               float *C, int ldc,
               float alpha, float beta);

/* Convenience wrappers. */
static inline void uenr_linear_fwd(UenrGemm *g, float *Y, const float *X,
                                   const float *W, int T, int in, int out) {
    uenr_gemm(g, T, out, in, X, in, 0, W, in, 1, Y, out, 1.0f, 0.0f);
}
static inline void uenr_linear_dx(UenrGemm *g, float *dX, const float *dY,
                                  const float *W, int T, int in, int out) {
    uenr_gemm(g, T, in, out, dY, out, 0, W, in, 0, dX, in, 1.0f, 0.0f);
}
static inline void uenr_linear_dx_acc(UenrGemm *g, float *dX, const float *dY,
                                      const float *W, int T, int in, int out,
                                      float beta) {
    uenr_gemm(g, T, in, out, dY, out, 0, W, in, 0, dX, in, 1.0f, beta);
}
static inline void uenr_linear_dw(UenrGemm *g, float *dW, const float *dY,
                                  const float *X, int T, int in, int out,
                                  float beta) {
    uenr_gemm(g, out, in, T, dY, out, 1, X, in, 0, dW, in, 1.0f, beta);
}

#endif /* UENR_GEMM_H */
