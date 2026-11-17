#include "uenr/gemm.h"
#include "uenr/rng.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void naive(int M, int N, int K,
                  const float *A, int lda, int ta,
                  const float *B, int ldb, int tb,
                  float *C, int ldc, float alpha, float beta) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) {
                float a = ta ? A[(size_t)k * lda + m] : A[(size_t)m * lda + k];
                float b = tb ? B[(size_t)n * ldb + k] : B[(size_t)k * ldb + n];
                s += a * b;
            }
            C[(size_t)m * ldc + n] = alpha * s + beta * C[(size_t)m * ldc + n];
        }
    }
}

static int run_case_ab(int M, int N, int K, int ta, int tb, int force,
                       float alpha, float beta, UenrPool *pool);

static int run_case(int M, int N, int K, int ta, int tb, UenrPool *pool) {
    return run_case_ab(M, N, K, ta, tb, 0, 1.0f, 0.0f, pool);
}

static int run_case_ab(int M, int N, int K, int ta, int tb, int force,
                       float alpha, float beta, UenrPool *pool) {
    int lda = ta ? M : K;
    int ldb = tb ? K : N;

    float *A = aligned_alloc(64, (((size_t)M * K * sizeof(float)) + 63) & ~(size_t)63);
    float *B = aligned_alloc(64, (((size_t)N * K * sizeof(float)) + 63) & ~(size_t)63);
    float *C = aligned_alloc(64, (((size_t)M * N * sizeof(float)) + 63) & ~(size_t)63);
    float *R = aligned_alloc(64, (((size_t)M * N * sizeof(float)) + 63) & ~(size_t)63);
    if (!A || !B || !C || !R) return 1;

    UenrRng rng;
    uenr_rng_seed(&rng, 42 + (unsigned)(M * 131 + N * 17 + K + ta * 7 + tb * 3));
    for (int i = 0; i < M * K; i++) A[i] = uenr_rng_normal(&rng);
    for (int i = 0; i < N * K; i++) B[i] = uenr_rng_normal(&rng);
    for (int i = 0; i < M * N; i++) { C[i] = 0.3f; R[i] = 0.3f; }

    UenrGemm g;
    if (!uenr_ok(uenr_gemm_init(&g, pool))) return 1;
    g.force_fallback = force;
    uenr_gemm(&g, M, N, K, A, lda, ta, B, ldb, tb, C, N, alpha, beta);
    naive(M, N, K, A, lda, ta, B, ldb, tb, R, N, alpha, beta);

    float maxerr = 0.0f, maxval = 1.0f;
    for (int i = 0; i < M * N; i++) {
        float e = fabsf(C[i] - R[i]);
        if (e > maxerr) maxerr = e;
        if (fabsf(R[i]) > maxval) maxval = fabsf(R[i]);
    }
    uenr_gemm_destroy(&g);
    free(A); free(B); free(C); free(R);

    if (maxerr / maxval > 2e-4f) {
        printf("FAIL gemm M=%d N=%d K=%d ta=%d tb=%d force=%d err=%g\n", M, N, K, ta, tb, force, maxerr);
        return 1;
    }
    printf("ok   gemm M=%d N=%d K=%d ta=%d tb=%d force=%d err=%.2e\n", M, N, K, ta, tb, force, maxerr / maxval);
    return 0;
}

int main(void) {
    UenrPool *pool = NULL;
    int cpus[2] = {0, 2};
    if (!uenr_ok(uenr_pool_init(&pool, 2, cpus, 2))) {
        printf("pool init failed, using 1 thread\n");
        pool = NULL;
    }
    int fails = 0;
    fails += run_case(4, 8, 16, 0, 0, pool);
    fails += run_case(7, 13, 21, 0, 0, pool);
    fails += run_case(16, 32, 41, 0, 0, pool);
    fails += run_case(64, 96, 192, 0, 0, pool);
    fails += run_case(33, 71, 55, 0, 1, pool);
    fails += run_case(64, 192, 96, 0, 1, pool);
    fails += run_case(5, 17, 23, 1, 0, pool);
    fails += run_case(192, 128, 257, 1, 0, pool);
    fails += run_case(1, 64, 192, 0, 1, pool);
    fails += run_case(1, 64, 192, 0, 0, pool);
    fails += run_case_ab(33, 71, 55, 0, 1, 1, 1.0f, 0.0f, pool);
    fails += run_case_ab(64, 96, 192, 0, 1, 1, 0.5f, 0.25f, pool);
    fails += run_case_ab(5, 17, 23, 1, 0, 1, 1.0f, 0.0f, pool);
    if (pool) uenr_pool_destroy(pool);
    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
