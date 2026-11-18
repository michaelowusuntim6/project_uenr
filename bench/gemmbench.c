#include "uenr/gemm.h"
#include "uenr/rng.h"
#include "uenr/thread.h"
#include "uenr/simd_dispatch.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float *alloc_mat(int m, int n) {
    size_t bytes = ((size_t)m * n * sizeof(float) + 63) & ~(size_t)63;
    return (float *)aligned_alloc(64, bytes);
}

static void bench(int n, int ta, int tb, UenrGemm *g) {
    float *A = alloc_mat(n, n), *B = alloc_mat(n, n), *C = alloc_mat(n, n);
    if (!A || !B || !C) { free(A); free(B); free(C); return; }
    UenrRng rng;
    uenr_rng_seed(&rng, 1234);
    for (int i = 0; i < n * n; i++) { A[i] = uenr_rng_normal(&rng); B[i] = uenr_rng_normal(&rng); }

    uenr_gemm(g, n, n, n, A, n, ta, B, n, tb, C, n, 1.0f, 0.0f);
    int iters = 0;
    double t0 = now_s(), dt = 0.0;
    do {
        uenr_gemm(g, n, n, n, A, n, ta, B, n, tb, C, n, 1.0f, 0.0f);
        iters++;
        dt = now_s() - t0;
    } while (dt < 0.4);
    double flops = 2.0 * (double)n * n * n * iters / dt;
    printf("  %s %4d^3: %7.2f GFLOP/s\n",
           ta ? "TN" : (tb ? "NT" : "NN"), n, flops / 1e9);
    free(A); free(B); free(C);
}

int main(int argc, char **argv) {
    const char *simd = argc > 2 ? argv[2] : "auto";
    if (uenr_simd_configure(simd) != UENR_OK) return 2;
    int sizes[4] = { 64, 128, 256, 384 };
    UenrPool *pool = NULL;
    int cpus[2] = { 0, 2 };
    uenr_pool_init(&pool, 2, cpus, 2);
    UenrGemm g;
    if (uenr_gemm_init(&g, pool) != UENR_OK) return 1;
    printf("UENR gemmbench — simd=%s; measured laptop 2-core roof 67.8 GFLOP/s (33.9/core)\n",
           uenr_simd_name());
    for (int i = 0; i < 4; i++) {
        int n = (argc > 1) ? atoi(argv[1]) : sizes[i];
        bench(n, 0, 0, &g);
        bench(n, 0, 1, &g);
        bench(n, 1, 0, &g);
        if (argc > 1) break;
    }
    uenr_gemm_destroy(&g);
    uenr_pool_destroy(pool);
    return 0;
}
