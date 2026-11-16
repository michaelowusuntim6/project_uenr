// cpubench.c - measure the real limits of THIS laptop CPU.
// Target: Intel i5-3210M (Ivy Bridge), 2C/4T, AVX1 + SSE4.2 + F16C, NO AVX2/FMA.
//
// Build:  gcc -O3 -march=native -funroll-loops -o cpubench cpubench.c -lm -pthread
// Run:    ./cpubench
//
// What it measures:
//   1. FP32 FMA-free throughput: scalar vs SSE (4-wide) vs AVX (8-wide, split uops)
//   2. Memory bandwidth: read / write / copy at streaming sizes
//   3. Cache-resident bandwidth: L1 / L2 / L3 working sets
//   4. A small GEMM roofline probe: compute vs memory bound crossover

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *xmalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n) != 0) { fprintf(stderr, "alloc fail\n"); exit(1); }
    memset(p, 0, n);
    return p;
}

// ---- 1. FP32 compute throughput -------------------------------------------------
// No FMA on this CPU: every mul and add is a separate uop. We count mul+add.
// Ivy Bridge FP add/mul: 1 per cycle per port (ports 0 and 1), 128-bit.
// 256-bit AVX is split into 2x128 uops, so 8-wide AVX != 2x throughput.

static double bench_scalar(long iters) {
    float a0=1.0f,a1=1.1f,a2=1.2f,a3=1.3f,a4=1.4f,a5=1.5f,a6=1.6f,a7=1.7f;
    float b = 1.0000001f, c = 0.0000001f;
    double t0 = now_sec();
    for (long i = 0; i < iters; i++) {
        a0 = a0*b + c; a1 = a1*b + c; a2 = a2*b + c; a3 = a3*b + c;
        a4 = a4*b + c; a5 = a5*b + c; a6 = a6*b + c; a7 = a7*b + c;
    }
    double t1 = now_sec();
    volatile float sink = a0+a1+a2+a3+a4+a5+a6+a7;
    (void)sink;
    // 8 lanes * 2 flops (mul+add) per iter
    return (double)iters * 8.0 * 2.0 / (t1 - t0) / 1e9; // GFLOP/s
}

static double bench_sse(long iters) {
    __m128 a0=_mm_set1_ps(1.0f), a1=_mm_set1_ps(1.1f);
    __m128 a2=_mm_set1_ps(1.2f), a3=_mm_set1_ps(1.3f);
    __m128 a4=_mm_set1_ps(1.4f), a5=_mm_set1_ps(1.5f);
    __m128 a6=_mm_set1_ps(1.6f), a7=_mm_set1_ps(1.7f);
    __m128 b=_mm_set1_ps(1.0000001f), c=_mm_set1_ps(0.0000001f);
    double t0 = now_sec();
    for (long i = 0; i < iters; i++) {
        a0=_mm_add_ps(_mm_mul_ps(a0,b),c); a1=_mm_add_ps(_mm_mul_ps(a1,b),c);
        a2=_mm_add_ps(_mm_mul_ps(a2,b),c); a3=_mm_add_ps(_mm_mul_ps(a3,b),c);
        a4=_mm_add_ps(_mm_mul_ps(a4,b),c); a5=_mm_add_ps(_mm_mul_ps(a5,b),c);
        a6=_mm_add_ps(_mm_mul_ps(a6,b),c); a7=_mm_add_ps(_mm_mul_ps(a7,b),c);
    }
    double t1 = now_sec();
    __m128 s=_mm_add_ps(_mm_add_ps(_mm_add_ps(a0,a1),_mm_add_ps(a2,a3)),
                        _mm_add_ps(_mm_add_ps(a4,a5),_mm_add_ps(a6,a7)));
    volatile float sink=_mm_cvtss_f32(s); (void)sink;
    // 8 vectors * 4 lanes * 2 flops
    return (double)iters * 8.0 * 4.0 * 2.0 / (t1 - t0) / 1e9;
}

static double bench_avx(long iters) {
    __m256 a0=_mm256_set1_ps(1.0f), a1=_mm256_set1_ps(1.1f);
    __m256 a2=_mm256_set1_ps(1.2f), a3=_mm256_set1_ps(1.3f);
    __m256 a4=_mm256_set1_ps(1.4f), a5=_mm256_set1_ps(1.5f);
    __m256 a6=_mm256_set1_ps(1.6f), a7=_mm256_set1_ps(1.7f);
    __m256 b=_mm256_set1_ps(1.0000001f), c=_mm256_set1_ps(0.0000001f);
    double t0 = now_sec();
    for (long i = 0; i < iters; i++) {
        a0=_mm256_add_ps(_mm256_mul_ps(a0,b),c); a1=_mm256_add_ps(_mm256_mul_ps(a1,b),c);
        a2=_mm256_add_ps(_mm256_mul_ps(a2,b),c); a3=_mm256_add_ps(_mm256_mul_ps(a3,b),c);
        a4=_mm256_add_ps(_mm256_mul_ps(a4,b),c); a5=_mm256_add_ps(_mm256_mul_ps(a5,b),c);
        a6=_mm256_add_ps(_mm256_mul_ps(a6,b),c); a7=_mm256_add_ps(_mm256_mul_ps(a7,b),c);
    }
    double t1 = now_sec();
    __m256 s=_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3)),
                           _mm256_add_ps(_mm256_add_ps(a4,a5),_mm256_add_ps(a6,a7)));
    __m128 lo=_mm256_castps256_ps128(s), hi=_mm256_extractf128_ps(s,1);
    __m128 t=_mm_add_ps(lo,hi);
    volatile float sink=_mm_cvtss_f32(t); (void)sink;
    return (double)iters * 8.0 * 8.0 * 2.0 / (t1 - t0) / 1e9;
}

// ---- 2/3. Bandwidth ------------------------------------------------------------
// Triad-like: c[i] = a[i]*s + b[i], plus pure read and pure write.

static double bw_read(const float *a, size_t n, int reps) {
    // 4 AVX accumulators to hide add latency and saturate load ports.
    __m256 s0=_mm256_setzero_ps(), s1=s0, s2=s0, s3=s0;
    size_t nv = n & ~7ull;
    double t0 = now_sec();
    for (int r = 0; r < reps; r++) {
        for (size_t i = 0; i < nv; i += 8) {
            s0 = _mm256_add_ps(s0, _mm256_load_ps(a + i));
            s1 = _mm256_add_ps(s1, _mm256_load_ps(a + i + 8));
            s2 = _mm256_add_ps(s2, _mm256_load_ps(a + i + 16));
            s3 = _mm256_add_ps(s3, _mm256_load_ps(a + i + 24));
        }
        for (size_t i = nv; i < n; i++) s0 = _mm256_add_ps(s0, _mm256_set1_ps(a[i]));
    }
    double t1 = now_sec();
    __m256 t = _mm256_add_ps(_mm256_add_ps(s0,s1), _mm256_add_ps(s2,s3));
    __m128 lo=_mm256_castps256_ps128(t), hi=_mm256_extractf128_ps(t,1);
    __m128 u=_mm_add_ps(lo,hi); u=_mm_hadd_ps(u,u); u=_mm_hadd_ps(u,u);
    volatile float sink=_mm_cvtss_f32(u); (void)sink;
    return (double)n * sizeof(float) * reps / (t1 - t0) / 1e9; // GB/s
}

static double bw_write(float *c, size_t n, int reps) {
    double t0 = now_sec();
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i < n; i++) c[i] = 1.0f;
    double t1 = now_sec();
    return (double)n * sizeof(float) * reps / (t1 - t0) / 1e9;
}

static double bw_copy(const float *a, float *c, size_t n, int reps) {
    double t0 = now_sec();
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i < n; i += 8)
            _mm256_store_ps(c + i, _mm256_load_ps(a + i));
    double t1 = now_sec();
    // force the stores to be observable so nothing is eliminated
    volatile float sink = c[n/2];
    (void)sink;
    return (double)n * sizeof(float) * 2 * reps / (t1 - t0) / 1e9; // r+w
}

static double bw_triad(const float *a, const float *b, float *c, size_t n, int reps) {
    const float s = 1.0001f;
    __m256 vs = _mm256_set1_ps(s);
    double t0 = now_sec();
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i < n; i += 8)
            _mm256_store_ps(c + i,
                _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(a + i), vs),
                              _mm256_load_ps(b + i)));
    double t1 = now_sec();
    volatile float sink = c[n/2];
    (void)sink;
    return (double)n * sizeof(float) * 3 * reps / (t1 - t0) / 1e9; // r+r+w
}

// ---- 4. tiny GEMM roofline probe ----------------------------------------------
// C[MxN] += A[MxK] * B[KxN] with a simple register-tiled kernel, no FMA.
static void gemm_nt(float *C, const float *A, const float *B, int M, int N, int K) {
    const int MR = 4, NR = 8;
    for (int i0 = 0; i0 < M; i0 += MR) {
        for (int j0 = 0; j0 < N; j0 += NR) {
            float acc[MR][NR];
            for (int i = 0; i < MR; i++)
                for (int j = 0; j < NR; j++) acc[i][j] = 0.0f;
            for (int k = 0; k < K; k++) {
                float bv[NR];
                for (int j = 0; j < NR; j++) bv[j] = B[k*N + j0 + j];
                for (int i = 0; i < MR; i++) {
                    float av = A[(i0+i)*K + k];
                    for (int j = 0; j < NR; j++) acc[i][j] += av * bv[j];
                }
            }
            for (int i = 0; i < MR; i++)
                for (int j = 0; j < NR; j++) C[(i0+i)*N + j0 + j] += acc[i][j];
        }
    }
}

static double bench_gemm(int M, int N, int K, int reps) {
    float *A = xmalloc((size_t)M*K*sizeof(float));
    float *B = xmalloc((size_t)K*N*sizeof(float));
    float *C = xmalloc((size_t)M*N*sizeof(float));
    for (int i = 0; i < M*K; i++) A[i] = 1.0f + (i % 7) * 0.01f;
    for (int i = 0; i < K*N; i++) B[i] = 1.0f + (i % 5) * 0.01f;
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        gemm_nt(C, A, B, M, N, K);
        double t1 = now_sec();
        if (t1 - t0 < best) best = t1 - t0;
    }
    double gflops = 2.0 * M * N * K / best / 1e9;
    free(A); free(B); free(C);
    return gflops;
}

int main(void) {
    printf("=== cpubench: i5-3210M (Ivy Bridge) ===\n");
    printf("FP32 peak theory: 2 cores * 2 (ports0/1) * 4 lanes * 2 (mul+add) * ~2.9GHz = ~92 GFLOP/s\n");
    printf("            (AVX 256-bit is split to 2x128 uops here: no extra FP throughput)\n\n");

    const long it = 200000000L;
    double s = bench_scalar(it);
    double v = bench_sse(it);
    double a = bench_avx(it);
    printf("[compute] scalar  : %7.2f GFLOP/s\n", s);
    printf("[compute] SSE 128 : %7.2f GFLOP/s  (x%.2f scalar)\n", v, v/s);
    printf("[compute] AVX 256 : %7.2f GFLOP/s  (x%.2f scalar)\n", a, a/s);

    // bandwidth across working-set sizes to expose cache hierarchy
    struct { const char *name; size_t bytes; } sets[] = {
        {"L1  16 KiB", 16u<<10},
        {"L2 128 KiB", 128u<<10},
        {"L3   1 MiB", 1u<<20},
        {"L3   2 MiB", 2u<<20},
        {"DRAM 64 MiB", 64u<<20},
        {"DRAM 512MiB", 512u<<20},
    };
    printf("\n[memory] read / write / copy / triad bandwidth\n");
    for (size_t i = 0; i < sizeof(sets)/sizeof(sets[0]); i++) {
        size_t n = sets[i].bytes / sizeof(float);
        float *A = xmalloc(n*sizeof(float));
        float *B = xmalloc(n*sizeof(float));
        float *C = xmalloc(n*sizeof(float));
        int reps = (sets[i].bytes <= (1u<<20)) ? 200 : ((sets[i].bytes <= (64u<<20)) ? 20 : 4);
        double r = bw_read(A, n, reps);
        double w = bw_write(C, n, reps);
        double cp = bw_copy(A, C, n, reps);
        double tr = bw_triad(A, B, C, n, reps);
        printf("  %-11s r=%6.1f w=%6.1f copy=%6.1f triad=%6.1f GB/s\n",
               sets[i].name, r, w, cp, tr);
        free(A); free(B); free(C);
    }

    printf("\n[tiny GEMM] naive register-tiled fp32 (no FMA):\n");
    printf("  64x64x64     : %6.2f GFLOP/s\n", bench_gemm(64,64,64,50));
    printf("  256x256x256  : %6.2f GFLOP/s\n", bench_gemm(256,256,256,10));
    printf("  768x768x768  : %6.2f GFLOP/s\n", bench_gemm(768,768,768,3));
    return 0;
}
