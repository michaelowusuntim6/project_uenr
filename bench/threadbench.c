// threadbench.c - does this 2C/4T Ivy Bridge scale for compute vs memory?
// Build: gcc -O3 -march=native -pthread -o threadbench threadbench.c
// Run:   ./threadbench            (pins to physical cores 0 and 2)

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static void *xmalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n)) { perror("alloc"); exit(1); }
    memset(p, 0, n); return p;
}

// ---------- compute: dependency-free mul+add chains ----------
typedef struct { long iters; double gflops; } ComputeArg;
static void *compute_worker(void *v) {
    ComputeArg *a = v;
    __m256 x0=_mm256_set1_ps(1.0f), x1=_mm256_set1_ps(1.1f);
    __m256 x2=_mm256_set1_ps(1.2f), x3=_mm256_set1_ps(1.3f);
    __m256 x4=_mm256_set1_ps(1.4f), x5=_mm256_set1_ps(1.5f);
    __m256 x6=_mm256_set1_ps(1.6f), x7=_mm256_set1_ps(1.7f);
    __m256 b=_mm256_set1_ps(1.0000001f), c=_mm256_set1_ps(1e-7f);
    double t0 = now_sec();
    for (long i = 0; i < a->iters; i++) {
        x0=_mm256_add_ps(_mm256_mul_ps(x0,b),c);
        x1=_mm256_add_ps(_mm256_mul_ps(x1,b),c);
        x2=_mm256_add_ps(_mm256_mul_ps(x2,b),c);
        x3=_mm256_add_ps(_mm256_mul_ps(x3,b),c);
        x4=_mm256_add_ps(_mm256_mul_ps(x4,b),c);
        x5=_mm256_add_ps(_mm256_mul_ps(x5,b),c);
        x6=_mm256_add_ps(_mm256_mul_ps(x6,b),c);
        x7=_mm256_add_ps(_mm256_mul_ps(x7,b),c);
    }
    double t1 = now_sec();
    __m256 t=_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(x0,x1),_mm256_add_ps(x2,x3)),
                            _mm256_add_ps(_mm256_add_ps(x4,x5),_mm256_add_ps(x6,x7)));
    __m128 lo=_mm256_castps256_ps128(t), hi=_mm256_extractf128_ps(t,1);
    __m128 u=_mm_add_ps(lo,hi); u=_mm_hadd_ps(u,u); u=_mm_hadd_ps(u,u);
    volatile float sink=_mm_cvtss_f32(u); (void)sink;
    a->gflops = (double)a->iters * 8.0 * 8.0 * 2.0 / (t1 - t0) / 1e9;
    return NULL;
}

// ---------- triad over a range ----------
typedef struct { const float *a, *b; float *c; size_t lo, hi; double gbps; } TriadArg;
static void *triad_worker(void *v) {
    TriadArg *t = v;
    __m256 s = _mm256_set1_ps(1.0001f);
    double t0 = now_sec();
    for (size_t i = t->lo; i < t->hi; i += 8)
        _mm256_store_ps(t->c + i,
            _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(t->a + i), s),
                          _mm256_load_ps(t->b + i)));
    double t1 = now_sec();
    volatile float sink = t->c[t->lo]; (void)sink;
    t->gbps = (double)(t->hi - t->lo) * sizeof(float) * 3 / (t1 - t0) / 1e9;
    return NULL;
}

static void __attribute__((unused)) pin(int cpu) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static double run_compute(int nthreads) {
    static const int cpus[2] = {0, 2}; // physical cores
    pthread_t th[2]; ComputeArg args[2];
    for (int i = 0; i < nthreads; i++) {
        args[i].iters = 150000000L;
        pthread_create(&th[i], NULL, compute_worker, &args[i]);
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpus[i], &set);
        pthread_setaffinity_np(th[i], sizeof(set), &set);
    }
    double total = 0;
    for (int i = 0; i < nthreads; i++) { pthread_join(th[i], NULL); total += args[i].gflops; }
    return total;
}

static double run_triad(size_t bytes, int nthreads) {
    static const int cpus[2] = {0, 2};
    size_t n = bytes / sizeof(float);
    float *a = xmalloc(n*sizeof(float)), *b = xmalloc(n*sizeof(float)), *c = xmalloc(n*sizeof(float));
    for (size_t i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }
    pthread_t th[2]; TriadArg args[2];
    size_t chunk = (n / nthreads) & ~7ull;
    for (int i = 0; i < nthreads; i++) {
        args[i] = (TriadArg){a, b, c, i*chunk, (i == nthreads-1) ? n : (i+1)*chunk, 0};
        pthread_create(&th[i], NULL, triad_worker, &args[i]);
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpus[i], &set);
        pthread_setaffinity_np(th[i], sizeof(set), &set);
    }
    double total = 0;
    for (int i = 0; i < nthreads; i++) { pthread_join(th[i], NULL); total += args[i].gbps; }
    free(a); free(b); free(c);
    return total;
}

int main(void) {
    printf("=== threadbench: scaling on 2 physical cores (0,2) ===\n");
    double c1 = run_compute(1), c2 = run_compute(2);
    printf("[compute AVX] 1 thread %.2f GFLOP/s | 2 threads %.2f GFLOP/s (x%.2f)\n", c1, c2, c2/c1);
    printf("[triad]       1 thread %.1f GB/s   | 2 threads %.1f GB/s   (x%.2f)\n",
           run_triad(512u<<20, 1), run_triad(512u<<20, 2), run_triad(512u<<20,2)/run_triad(512u<<20,1));
    printf("[triad L2]    1 thread %.1f GB/s   | 2 threads %.1f GB/s\n",
           run_triad(128u<<10, 1), run_triad(128u<<10, 2));
    return 0;
}
