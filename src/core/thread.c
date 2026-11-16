#include "uenr/thread.h"

#include <limits.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

struct UenrPool {
    int nthreads;
    _Atomic int shutdown;
    _Atomic int generation;
    _Atomic int done;
    pthread_t *threads;
    struct WorkerArg *args;
    UenrRangeFn fn;
    void *ctx;
};

struct WorkerArg {
    UenrPool *pool;
    int id;
    const int *cpus;
    int ncpus;
};

static void pin_self(int cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)cpu;
#endif
}

static inline void cpu_relax(void) {
#if defined(__i386__) || defined(__x86_64__)
    __builtin_ia32_pause();
#else
    sched_yield();
#endif
}

static int futex_wait_int(_Atomic int *addr, int val) {
    /* Bounded wait: a lost wakeup must never deadlock the pool. */
    struct timespec ts = { 0, 1000000 }; /* 1 ms */
    return (int)syscall(SYS_futex, (int *)addr, FUTEX_WAIT_PRIVATE, val, &ts, NULL, 0);
}
static int futex_wake_int(_Atomic int *addr) {
    return (int)syscall(SYS_futex, (int *)addr, FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0);
}

static void *worker_main(void *p) {
    struct WorkerArg *a = (struct WorkerArg *)p;
    UenrPool *pool = a->pool;
    if (a->ncpus > 0) pin_self(a->cpus[a->id % a->ncpus]);
    int seen = 0;

    for (;;) {
        int spin = 0;
        while (atomic_load_explicit(&pool->generation, memory_order_acquire) == seen &&
               !atomic_load_explicit(&pool->shutdown, memory_order_relaxed)) {
            if (++spin < 2048) cpu_relax();
            else futex_wait_int(&pool->generation, seen);
        }
        if (atomic_load_explicit(&pool->shutdown, memory_order_relaxed)) break;
        seen = atomic_load_explicit(&pool->generation, memory_order_acquire);
        UenrRangeFn fn = pool->fn;
        void *ctx = pool->ctx;
        fn(ctx, a->id, pool->nthreads);
        atomic_fetch_add_explicit(&pool->done, 1, memory_order_release);
        futex_wake_int(&pool->done);
    }
    return NULL;
}

UenrResult uenr_pool_init(UenrPool **out, int nthreads, const int *cpus, int ncpus) {
    if (!out || nthreads < 1) return UENR_ERR_INVALID;
    UenrPool *pool = (UenrPool *)calloc(1, sizeof(*pool));
    if (!pool) return UENR_ERR_ALLOC;
    pool->nthreads = nthreads;
    atomic_init(&pool->shutdown, 0);
    atomic_init(&pool->generation, 0);
    atomic_init(&pool->done, 0);

    if (nthreads > 1) {
        pool->threads = (pthread_t *)calloc((size_t)(nthreads - 1), sizeof(pthread_t));
        pool->args = (struct WorkerArg *)calloc((size_t)(nthreads - 1), sizeof(struct WorkerArg));
        if (!pool->threads || !pool->args) {
            free(pool->threads); free(pool->args); free(pool);
            return UENR_ERR_ALLOC;
        }
        int made = 0;
        for (int i = 1; i < nthreads; i++) {
            pool->args[i - 1].pool = pool;
            pool->args[i - 1].id = i;
            pool->args[i - 1].cpus = cpus;
            pool->args[i - 1].ncpus = ncpus;
            if (pthread_create(&pool->threads[i - 1], NULL, worker_main, &pool->args[i - 1]) != 0)
                break;
            made++;
        }
        if (made != nthreads - 1) {
            atomic_store(&pool->shutdown, 1);
            atomic_fetch_add(&pool->generation, 1);
            futex_wake_int(&pool->generation);
            for (int i = 0; i < made; i++) pthread_join(pool->threads[i], NULL);
            free(pool->threads); free(pool->args); free(pool);
            return UENR_ERR_THREAD;
        }
    }
    *out = pool;
    return UENR_OK;
}

void uenr_pool_run(UenrPool *pool, UenrRangeFn fn, void *ctx) {
    if (!pool || !fn) return;
    if (pool->nthreads == 1) { fn(ctx, 0, 1); return; }
    pool->fn = fn;
    pool->ctx = ctx;
    atomic_store_explicit(&pool->done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&pool->generation, 1, memory_order_release);
    futex_wake_int(&pool->generation);

    fn(ctx, 0, pool->nthreads);

    int spin = 0;
    while (atomic_load_explicit(&pool->done, memory_order_acquire) < pool->nthreads - 1) {
        if (++spin < 4096) cpu_relax();
        else futex_wait_int(&pool->done, atomic_load_explicit(&pool->done, memory_order_relaxed));
    }
}

void uenr_pool_destroy(UenrPool *pool) {
    if (!pool) return;
    if (pool->nthreads > 1) {
        atomic_store_explicit(&pool->shutdown, 1, memory_order_release);
        atomic_fetch_add_explicit(&pool->generation, 1, memory_order_release);
        futex_wake_int(&pool->generation);
        for (int i = 1; i < pool->nthreads; i++) pthread_join(pool->threads[i - 1], NULL);
        free(pool->threads);
        free(pool->args);
    }
    free(pool);
}

void uenr_pin_cpu(int cpu) { pin_self(cpu); }

typedef struct {
    UenrRangeFn2 fn;
    void *ctx;
    int n;
} RangeWrap;

static void range_trampoline(void *p, int worker, int nworkers) {
    RangeWrap *w = (RangeWrap *)p;
    int per = (w->n + nworkers - 1) / nworkers;
    int start = worker * per;
    int end = start + per;
    if (end > w->n) end = w->n;
    if (start < end) w->fn(w->ctx, start, end, worker);
}

void uenr_pool_range(UenrPool *pool, UenrRangeFn2 fn, void *ctx, int n) {
    if (!fn || n <= 0) return;
    int nt = pool ? uenr_pool_threads(pool) : 1;
    if (!pool || nt <= 1 || n < 2) {
        fn(ctx, 0, n, 0);
        return;
    }
    RangeWrap w = { fn, ctx, n };
    uenr_pool_run(pool, range_trampoline, &w);
}

int uenr_pool_threads(const UenrPool *pool) {
    return pool ? pool->nthreads : 1;
}
