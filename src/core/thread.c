#include "uenr/thread.h"

#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>

typedef struct WorkerArg WorkerArg;

struct UenrPool {
    int nthreads;
    int shutdown;
    unsigned generation;
    int pending;
    pthread_t *threads;
    WorkerArg *args;
    pthread_mutex_t mutex;
    pthread_cond_t work_cv;
    pthread_cond_t done_cv;
    UenrRangeFn fn;
    void *ctx;
};

struct WorkerArg {
    UenrPool *pool;
    int id;
    unsigned seen;
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

static void *worker_main(void *p) {
    WorkerArg *a = (WorkerArg *)p;
    UenrPool *pool = a->pool;
    if (a->ncpus > 0) pin_self(a->cpus[a->id % a->ncpus]);

    for (;;) {
        pthread_mutex_lock(&pool->mutex);
        while (!pool->shutdown && pool->generation == a->seen)
            pthread_cond_wait(&pool->work_cv, &pool->mutex);
        if (pool->shutdown) { pthread_mutex_unlock(&pool->mutex); break; }
        a->seen = pool->generation;
        UenrRangeFn fn = pool->fn;
        void *ctx = pool->ctx;
        pthread_mutex_unlock(&pool->mutex);

        fn(ctx, a->id, pool->nthreads);

        pthread_mutex_lock(&pool->mutex);
        if (--pool->pending == 0) pthread_cond_signal(&pool->done_cv);
        pthread_mutex_unlock(&pool->mutex);
    }
    return NULL;
}

UenrResult uenr_pool_init(UenrPool **out, int nthreads, const int *cpus, int ncpus) {
    if (!out || nthreads < 1) return UENR_ERR_INVALID;
    UenrPool *pool = (UenrPool *)calloc(1, sizeof(*pool));
    if (!pool) return UENR_ERR_ALLOC;
    pool->nthreads = nthreads;

    if (pthread_mutex_init(&pool->mutex, NULL) != 0 ||
        pthread_cond_init(&pool->work_cv, NULL) != 0 ||
        pthread_cond_init(&pool->done_cv, NULL) != 0) {
        free(pool);
        return UENR_ERR_THREAD;
    }

    if (nthreads > 1) {
        pool->threads = (pthread_t *)calloc((size_t)(nthreads - 1), sizeof(pthread_t));
        WorkerArg *args = (WorkerArg *)calloc((size_t)(nthreads - 1), sizeof(WorkerArg));
        pool->args = args;
        if (!pool->threads || !args) {
            free(pool->threads); free(args); pool->args = NULL;
            pthread_mutex_destroy(&pool->mutex);
            pthread_cond_destroy(&pool->work_cv);
            pthread_cond_destroy(&pool->done_cv);
            free(pool);
            return UENR_ERR_ALLOC;
        }
        int made = 0;
        for (int i = 1; i < nthreads; i++) {
            args[i - 1].pool = pool;
            args[i - 1].id = i;
            args[i - 1].seen = 0;
            args[i - 1].cpus = cpus;
            args[i - 1].ncpus = ncpus;
            if (pthread_create(&pool->threads[i - 1], NULL, worker_main, &args[i - 1]) != 0)
                break;
            made++;
        }
        if (made != nthreads - 1) {
            pthread_mutex_lock(&pool->mutex);
            pool->shutdown = 1;
            pthread_cond_broadcast(&pool->work_cv);
            pthread_mutex_unlock(&pool->mutex);
            for (int i = 0; i < made; i++) pthread_join(pool->threads[i], NULL);
            free(pool->threads);
            free(pool->args);
            pthread_mutex_destroy(&pool->mutex);
            pthread_cond_destroy(&pool->work_cv);
            pthread_cond_destroy(&pool->done_cv);
            free(pool);
            return UENR_ERR_THREAD;
        }
    }
    *out = pool;
    return UENR_OK;
}

void uenr_pool_run(UenrPool *pool, UenrRangeFn fn, void *ctx) {
    if (!pool || !fn) return;
    if (pool->nthreads == 1) {
        fn(ctx, 0, 1);
        return;
    }
    pthread_mutex_lock(&pool->mutex);
    pool->fn = fn;
    pool->ctx = ctx;
    pool->pending = pool->nthreads - 1;
    pool->generation++;
    pthread_cond_broadcast(&pool->work_cv);
    pthread_mutex_unlock(&pool->mutex);

    fn(ctx, 0, pool->nthreads);

    pthread_mutex_lock(&pool->mutex);
    while (pool->pending > 0) pthread_cond_wait(&pool->done_cv, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
}

void uenr_pool_destroy(UenrPool *pool) {
    if (!pool) return;
    if (pool->nthreads > 1) {
        pthread_mutex_lock(&pool->mutex);
        pool->shutdown = 1;
        pthread_cond_broadcast(&pool->work_cv);
        pthread_mutex_unlock(&pool->mutex);
        for (int i = 1; i < pool->nthreads; i++) pthread_join(pool->threads[i - 1], NULL);
        free(pool->threads);
        free(pool->args);
    }
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->work_cv);
    pthread_cond_destroy(&pool->done_cv);
    free(pool);
}

void uenr_pin_cpu(int cpu) { pin_self(cpu); }

int uenr_pool_threads(const UenrPool *pool) {
    return pool ? pool->nthreads : 1;
}
