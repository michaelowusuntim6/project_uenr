#ifndef UENR_THREAD_H
#define UENR_THREAD_H

#include "uenr/result.h"

typedef struct UenrPool UenrPool;
typedef void (*UenrRangeFn)(void *ctx, int worker, int nworkers);
typedef void (*UenrRangeFn2)(void *ctx, int start, int end, int worker);

/* nthreads includes the caller. Workers are pinned to the given CPU ids. */
UenrResult uenr_pool_init(UenrPool **out, int nthreads, const int *cpus, int ncpus);
void       uenr_pool_run(UenrPool *pool, UenrRangeFn fn, void *ctx);
void       uenr_pool_destroy(UenrPool *pool);
int        uenr_pool_threads(const UenrPool *pool);
void       uenr_pin_cpu(int cpu);
/* Split [0,n) into contiguous chunks across the pool. */
void       uenr_pool_range(UenrPool *pool, UenrRangeFn2 fn, void *ctx, int n);

#endif /* UENR_THREAD_H */
