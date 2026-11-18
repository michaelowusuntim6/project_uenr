/* Thread-pool worker resolution: when more threads are requested than CPU ids
 * are supplied, all workers must still run (auto-pinned to unused CPUs). */
#include "uenr/thread.h"
#include "uenr/uenr.h"

#include <stdio.h>
#include <string.h>

static int fails;

static int check(int cond, const char *what) {
    if (!cond) printf("FAIL %s\n", what);
    else printf("ok   %s\n", what);
    if (!cond) fails++;
    return cond;
}

static void worker_hit(void *ctx, int worker, int nworkers) {
    int *seen = (int *)ctx;
    if (worker >= 0 && worker < 8) seen[worker] = 1;
    (void)nworkers;
}

static int run_pool(int nthreads, const int *cpus, int ncpus) {
    UenrPool *pool = NULL;
    if (uenr_pool_init(&pool, nthreads, cpus, ncpus) != UENR_OK) return 0;
    int seen[8] = {0};
    uenr_pool_run(pool, worker_hit, seen);
    int count = 0;
    for (int i = 0; i < 8; i++) count += seen[i];
    int reported = uenr_pool_threads(pool);
    uenr_pool_destroy(pool);
    return count == nthreads && reported == nthreads;
}

int main(void) {
    int two[2] = { 0, 1 };
    check(run_pool(1, two, 1), "single-thread pool runs worker 0");
    check(run_pool(2, two, 2), "two-thread laptop pool runs both workers");
    check(run_pool(4, two, 2),
          "four-thread runner pool auto-resolves cpus 2 and 3 and runs 4 workers");
    check(run_pool(4, NULL, 0),
          "four-thread pool with no explicit pins still runs 4 workers");

    printf(fails ? "%d FAILURES\n" : "all passed\n", fails);
    return fails != 0;
}
