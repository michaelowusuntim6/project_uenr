#include "impl_suffix.h"
#define uenr_softmax_ce_fwd UENR_OP(softmax_ce_fwd)
#include "uenr/ops.h"
#include "uenr/math.h"
#include "uenr/thread.h"

typedef struct {
    const float *logits;
    const int *targets;
    int T, V, ignore_index, write_grad;
    float z_coef;
    float *dlogits;
    float *parts;
} CECtx;

static void ce_range(void *ctx, int start, int end, int worker) {
    CECtx *c = (CECtx *)ctx;
    const int V = c->V;
    float acc = 0.0f;
    for (int t = start; t < end; t++) {
        const float *x = c->logits + (size_t)t * V;
        int y = c->targets[t];
        float *d = c->dlogits ? c->dlogits + (size_t)t * V : NULL;

        if (y == c->ignore_index || y < 0 || y >= V) {
            if (c->write_grad && d) {
                for (int v = 0; v < V; v++) d[v] = 0.0f;
            }
            continue;
        }

        float maxv = -3.402823466e+38f;
        for (int v = 0; v < V; v++) if (x[v] > maxv) maxv = x[v];
        float xy = x[y]; /* in-place CE may overwrite x[y] below */

        float sum = 0.0f;
        if (c->write_grad && d) {
            for (int v = 0; v < V; v++) {
                float e = uenr_expf(x[v] - maxv);
                d[v] = e;
                sum += e;
            }
            float inv = uenr_rcpf(sum);
            for (int v = 0; v < V; v++) d[v] *= inv;
            if (c->z_coef > 0.0f) {
                float z = uenr_logf(sum) + maxv;
                float g = 2.0f * c->z_coef * z;
                for (int v = 0; v < V; v++) d[v] += g * d[v];
                acc += c->z_coef * z * z;
            }
            d[y] -= 1.0f;
        } else {
            for (int v = 0; v < V; v++) sum += uenr_expf(x[v] - maxv);
            if (c->z_coef > 0.0f) {
                float z = uenr_logf(sum) + maxv;
                acc += c->z_coef * z * z;
            }
        }
        acc += (uenr_logf(sum) + maxv) - xy;
    }
    c->parts[worker] += acc;
}

float uenr_softmax_ce_fwd(const float *logits, const int *targets,
                          int T, int V, int ignore_index,
                          float *dlogits, int write_grad, UenrPool *pool,
                          float z_coef) {
    int nw = pool ? uenr_pool_threads(pool) : 1;
    if (nw > 64) nw = 64;
    float parts[64];
    for (int i = 0; i < 64; i++) parts[i] = 0.0f;
    CECtx c = { logits, targets, T, V, ignore_index, write_grad, z_coef, dlogits, parts };
    uenr_pool_range(pool, ce_range, &c, T);
    double total = 0.0;
    for (int i = 0; i < nw; i++) total += parts[i];
    return (float)total;
}
