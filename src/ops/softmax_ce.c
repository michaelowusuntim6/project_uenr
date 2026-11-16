#include "uenr/ops.h"
#include "uenr/math.h"

float uenr_softmax_ce_fwd(const float *logits, const int *targets,
                          int T, int V, int ignore_index,
                          float *dlogits, int write_grad) {
    float total_loss = 0.0f;
    for (int t = 0; t < T; t++) {
        const float *x = logits + (size_t)t * V;
        int y = targets[t];
        float *d = dlogits ? dlogits + (size_t)t * V : NULL;

        if (y == ignore_index || y < 0 || y >= V) {
            if (write_grad && d) {
                for (int v = 0; v < V; v++) d[v] = 0.0f;
            }
            continue;
        }

        float maxv = -3.402823466e+38f;
        for (int v = 0; v < V; v++) if (x[v] > maxv) maxv = x[v];

        float sum = 0.0f;
        if (write_grad && d) {
            for (int v = 0; v < V; v++) {
                float e = uenr_expf(x[v] - maxv);
                d[v] = e;
                sum += e;
            }
            float inv = uenr_rcpf(sum);
            for (int v = 0; v < V; v++) d[v] *= inv;
            d[y] -= 1.0f;
        } else {
            for (int v = 0; v < V; v++) sum += uenr_expf(x[v] - maxv);
        }
        total_loss += (uenr_logf(sum) + maxv) - x[y];
    }
    return total_loss;
}
