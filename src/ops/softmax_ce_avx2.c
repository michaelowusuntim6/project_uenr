/* AVX2+FMA softmax + cross-entropy.
 *
 * The polynomial is evaluated with mul+add (uenr_v_muladd) so the AVX2 path
 * matches the scalar/AVX1 exp polynomial; only the linear-algebra kernels use
 * FMA.  The 2^k reconstruction uses a single 256-bit integer shift instead
 * of the two 128-bit halves needed on AVX1. */
#include "uenr/ops.h"
#include "uenr/math.h"
#include "uenr/simd.h"
#include "uenr/thread.h"

static inline __m256 exp8(__m256 x) {
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 inv_ln2 = _mm256_set1_ps(1.4426950408889634f);
    const __m256 hi = _mm256_set1_ps(88.0f);
    const __m256 lo = _mm256_set1_ps(-88.0f);
    const __m256 zero = _mm256_set1_ps(0.0f);
    __m256 hi_mask = _mm256_cmp_ps(x, hi, _CMP_GT_OQ);
    __m256 lo_mask = _mm256_cmp_ps(x, lo, _CMP_LT_OQ);

    x = _mm256_max_ps(_mm256_min_ps(x, hi), lo);
    __m256 kf = _mm256_round_ps(_mm256_mul_ps(x, inv_ln2),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(kf, ln2));

    __m256 p = _mm256_set1_ps(1.0f / 720.0f);
    p = uenr_v_muladd(p, r, _mm256_set1_ps(1.0f / 120.0f));
    p = uenr_v_muladd(p, r, _mm256_set1_ps(1.0f / 24.0f));
    p = uenr_v_muladd(p, r, _mm256_set1_ps(1.0f / 6.0f));
    p = uenr_v_muladd(p, r, _mm256_set1_ps(0.5f));
    p = uenr_v_muladd(p, r, _mm256_set1_ps(1.0f));
    p = uenr_v_muladd(p, r, _mm256_set1_ps(1.0f));

    __m256 out = _mm256_mul_ps(p, uenr_v_pow2i(kf));
    out = _mm256_blendv_ps(out, _mm256_set1_ps(3.402823466e+38f), hi_mask);
    return _mm256_blendv_ps(out, zero, lo_mask);
}

typedef struct {
    const float *logits;
    const int *targets;
    int T, V, ignore_index, write_grad;
    float z_coef;
    float *dlogits;
    float *parts;
} CECtx;

static void zero_row(float *d, int V) {
    int v = 0;
    __m256 z = _mm256_setzero_ps();
    for (; v + 8 <= V; v += 8) _mm256_storeu_ps(d + v, z);
    for (; v < V; v++) d[v] = 0.0f;
}

static void ce_range_avx2(void *ctx, int start, int end, int worker) {
    CECtx *c = (CECtx *)ctx;
    const int V = c->V;
    float acc = 0.0f;
    for (int t = start; t < end; t++) {
        const float *x = c->logits + (size_t)t * V;
        int y = c->targets[t];
        float *d = c->dlogits ? c->dlogits + (size_t)t * V : NULL;

        if (y == c->ignore_index || y < 0 || y >= V) {
            if (c->write_grad && d) zero_row(d, V);
            continue;
        }

        __m256 vmax = _mm256_set1_ps(-3.402823466e+38f);
        int v = 0;
        for (; v + 8 <= V; v += 8)
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + v));
        float maxv = uenr_v_hmax(vmax);
        for (; v < V; v++) if (x[v] > maxv) maxv = x[v];
        float xy = x[y];
        __m256 vmaxv = _mm256_set1_ps(maxv);

        float sum;
        if (c->write_grad && d) {
            __m256 vsum = _mm256_setzero_ps();
            v = 0;
            for (; v + 8 <= V; v += 8) {
                __m256 e = exp8(_mm256_sub_ps(_mm256_loadu_ps(x + v), vmaxv));
                _mm256_storeu_ps(d + v, e);
                vsum = _mm256_add_ps(vsum, e);
            }
            sum = uenr_v_hsum(vsum);
            for (; v < V; v++) {
                float e = uenr_expf(x[v] - maxv);
                d[v] = e;
                sum += e;
            }
            float inv = uenr_rcpf(sum);
            __m256 vinv = _mm256_set1_ps(inv);
            v = 0;
            for (; v + 8 <= V; v += 8)
                _mm256_storeu_ps(d + v, _mm256_mul_ps(_mm256_loadu_ps(d + v), vinv));
            for (; v < V; v++) d[v] *= inv;
            if (c->z_coef > 0.0f) {
                float z = uenr_logf(sum) + maxv;
                __m256 vg = _mm256_set1_ps(2.0f * c->z_coef * z);
                v = 0;
                for (; v + 8 <= V; v += 8) {
                    __m256 dv = _mm256_loadu_ps(d + v);
                    _mm256_storeu_ps(d + v, uenr_v_fmadd(vg, dv, dv));
                }
                for (; v < V; v++) d[v] += (2.0f * c->z_coef * z) * d[v];
                acc += c->z_coef * z * z;
            }
            d[y] -= 1.0f;
        } else {
            __m256 vsum = _mm256_setzero_ps();
            v = 0;
            for (; v + 8 <= V; v += 8)
                vsum = _mm256_add_ps(vsum,
                    exp8(_mm256_sub_ps(_mm256_loadu_ps(x + v), vmaxv)));
            sum = uenr_v_hsum(vsum);
            for (; v < V; v++) sum += uenr_expf(x[v] - maxv);
            if (c->z_coef > 0.0f) {
                float z = uenr_logf(sum) + maxv;
                acc += c->z_coef * z * z;
            }
        }
        acc += (uenr_logf(sum) + maxv) - xy;
    }
    c->parts[worker] += acc;
}

float uenr_avx2_softmax_ce_fwd(const float *logits, const int *targets,
                               int T, int V, int ignore_index,
                               float *dlogits, int write_grad, UenrPool *pool,
                               float z_coef) {
    int nw = pool ? uenr_pool_threads(pool) : 1;
    if (nw > 64) nw = 64;
    float parts[64];
    for (int i = 0; i < 64; i++) parts[i] = 0.0f;
    CECtx c = { logits, targets, T, V, ignore_index, write_grad, z_coef,
                dlogits, parts };
    uenr_pool_range(pool, ce_range_avx2, &c, T);
    double total = 0.0;
    for (int i = 0; i < nw; i++) total += parts[i];
    return (float)total;
}
