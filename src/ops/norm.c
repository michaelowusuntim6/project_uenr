#include "uenr/ops.h"
#include "uenr/math.h"
#include "uenr/simd.h"

void uenr_rmsnorm_fwd(float *y, float *rstd, const float *x, const float *w,
                      int T, int D, float eps) {
    const float invD = 1.0f / (float)D;
    for (int t = 0; t < T; t++) {
        const float *xr = x + (size_t)t * D;
        float *yr = y + (size_t)t * D;

        __m256 acc = _mm256_setzero_ps();
        int d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 v = _mm256_loadu_ps(xr + d);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
        }
        float ss = uenr_v_hsum(acc);
        for (; d < D; d++) ss += xr[d] * xr[d];

        float r = uenr_rsqrtf(ss * invD + eps);
        rstd[t] = r;

        d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 v = _mm256_loadu_ps(xr + d);
            __m256 wv = _mm256_loadu_ps(w + d);
            _mm256_storeu_ps(yr + d, _mm256_mul_ps(_mm256_mul_ps(v, _mm256_set1_ps(r)), wv));
        }
        for (; d < D; d++) yr[d] = xr[d] * r * w[d];
    }
}

void uenr_rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x,
                      const float *w, const float *rstd, int T, int D) {
    const float invD = 1.0f / (float)D;
    for (int t = 0; t < T; t++) {
        const float *xr = x + (size_t)t * D;
        const float *dyr = dy + (size_t)t * D;
        float *dxr = dx + (size_t)t * D;
        float r = rstd[t];

        /* sum_i dy_i * w_i * x_i */
        __m256 acc = _mm256_setzero_ps();
        int d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 v = _mm256_loadu_ps(xr + d);
            __m256 g = _mm256_loadu_ps(dyr + d);
            __m256 wv = _mm256_loadu_ps(w + d);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_mul_ps(g, wv), v));
        }
        float s = uenr_v_hsum(acc);
        for (; d < D; d++) s += dyr[d] * w[d] * xr[d];

        float c = r * r * r * invD * s;

        d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 g = _mm256_loadu_ps(dyr + d);
            __m256 wv = _mm256_loadu_ps(w + d);
            __m256 v = _mm256_loadu_ps(xr + d);
            __m256 dxv = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(r), _mm256_mul_ps(g, wv)),
                                       _mm256_mul_ps(_mm256_set1_ps(c), v));
            _mm256_storeu_ps(dxr + d, dxv);
            /* dw += dy * x * r */
            __m256 dwv = _mm256_loadu_ps(dw + d);
            dwv = _mm256_add_ps(dwv, _mm256_mul_ps(_mm256_set1_ps(r), _mm256_mul_ps(g, v)));
            _mm256_storeu_ps(dw + d, dwv);
        }
        for (; d < D; d++) {
            dxr[d] = r * dyr[d] * w[d] - c * xr[d];
            dw[d] += r * dyr[d] * xr[d];
        }
    }
}
