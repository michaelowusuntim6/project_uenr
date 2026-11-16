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

/* Fused per-head RMSNorm + RoPE. Interleaved pair rotation (2j, 2j+1),
 * matching uenr_rope_apply. One pass to write normalized+rotated values;
 * the backward inverse-rotates into a row buffer, then applies norm backward
 * in place (dx may alias dy). */
void uenr_rmsnorm_rope_fwd(float *y, float *rstd, const float *x, const float *w,
                           const UenrRope *r, int T, int S, int nheads, int hd,
                           float eps) {
    const int rows = T * nheads;
    const int half = hd / 2;
    const float invD = 1.0f / (float)hd;
    for (int row = 0; row < rows; row++) {
        const float *xr = x + (size_t)row * hd;
        float *yr = y + (size_t)row * hd;
        float ss = 0.0f;
        for (int d = 0; d < hd; d++) ss += xr[d] * xr[d];
        float rr = uenr_rsqrtf(ss * invD + eps);
        rstd[row] = rr;
        int t = row / nheads;
        int p = t % S;
        const float *ct = r->cos_tab + (size_t)p * r->half;
        const float *st = r->sin_tab + (size_t)p * r->half;
        if (half == r->half) {
            for (int j = 0; j < half; j++) {
                float a = xr[j] * rr * w[j];
                float b = xr[j + half] * rr * w[j + half];
                float c = ct[j], s = st[j];
                yr[j] = a * c - b * s;
                yr[j + half] = a * s + b * c;
            }
        } else {
            for (int d = 0; d < hd; d++) yr[d] = xr[d] * rr * w[d];
        }
    }
}

void uenr_rmsnorm_rope_bwd(float *dx, float *dw, const float *dy, const float *x,
                           const float *w, const float *rstd, const UenrRope *r,
                           int T, int S, int nheads, int hd) {
    if (hd > 512) return;
    const int rows = T * nheads;
    const int half = hd / 2;
    const float invD = 1.0f / (float)hd;
    float g[512];
    for (int row = 0; row < rows; row++) {
        const float *xr = x + (size_t)row * hd;
        const float *dyr = dy + (size_t)row * hd;
        float *dxr = dx + (size_t)row * hd;
        float rr = rstd[row];
        int t = row / nheads;
        int p = t % S;
        const float *ct = r->cos_tab + (size_t)p * r->half;
        const float *st = r->sin_tab + (size_t)p * r->half;
        if (half == r->half) {
            for (int j = 0; j < half; j++) {
                float a = dyr[j], b = dyr[j + half];
                float c = ct[j], s = st[j];
                g[j] = a * c + b * s;
                g[j + half] = -a * s + b * c;
            }
        } else {
            for (int d = 0; d < hd; d++) g[d] = dyr[d];
        }
        float ss = 0.0f;
        for (int d = 0; d < hd; d++) ss += g[d] * w[d] * xr[d];
        float cst = rr * rr * rr * invD * ss;
        for (int d = 0; d < hd; d++) {
            dxr[d] = rr * w[d] * g[d] - cst * xr[d];
            dw[d] += rr * g[d] * xr[d];
        }
    }
}

void uenr_rmsnorm_noscale_fwd(float *y, float *rstd, const float *x,
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
        __m256 rv = _mm256_set1_ps(r);
        d = 0;
        for (; d + 8 <= D; d += 8)
            _mm256_storeu_ps(yr + d, _mm256_mul_ps(_mm256_loadu_ps(xr + d), rv));
        for (; d < D; d++) yr[d] = xr[d] * r;
    }
}

void uenr_rmsnorm_noscale_bwd(float *dx, const float *dy, const float *x,
                              const float *rstd, int T, int D) {
    const float invD = 1.0f / (float)D;
    for (int t = 0; t < T; t++) {
        const float *xr = x + (size_t)t * D;
        const float *dyr = dy + (size_t)t * D;
        float *dxr = dx + (size_t)t * D;
        float r = rstd[t];
        __m256 acc = _mm256_setzero_ps();
        int d = 0;
        for (; d + 8 <= D; d += 8)
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(dyr + d),
                                                   _mm256_loadu_ps(xr + d)));
        float s = uenr_v_hsum(acc);
        for (; d < D; d++) s += dyr[d] * xr[d];
        float c = r * r * r * invD * s;
        __m256 rv = _mm256_set1_ps(r), cv = _mm256_set1_ps(c);
        d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 g = _mm256_loadu_ps(dyr + d);
            __m256 v = _mm256_loadu_ps(xr + d);
            _mm256_storeu_ps(dxr + d, _mm256_sub_ps(_mm256_mul_ps(rv, g),
                                                   _mm256_mul_ps(cv, v)));
        }
        for (; d < D; d++) dxr[d] = r * dyr[d] - c * xr[d];
    }
}

void uenr_residual_rmsnorm_fwd(float *x2, float *xf, float *rstd,
                               const float *x, const float *proj,
                               const float *w, int T, int D, float eps) {
    const float invD = 1.0f / (float)D;
    for (int t = 0; t < T; t++) {
        const float *xr = x + (size_t)t * D;
        const float *pr = proj + (size_t)t * D;
        float *x2r = x2 + (size_t)t * D;
        float *fr = xf + (size_t)t * D;
        __m256 acc = _mm256_setzero_ps();
        int d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 v = _mm256_add_ps(_mm256_loadu_ps(xr + d), _mm256_loadu_ps(pr + d));
            _mm256_storeu_ps(x2r + d, v);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
        }
        float ss = uenr_v_hsum(acc);
        for (; d < D; d++) { float v = xr[d] + pr[d]; x2r[d] = v; ss += v * v; }
        float r = uenr_rsqrtf(ss * invD + eps);
        rstd[t] = r;
        __m256 rv = _mm256_set1_ps(r);
        d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x2r + d), rv);
            _mm256_storeu_ps(fr + d, _mm256_mul_ps(v, _mm256_loadu_ps(w + d)));
        }
        for (; d < D; d++) fr[d] = x2r[d] * r * w[d];
    }
}

void uenr_residual_rmsnorm_bwd(float *dx2, float *dw, const float *dproj,
                               const float *dx, const float *x2,
                               const float *w, const float *rstd,
                               int T, int D) {
    const float invD = 1.0f / (float)D;
    for (int t = 0; t < T; t++) {
        const float *xr = x2 + (size_t)t * D;
        const float *gr = dproj + (size_t)t * D;
        const float *dr = dx + (size_t)t * D;
        float *out = dx2 + (size_t)t * D;
        float r = rstd[t];
        __m256 acc = _mm256_setzero_ps();
        int d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 g = _mm256_loadu_ps(gr + d);
            __m256 wv = _mm256_loadu_ps(w + d);
            __m256 v = _mm256_loadu_ps(xr + d);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_mul_ps(g, wv), v));
        }
        float s = uenr_v_hsum(acc);
        for (; d < D; d++) s += gr[d] * w[d] * xr[d];
        float cst = r * r * r * invD * s;
        __m256 rv = _mm256_set1_ps(r);
        __m256 cv = _mm256_set1_ps(cst);
        d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 g = _mm256_loadu_ps(gr + d);
            __m256 wv = _mm256_loadu_ps(w + d);
            __m256 v = _mm256_loadu_ps(xr + d);
            __m256 outv = _mm256_add_ps(_mm256_loadu_ps(dr + d),
                                        _mm256_sub_ps(_mm256_mul_ps(rv, _mm256_mul_ps(g, wv)),
                                                      _mm256_mul_ps(cv, v)));
            _mm256_storeu_ps(out + d, outv);
            __m256 dwv = _mm256_loadu_ps(dw + d);
            dwv = _mm256_add_ps(dwv, _mm256_mul_ps(rv, _mm256_mul_ps(g, v)));
            _mm256_storeu_ps(dw + d, dwv);
        }
        for (; d < D; d++) {
            out[d] = dr[d] + r * w[d] * gr[d] - cst * xr[d];
            dw[d] += r * gr[d] * xr[d];
        }
    }
}
