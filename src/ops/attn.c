#include "uenr/ops.h"
#include "uenr/math.h"
#include "uenr/simd.h"

#include <stdlib.h>

static inline float dot_f(const float *a, const float *b, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8)));
        a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16)));
        a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24)));
    }
    float s = uenr_v_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

void uenr_attn_fwd(float *out, float *probs,
                   const float *q, const float *k, const float *v,
                   int B, int S, int H, int G, int hd, float scale) {
    const int group = H / G;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int g = h / group;
            for (int s = 0; s < S; s++) {
                int t = b * S + s;
                const float *qr = q + ((size_t)t * H + h) * hd;
                float *prow = probs + (((size_t)b * H + h) * S + s) * S;
                float maxv = -3.402823466e+38f;
                for (int j = 0; j <= s; j++) {
                    const float *kr = k + ((size_t)(b * S + j) * G + g) * hd;
                    float sc = dot_f(qr, kr, hd) * scale;
                    prow[j] = sc;
                    if (sc > maxv) maxv = sc;
                }
                float sum = 0.0f;
                for (int j = 0; j <= s; j++) {
                    float e = uenr_expf(prow[j] - maxv);
                    prow[j] = e;
                    sum += e;
                }
                float inv = uenr_rcpf(sum);
                for (int j = 0; j <= s; j++) prow[j] *= inv;
                for (int j = s + 1; j < S; j++) prow[j] = 0.0f;

                float *orr = out + ((size_t)t * H + h) * hd;
                for (int d = 0; d < hd; d++) orr[d] = 0.0f;
                for (int j = 0; j <= s; j++) {
                    const float *vr = v + ((size_t)(b * S + j) * G + g) * hd;
                    float p = prow[j];
                    for (int d = 0; d < hd; d++) orr[d] += p * vr[d];
                }
            }
        }
    }
}

void uenr_attn_bwd(float *dq, float *dk, float *dv,
                   const float *dout, const float *q, const float *k,
                   const float *v, const float *probs,
                   int B, int S, int H, int G, int hd, float scale) {
    const int group = H / G;
    float *dprob = (float *)malloc((size_t)S * sizeof(float));
    if (!dprob) return;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int g = h / group;
            for (int s = 0; s < S; s++) {
                int t = b * S + s;
                const float *qr = q + ((size_t)t * H + h) * hd;
                const float *dor = dout + ((size_t)t * H + h) * hd;
                float *dqr = dq + ((size_t)t * H + h) * hd;
                const float *prow = probs + (((size_t)b * H + h) * S + s) * S;

                float dotp = 0.0f;
                for (int j = 0; j <= s; j++) {
                    const float *vr = v + ((size_t)(b * S + j) * G + g) * hd;
                    float dp = dot_f(dor, vr, hd);
                    dprob[j] = dp;
                    dotp += prow[j] * dp;
                }
                for (int j = 0; j <= s; j++) {
                    const float *kr = k + ((size_t)(b * S + j) * G + g) * hd;
                    float ds = prow[j] * (dprob[j] - dotp);
                    float dsc = ds * scale;
                    for (int d = 0; d < hd; d++) {
                        dqr[d] += dsc * kr[d];
                        dk[((size_t)(b * S + j) * G + g) * hd + d] += dsc * qr[d];
                    }
                    float p = prow[j];
                    for (int d = 0; d < hd; d++)
                        dv[((size_t)(b * S + j) * G + g) * hd + d] += p * dor[d];
                }
            }
        }
    }
    free(dprob);
}
