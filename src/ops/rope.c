#include "uenr/ops.h"
#include "uenr/math.h"

#include <stdlib.h>

UenrResult uenr_rope_init(UenrRope *r, int max_seq, int half, float theta) {
    if (!r || max_seq <= 0 || half <= 0 || theta <= 0.0f) return UENR_ERR_INVALID;
    r->cos_tab = (float *)malloc((size_t)max_seq * half * sizeof(float));
    r->sin_tab = (float *)malloc((size_t)max_seq * half * sizeof(float));
    if (!r->cos_tab || !r->sin_tab) {
        free(r->cos_tab); free(r->sin_tab);
        return UENR_ERR_ALLOC;
    }
    r->max_seq = max_seq;
    r->half = half;

    const float ln_theta = uenr_logf(theta);
    for (int j = 0; j < half; j++) {
        /* inv_freq[j] = theta^(-2j/hd) */
        float inv_freq = uenr_expf(-(2.0f * (float)j / (float)(half * 2)) * ln_theta);
        for (int p = 0; p < max_seq; p++) {
            float angle = (float)p * inv_freq;
            r->cos_tab[(size_t)p * half + j] = uenr_cosf(angle);
            r->sin_tab[(size_t)p * half + j] = uenr_sinf(angle);
        }
    }
    return UENR_OK;
}

void uenr_rope_destroy(UenrRope *r) {
    if (!r) return;
    free(r->cos_tab);
    free(r->sin_tab);
    r->cos_tab = NULL;
    r->sin_tab = NULL;
}

void uenr_rope_apply(const UenrRope *r, float *x, int T, int S,
                     int nheads, int hd) {
    const int half = hd / 2;
    for (int t = 0; t < T; t++) {
        int p = t % S;
        const float *ct = r->cos_tab + (size_t)p * r->half;
        const float *st = r->sin_tab + (size_t)p * r->half;
        for (int h = 0; h < nheads; h++) {
            float *v = x + ((size_t)t * nheads + h) * hd;
            for (int j = 0; j < half; j++) {
                float a = v[2 * j], b = v[2 * j + 1];
                float c = ct[j], s = st[j];
                v[2 * j] = a * c - b * s;
                v[2 * j + 1] = a * s + b * c;
            }
        }
    }
}

void uenr_rope_apply_bwd(const UenrRope *r, float *dx, int T, int S,
                         int nheads, int hd) {
    const int half = hd / 2;
    for (int t = 0; t < T; t++) {
        int p = t % S;
        const float *ct = r->cos_tab + (size_t)p * r->half;
        const float *st = r->sin_tab + (size_t)p * r->half;
        for (int h = 0; h < nheads; h++) {
            float *v = dx + ((size_t)t * nheads + h) * hd;
            for (int j = 0; j < half; j++) {
                float a = v[2 * j], b = v[2 * j + 1];
                float c = ct[j], s = st[j];
                v[2 * j] = a * c + b * s;
                v[2 * j + 1] = -a * s + b * c;
            }
        }
    }
}
