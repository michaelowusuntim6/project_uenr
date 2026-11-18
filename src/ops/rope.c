#include "impl_suffix.h"
#define uenr_rope_init UENR_OP(rope_init)
#define uenr_rope_init_ex UENR_OP(rope_init_ex)
#define uenr_rope_init_partial UENR_OP(rope_init_partial)
#define uenr_rope_destroy UENR_OP(rope_destroy)
#define uenr_rope_apply UENR_OP(rope_apply)
#define uenr_rope_apply_bwd UENR_OP(rope_apply_bwd)
#include "uenr/ops.h"
#include "uenr/math.h"

#include <stdlib.h>

UenrResult uenr_rope_init(UenrRope *r, int max_seq, int half, float theta) {
    return uenr_rope_init_ex(r, max_seq, half, theta, 1.0f, 0, 32.0f, 1.0f, 1.0f);
}

UenrResult uenr_rope_init_ex(UenrRope *r, int max_seq, int half, float theta,
                             float scale, int yarn, float beta_fast,
                             float beta_slow, float attn_factor) {
    return uenr_rope_init_partial(r, max_seq, half, theta, 1.0f, scale, yarn,
                                  beta_fast, beta_slow, attn_factor);
}

UenrResult uenr_rope_init_partial(UenrRope *r, int max_seq, int half, float theta,
                                  float partial, float scale, int yarn,
                                  float beta_fast, float beta_slow,
                                  float attn_factor) {
    if (!r || max_seq <= 0 || half <= 0 || theta <= 0.0f || scale <= 0.0f ||
        attn_factor <= 0.0f || partial <= 0.0f) return UENR_ERR_INVALID;
    int rot_pairs = (int)(partial * (float)half);
    if (rot_pairs <= 0) rot_pairs = half;
    if (rot_pairs > half) rot_pairs = half;
    const int hd = half * 2;
    int low = 0, high = hd - 1;
    if (yarn && scale > 1.0f) {
        /* YaRN ramp over dimension index, using the pre-scaling length. */
        float orig = (float)max_seq / scale;
        float ln_theta = uenr_logf(theta);
        float lf = (float)hd * uenr_logf(orig / (beta_fast * 6.2831853f)) / (2.0f * ln_theta);
        float hf = (float)hd * uenr_logf(orig / (beta_slow * 6.2831853f)) / (2.0f * ln_theta);
        low = (int)lf; if (low < 0) low = 0;
        high = (int)hf + 1; if (high > hd - 1) high = hd - 1;
        if (high <= low) high = low + 1;
    }
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
        /* inv_freq[j] = theta^(-2j/hd); interpolate the low-frequency dims. */
        float inv_freq = (j < rot_pairs)
            ? uenr_expf(-(2.0f * (float)j / (float)hd) * ln_theta) : 0.0f;
        if (j < rot_pairs && scale > 1.0f) {
            float ramp = 1.0f;
            if (yarn) {
                ramp = ((float)j - (float)low) / (float)(high - low);
                if (ramp < 0.0f) ramp = 0.0f;
                if (ramp > 1.0f) ramp = 1.0f;
            }
            float interp = inv_freq / scale;
            inv_freq = inv_freq * (1.0f - ramp) + interp * ramp;
        }
        for (int p = 0; p < max_seq; p++) {
            float angle = (float)p * inv_freq;
            r->cos_tab[(size_t)p * half + j] = attn_factor * uenr_cosf(angle);
            r->sin_tab[(size_t)p * half + j] = attn_factor * uenr_sinf(angle);
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
            int j = 0;
            for (; j + 8 <= half; j += 8) {
                __m256 a = _mm256_loadu_ps(v + j);
                __m256 b = _mm256_loadu_ps(v + j + half);
                __m256 c = _mm256_loadu_ps(ct + j);
                __m256 s = _mm256_loadu_ps(st + j);
                _mm256_storeu_ps(v + j, uenr_v_fmsub(a, c, _mm256_mul_ps(b, s)));
                _mm256_storeu_ps(v + j + half, uenr_v_fmadd(a, s, _mm256_mul_ps(b, c)));
            }
            for (; j < half; j++) {
                float a = v[j], b = v[j + half];
                float c = ct[j], s = st[j];
                v[j] = a * c - b * s;
                v[j + half] = a * s + b * c;
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
            int j = 0;
            for (; j + 8 <= half; j += 8) {
                __m256 a = _mm256_loadu_ps(v + j);
                __m256 b = _mm256_loadu_ps(v + j + half);
                __m256 c = _mm256_loadu_ps(ct + j);
                __m256 s = _mm256_loadu_ps(st + j);
                _mm256_storeu_ps(v + j, uenr_v_fmadd(b, s, _mm256_mul_ps(a, c)));
                _mm256_storeu_ps(v + j + half, uenr_v_fmsub(b, c, _mm256_mul_ps(a, s)));
            }
            for (; j < half; j++) {
                float a = v[j], b = v[j + half];
                float c = ct[j], s = st[j];
                v[j] = a * c + b * s;
                v[j + half] = -a * s + b * c;
            }
        }
    }
}
