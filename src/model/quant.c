#include "uenr/model.h"
#include "uenr/math.h"
#include "uenr/simd.h"

#include <stdlib.h>
#include <string.h>

static void qmat_build(UenrQMat *q, const float *W, int rows, int cols) {
    if (!q) return;
    q->rows = rows; q->cols = cols;
    if (!W || rows <= 0 || cols <= 0) { q->q = NULL; q->scale = NULL; return; }
    q->q = (int8_t *)aligned_alloc(64, (((size_t)rows * cols) + 63) & ~(size_t)63);
    q->scale = (float *)aligned_alloc(64, (((size_t)rows * sizeof(float)) + 63) & ~(size_t)63);
    if (!q->q || !q->scale) { free(q->q); free(q->scale); q->q = NULL; q->scale = NULL; return; }
    for (int r = 0; r < rows; r++) {
        const float *row = W + (size_t)r * cols;
        float mx = 0.0f;
        for (int c = 0; c < cols; c++) { float a = row[c] < 0 ? -row[c] : row[c]; if (a > mx) mx = a; }
        float s = mx > 0.0f ? mx / 127.0f : 1.0f;
        q->scale[r] = s;
        for (int c = 0; c < cols; c++) {
            float v = row[c] / s;
            int iv = (int)(v + (v >= 0 ? 0.5f : -0.5f));
            if (iv > 127) iv = 127;
            if (iv < -127) iv = -127;
            q->q[(size_t)r * cols + c] = (int8_t)iv;
        }
    }
}

static void qmat_free(UenrQMat *q) {
    if (!q) return;
    free(q->q); free(q->scale);
    q->q = NULL; q->scale = NULL; q->rows = q->cols = 0;
}

UenrResult uenr_model_quantize_int8(UenrModel *m) {
    if (!m) return UENR_ERR_INVALID;
    uenr_model_free_q8(m);
    m->q_layers = (UenrQLayer *)calloc((size_t)m->L, sizeof(UenrQLayer));
    if (!m->q_layers) return UENR_ERR_ALLOC;
    int C = m->C, KV = m->KV, E = m->E, EF = m->EF, FF = m->FF;
    for (int l = 0; l < m->L; l++) {
        UenrLayer *y = &m->layers[l];
        UenrQLayer *q = &m->q_layers[l];
        qmat_build(&q->wq, y->wq, C, C);
        qmat_build(&q->wk, y->wk, KV, C);
        qmat_build(&q->wv, y->wv, KV, C);
        qmat_build(&q->wo, y->wo, C, C);
        qmat_build(&q->router, y->router, E, C);
        if (y->expert_wg) {
            qmat_build(&q->expert_wg, y->expert_wg, E * EF, C);
            qmat_build(&q->expert_wu, y->expert_wu, E * EF, C);
            qmat_build(&q->expert_wd, y->expert_wd, E * C, EF);
        }
        if (y->shared_wg) {
            qmat_build(&q->shared_wg, y->shared_wg, FF, C);
            qmat_build(&q->shared_wu, y->shared_wu, FF, C);
            qmat_build(&q->shared_wd, y->shared_wd, C, FF);
        }
    }
    qmat_build(&m->q_lm_head, m->lm_head, m->V, C);
    m->q8_on = 1;
    return UENR_OK;
}

void uenr_model_free_q8(UenrModel *m) {
    if (!m) return;
    if (m->q_layers) {
        for (int l = 0; l < m->L; l++) {
            UenrQLayer *q = &m->q_layers[l];
            qmat_free(&q->wq); qmat_free(&q->wk); qmat_free(&q->wv); qmat_free(&q->wo);
            qmat_free(&q->router);
            qmat_free(&q->expert_wg); qmat_free(&q->expert_wu); qmat_free(&q->expert_wd);
            qmat_free(&q->shared_wg); qmat_free(&q->shared_wu); qmat_free(&q->shared_wd);
        }
        free(m->q_layers);
        m->q_layers = NULL;
    }
    qmat_free(&m->q_lm_head);
    m->q8_on = 0;
}

void uenr_gemv_q8(float *y, const float *x, const int8_t *q,
                  const float *scale, int out, int in) {
    for (int o = 0; o < out; o++) {
        const int8_t *row = q + (size_t)o * in;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= in; i += 8) {
            __m128i b8 = _mm_loadl_epi64((const __m128i *)(row + i));
            __m128i lo = _mm_cvtepi8_epi32(b8);
            __m128i hi = _mm_cvtepi8_epi32(_mm_srli_si128(b8, 4));
            __m256 bf = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_cvtepi32_ps(lo)),
                _mm_cvtepi32_ps(hi), 1);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(x + i), bf));
        }
        float s = uenr_v_hsum(acc);
        for (; i < in; i++) s += x[i] * (float)row[i];
        y[o] = s * scale[o];
    }
}

void uenr_swiglu_fwd_q8_1(float *Y, const float *X,
                          const UenrQMat *Wg, const UenrQMat *Wu,
                          const UenrQMat *Wd, float *gbuf, float *ubuf) {
    int C = Wg->cols, F = Wg->rows;
    uenr_gemv_q8(gbuf, X, Wg->q, Wg->scale, F, C);
    uenr_gemv_q8(ubuf, X, Wu->q, Wu->scale, F, C);
    for (int i = 0; i < F; i++) gbuf[i] = uenr_siluf(gbuf[i]) * ubuf[i];
    uenr_gemv_q8(Y, gbuf, Wd->q, Wd->scale, C, F);
}
