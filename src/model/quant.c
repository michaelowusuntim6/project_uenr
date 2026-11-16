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

static void qmat_build4(UenrQMat *q, const float *W, int rows, int cols) {
    if (!q) return;
    q->rows = rows; q->cols = cols;
    if (!W || rows <= 0 || cols <= 0) { q->q4 = NULL; q->scale = NULL; return; }
    size_t nb = ((size_t)rows * cols + 1) / 2;
    q->q4 = (uint8_t *)aligned_alloc(64, ((nb + 63) & ~(size_t)63));
    q->scale = (float *)aligned_alloc(64, (((size_t)rows * sizeof(float)) + 63) & ~(size_t)63);
    if (!q->q4 || !q->scale) { free(q->q4); free(q->scale); q->q4 = NULL; q->scale = NULL; return; }
    memset(q->q4, 0, nb);
    for (int r = 0; r < rows; r++) {
        const float *row = W + (size_t)r * cols;
        float mx = 0.0f;
        for (int c = 0; c < cols; c++) { float a = row[c] < 0 ? -row[c] : row[c]; if (a > mx) mx = a; }
        float sc = mx > 0.0f ? mx / 7.0f : 1.0f;
        q->scale[r] = sc;
        for (int c = 0; c < cols; c++) {
            int iv = (int)(row[c] / sc + (row[c] >= 0 ? 0.5f : -0.5f));
            if (iv > 7) iv = 7;
            if (iv < -7) iv = -7;
            size_t pos = (size_t)r * cols + c;
            if ((c & 1) == 0) q->q4[pos / 2] = (uint8_t)(iv & 0xF);
            else q->q4[pos / 2] |= (uint8_t)((iv & 0xF) << 4);
        }
    }
}

static void qmat_free(UenrQMat *q) {
    if (!q) return;
    free(q->q); free(q->q4); free(q->scale);
    q->q = NULL; q->q4 = NULL; q->scale = NULL; q->rows = q->cols = 0;
}

UenrResult uenr_model_quantize_int4(UenrModel *m) {
    if (!m) return UENR_ERR_INVALID;
    uenr_model_free_q8(m);
    m->q_layers = (UenrQLayer *)calloc((size_t)m->L, sizeof(UenrQLayer));
    if (!m->q_layers) return UENR_ERR_ALLOC;
    int C = m->C, KV = m->KV, E = m->E, EF = m->EF, FF = m->FF;
    for (int l = 0; l < m->L; l++) {
        UenrLayer *y = &m->layers[l];
        UenrQLayer *q = &m->q_layers[l];
        qmat_build4(&q->wq, y->wq, C, C);
        qmat_build4(&q->wk, y->wk, KV, C);
        qmat_build4(&q->wv, y->wv, KV, C);
        qmat_build4(&q->wo, y->wo, C, C);
        qmat_build4(&q->router, y->router, E, C);
        if (y->expert_wg) {
            qmat_build4(&q->expert_wg, y->expert_wg, E * EF, C);
            qmat_build4(&q->expert_wu, y->expert_wu, E * EF, C);
            qmat_build4(&q->expert_wd, y->expert_wd, E * C, EF);
        }
        if (y->shared_wg) {
            qmat_build4(&q->shared_wg, y->shared_wg, FF, C);
            qmat_build4(&q->shared_wu, y->shared_wu, FF, C);
            qmat_build4(&q->shared_wd, y->shared_wd, C, FF);
        }
    }
    qmat_build4(&m->q_lm_head, m->lm_head, m->V, C);
    m->q4_on = 1;
    m->q8_on = 0;
    return UENR_OK;
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
    m->q4_on = 0;
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

static inline __m256 q4_i8_to_ps(__m128i v) {
    __m128i lo = _mm_cvtepi8_epi32(v);
    __m128i hi = _mm_cvtepi8_epi32(_mm_srli_si128(v, 4));
    __m256 f = _mm256_castps128_ps256(_mm_cvtepi32_ps(lo));
    return _mm256_insertf128_ps(f, _mm_cvtepi32_ps(hi), 1);
}
void uenr_gemv_q4(float *y, const float *x, const uint8_t *q4,
                  const float *scale, int out, int in) {
    const __m128i mask = _mm_set1_epi8(0x0F);
    for (int o = 0; o < out; o++) {
        const uint8_t *row = q4 + (size_t)o * in / 2;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 32 <= in; i += 32) {
            __m128i b = _mm_loadu_si128((const __m128i *)(row + i / 2));
            __m128i lo = _mm_and_si128(b, mask);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), mask);
            __m128i v0 = _mm_unpacklo_epi8(lo, hi);
            __m128i v1 = _mm_unpackhi_epi8(lo, hi);
            v0 = _mm_sub_epi8(_mm_xor_si128(v0, _mm_set1_epi8(8)), _mm_set1_epi8(8));
            v1 = _mm_sub_epi8(_mm_xor_si128(v1, _mm_set1_epi8(8)), _mm_set1_epi8(8));
            __m256 f0 = q4_i8_to_ps(v0);
            __m256 f1 = q4_i8_to_ps(_mm_srli_si128(v0, 8));
            __m256 f2 = q4_i8_to_ps(v1);
            __m256 f3 = q4_i8_to_ps(_mm_srli_si128(v1, 8));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(f0, _mm256_loadu_ps(x + i)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(f1, _mm256_loadu_ps(x + i + 8)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(f2, _mm256_loadu_ps(x + i + 16)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(f3, _mm256_loadu_ps(x + i + 24)));
        }
        float s = uenr_v_hsum(acc);
        for (; i < in; i++) {
            uint8_t b = row[i / 2];
            int qv = (i & 1) ? (b >> 4) : (b & 0xF);
            if (qv >= 8) qv -= 16;
            s += x[i] * (float)qv;
        }
        y[o] = s * scale[o];
    }
}

void uenr_swiglu_fwd_q4_1(float *Y, const float *X,
                          const UenrQMat *Wg, const UenrQMat *Wu,
                          const UenrQMat *Wd, float *gbuf, float *ubuf) {
    int C = Wg->cols, F = Wg->rows;
    uenr_gemv_q4(gbuf, X, Wg->q4, Wg->scale, F, C);
    uenr_gemv_q4(ubuf, X, Wu->q4, Wu->scale, F, C);
    for (int i = 0; i < F; i++) gbuf[i] = uenr_siluf(gbuf[i]) * ubuf[i];
    uenr_gemv_q4(Y, gbuf, Wd->q4, Wd->scale, C, F);
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
