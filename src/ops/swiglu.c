#include "impl_suffix.h"
#define uenr_swiglu_fwd UENR_OP(swiglu_fwd)
#define uenr_swiglu_fwd_save UENR_OP(swiglu_fwd_save)
#define uenr_swiglu_bwd_saved UENR_OP(swiglu_bwd_saved)
#define uenr_swiglu_bwd UENR_OP(swiglu_bwd)
#include "uenr/ops.h"
#include "uenr/math.h"
#include "uenr/simd.h"


/* Elementwise SiLU(gate)*up.  The AVX2 translation unit uses the vector
 * polynomial (non-FMA, same coefficients as the scalar uenr_expf) so the
 * scalar exp/sigmoid calls disappear from the profile. */
static inline void silu_mul(float *dst, const float *gate, const float *up,
                            size_t n) {
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        _mm256_storeu_ps(dst + i,
                         _mm256_mul_ps(_mm256_mul_ps(g, uenr_v_sigmoid(g)), u));
    }
    for (; i < n; i++) dst[i] = uenr_siluf(gate[i]) * up[i];
#else
    for (size_t i = 0; i < n; i++) dst[i] = uenr_siluf(gate[i]) * up[i];
#endif
}

static inline void swiglu_grads(float *dg, float *du, const float *dh,
                                const float *gate, const float *up, size_t n) {
#if defined(__AVX2__)
    size_t i = 0;
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 s = uenr_v_sigmoid(g);
        __m256 u = _mm256_loadu_ps(up + i);
        __m256 h = _mm256_loadu_ps(dh + i);
        __m256 ds = _mm256_mul_ps(s,
            _mm256_add_ps(one, _mm256_mul_ps(g, _mm256_sub_ps(one, s))));
        _mm256_storeu_ps(dg + i, _mm256_mul_ps(_mm256_mul_ps(h, u), ds));
        _mm256_storeu_ps(du + i, _mm256_mul_ps(h, _mm256_mul_ps(g, s)));
    }
    for (; i < n; i++) {
        float gv = gate[i];
        float sv = uenr_sigmoidf(gv);
        float dsilu = sv * (1.0f + gv * (1.0f - sv));
        dg[i] = dh[i] * up[i] * dsilu;
        du[i] = dh[i] * (gv * sv);
    }
#else
    for (size_t i = 0; i < n; i++) {
        float gv = gate[i];
        float sv = uenr_sigmoidf(gv);
        float dsilu = sv * (1.0f + gv * (1.0f - sv));
        dg[i] = dh[i] * up[i] * dsilu;
        du[i] = dh[i] * (gv * sv);
    }
#endif
}

/* h = silu(gate) * up ; Y = h Wd^T
 * gbuf becomes h on output; ubuf holds up. */
void uenr_swiglu_fwd(UenrGemm *g, int n, int C, int F,
                     const float *X, const float *Wg, const float *Wu,
                     const float *Wd, float *Y, float *gbuf, float *ubuf) {
    uenr_linear_fwd(g, gbuf, X, Wg, n, C, F); /* gate */
    uenr_linear_fwd(g, ubuf, X, Wu, n, C, F); /* up   */
    size_t total = (size_t)n * F;
    silu_mul(gbuf, gbuf, ubuf, total); /* h */
    uenr_linear_fwd(g, Y, gbuf, Wd, n, F, C);
}

void uenr_swiglu_fwd_save(UenrGemm *g, int n, int C, int F,
                          const float *X, const float *Wg, const float *Wu,
                          const float *Wd, float *Y, float *gate, float *up,
                          float *gbuf) {
    uenr_linear_fwd(g, gate, X, Wg, n, C, F);
    uenr_linear_fwd(g, up, X, Wu, n, C, F);
    size_t total = (size_t)n * F;
    silu_mul(gbuf, gate, up, total);
    uenr_linear_fwd(g, Y, gbuf, Wd, n, F, C);
}

void uenr_swiglu_bwd_saved(UenrGemm *g, int n, int C, int F,
                           const float *dY, const float *X,
                           const float *Wg, const float *Wu, const float *Wd,
                           const float *gate, const float *up,
                           float *dX, float *dWg, float *dWu, float *dWd,
                           float *scratch) {
    float *h = scratch;
    float *dg = h + (size_t)n * F;
    float *du = dg + (size_t)n * F;
    size_t total = (size_t)n * F;
    silu_mul(h, gate, up, total);
    uenr_gemm(g, C, F, n, dY, C, 1, h, F, 0, dWd, F, 1.0f, 1.0f);
    uenr_gemm(g, n, F, C, dY, C, 0, Wd, F, 0, h, F, 1.0f, 0.0f);
    swiglu_grads(dg, du, h, gate, up, total);
    uenr_gemm(g, F, C, n, dg, F, 1, X, C, 0, dWg, C, 1.0f, 1.0f);
    uenr_gemm(g, F, C, n, du, F, 1, X, C, 0, dWu, C, 1.0f, 1.0f);
    uenr_gemm(g, n, C, F, dg, F, 0, Wg, C, 0, dX, C, 1.0f, 0.0f);
    uenr_gemm(g, n, C, F, du, F, 0, Wu, C, 0, dX, C, 1.0f, 1.0f);
}

void uenr_swiglu_bwd(UenrGemm *g, int n, int C, int F,
                     const float *dY, const float *X,
                     const float *Wg, const float *Wu, const float *Wd,
                     float *dX, float *dWg, float *dWu, float *dWd,
                     float *scratch) {
    float *gate = scratch;
    float *up   = gate + (size_t)n * F;
    float *h    = up   + (size_t)n * F;   /* reused for dh after dWd */
    float *dg   = h    + (size_t)n * F;
    float *du   = dg   + (size_t)n * F;

    /* Recompute gate/up/h from the saved input. */
    uenr_linear_fwd(g, gate, X, Wg, n, C, F);
    uenr_linear_fwd(g, up, X, Wu, n, C, F);
    size_t total = (size_t)n * F;
    silu_mul(h, gate, up, total);

    /* dWd += dY^T h */
    uenr_gemm(g, C, F, n, dY, C, 1, h, F, 0, dWd, F, 1.0f, 1.0f);

    /* dh = dY Wd ; h is no longer needed, reuse it. */
    float *dh = h;
    uenr_gemm(g, n, F, C, dY, C, 0, Wd, F, 0, dh, F, 1.0f, 0.0f);

    /* dg = dh * up * silu'(gate), du = dh * silu(gate) */
    swiglu_grads(dg, du, dh, gate, up, total);

    /* dWg += dg^T X ; dWu += du^T X */
    uenr_gemm(g, F, C, n, dg, F, 1, X, C, 0, dWg, C, 1.0f, 1.0f);
    uenr_gemm(g, F, C, n, du, F, 1, X, C, 0, dWu, C, 1.0f, 1.0f);

    /* dX = dg Wg + du Wu */
    uenr_gemm(g, n, C, F, dg, F, 0, Wg, C, 0, dX, C, 1.0f, 0.0f);
    uenr_gemm(g, n, C, F, du, F, 0, Wu, C, 0, dX, C, 1.0f, 1.0f);
}
