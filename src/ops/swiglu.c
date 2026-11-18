#include "impl_suffix.h"
#define uenr_swiglu_fwd UENR_OP(swiglu_fwd)
#define uenr_swiglu_fwd_save UENR_OP(swiglu_fwd_save)
#define uenr_swiglu_bwd_saved UENR_OP(swiglu_bwd_saved)
#define uenr_swiglu_bwd UENR_OP(swiglu_bwd)
#include "uenr/ops.h"
#include "uenr/math.h"

/* h = silu(gate) * up ; Y = h Wd^T
 * gbuf becomes h on output; ubuf holds up. */
void uenr_swiglu_fwd(UenrGemm *g, int n, int C, int F,
                     const float *X, const float *Wg, const float *Wu,
                     const float *Wd, float *Y, float *gbuf, float *ubuf) {
    uenr_linear_fwd(g, gbuf, X, Wg, n, C, F); /* gate */
    uenr_linear_fwd(g, ubuf, X, Wu, n, C, F); /* up   */
    size_t total = (size_t)n * F;
    for (size_t i = 0; i < total; i++)
        gbuf[i] = uenr_siluf(gbuf[i]) * ubuf[i]; /* h */
    uenr_linear_fwd(g, Y, gbuf, Wd, n, F, C);
}

void uenr_swiglu_fwd_save(UenrGemm *g, int n, int C, int F,
                          const float *X, const float *Wg, const float *Wu,
                          const float *Wd, float *Y, float *gate, float *up,
                          float *gbuf) {
    uenr_linear_fwd(g, gate, X, Wg, n, C, F);
    uenr_linear_fwd(g, up, X, Wu, n, C, F);
    size_t total = (size_t)n * F;
    for (size_t i = 0; i < total; i++)
        gbuf[i] = uenr_siluf(gate[i]) * up[i];
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
    for (size_t i = 0; i < total; i++)
        h[i] = uenr_siluf(gate[i]) * up[i];
    uenr_gemm(g, C, F, n, dY, C, 1, h, F, 0, dWd, F, 1.0f, 1.0f);
    uenr_gemm(g, n, F, C, dY, C, 0, Wd, F, 0, h, F, 1.0f, 0.0f);
    for (size_t i = 0; i < total; i++) {
        float gv = gate[i];
        float sv = uenr_sigmoidf(gv);
        float dsilu = sv * (1.0f + gv * (1.0f - sv));
        dg[i] = h[i] * up[i] * dsilu;
        du[i] = h[i] * (gv * sv);
    }
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
    for (size_t i = 0; i < total; i++)
        h[i] = uenr_siluf(gate[i]) * up[i];

    /* dWd += dY^T h */
    uenr_gemm(g, C, F, n, dY, C, 1, h, F, 0, dWd, F, 1.0f, 1.0f);

    /* dh = dY Wd ; h is no longer needed, reuse it. */
    float *dh = h;
    uenr_gemm(g, n, F, C, dY, C, 0, Wd, F, 0, dh, F, 1.0f, 0.0f);

    /* dg = dh * up * silu'(gate), du = dh * silu(gate) */
    for (size_t i = 0; i < total; i++) {
        float gv = gate[i];
        float sv = uenr_sigmoidf(gv);
        float dsilu = sv * (1.0f + gv * (1.0f - sv));
        dg[i] = dh[i] * up[i] * dsilu;
        du[i] = dh[i] * (gv * sv);
    }

    /* dWg += dg^T X ; dWu += du^T X */
    uenr_gemm(g, F, C, n, dg, F, 1, X, C, 0, dWg, C, 1.0f, 1.0f);
    uenr_gemm(g, F, C, n, du, F, 1, X, C, 0, dWu, C, 1.0f, 1.0f);

    /* dX = dg Wg + du Wu */
    uenr_gemm(g, n, C, F, dg, F, 0, Wg, C, 0, dX, C, 1.0f, 0.0f);
    uenr_gemm(g, n, C, F, du, F, 0, Wu, C, 0, dX, C, 1.0f, 1.0f);
}
