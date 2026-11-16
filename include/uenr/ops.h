#ifndef UENR_OPS_H
#define UENR_OPS_H

#include <stddef.h>
#include "uenr/result.h"
#include "uenr/gemm.h"

/* ---- elementwise ------------------------------------------------------- */
void  uenr_zero(float *x, size_t n);
void  uenr_scale(float *x, size_t n, float s);
void  uenr_copy(float *dst, const float *src, size_t n);
float uenr_l2_norm(const float *x, size_t n);
float uenr_l2_norm_p(UenrPool *pool, const float *x, size_t n);
void  uenr_scale_p(UenrPool *pool, float *x, size_t n, float s);
void  uenr_add_scaled(float *dst, const float *src, float s, size_t n);

typedef struct {
    float *cos_tab; /* [max_seq][half] */
    float *sin_tab;
    int    max_seq;
    int    half;
} UenrRope;

/* ---- RMSNorm ----------------------------------------------------------- */
void uenr_rmsnorm_fwd(float *y, float *rstd, const float *x, const float *w,
                      int T, int D, float eps);
void uenr_rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x,
                      const float *w, const float *rstd, int T, int D);
/* Fused per-head RMSNorm + RoPE (q/k prep). rows = T*nheads, D=head_dim. */
void uenr_rmsnorm_rope_fwd(float *y, float *rstd, const float *x, const float *w,
                           const UenrRope *r, int T, int S, int nheads, int hd,
                           float eps);
void uenr_rmsnorm_rope_bwd(float *dx, float *dw, const float *dy, const float *x,
                           const float *w, const float *rstd, const UenrRope *r,
                           int T, int S, int nheads, int hd);
/* x2 = x + proj ; xf = RMSNorm(x2; w). */
void uenr_residual_rmsnorm_fwd(float *x2, float *xf, float *rstd,
                               const float *x, const float *proj,
                               const float *w, int T, int D, float eps);
/* dx2 = dx + RMSNormBackward(dproj; x2). */
void uenr_residual_rmsnorm_bwd(float *dx2, float *dw, const float *dproj,
                               const float *dx, const float *x2,
                               const float *w, const float *rstd,
                               int T, int D);

/* ---- RoPE -------------------------------------------------------------- */
UenrResult uenr_rope_init(UenrRope *r, int max_seq, int half, float theta);
void       uenr_rope_destroy(UenrRope *r);
/* x is [T, nheads, hd], position of row t is (t % S). */
void       uenr_rope_apply(const UenrRope *r, float *x, int T, int S,
                           int nheads, int hd);
void       uenr_rope_apply_bwd(const UenrRope *r, float *dx, int T, int S,
                               int nheads, int hd);

/* ---- GQA causal attention ---------------------------------------------- */
void uenr_attn_fwd(float *out, float *probs,
                   const float *q, const float *k, const float *v,
                   int B, int S, int H, int G, int hd, float scale,
                   UenrPool *pool);
void uenr_attn_bwd(float *dq, float *dk, float *dv,
                   const float *dout, const float *q, const float *k,
                   const float *v, const float *probs,
                   int B, int S, int H, int G, int hd, float scale,
                   UenrPool *pool);

/* ---- FlashAttention-style online softmax (no S^2 buffer) ---------------- */
void uenr_flash_attn_fwd(float *out, float *lse,
                         const float *q, const float *k, const float *v,
                         int B, int S, int H, int G, int hd, float scale,
                         UenrPool *pool);
void uenr_flash_attn_bwd(float *dq, float *dk, float *dv,
                         const float *dout, const float *q, const float *k,
                         const float *v, const float *out, const float *lse,
                         int B, int S, int H, int G, int hd, float scale,
                         UenrPool *pool);

/* ---- SwiGLU ------------------------------------------------------------ */
void uenr_swiglu_fwd(UenrGemm *g, int n, int C, int F,
                     const float *X, const float *Wg, const float *Wu,
                     const float *Wd, float *Y, float *gbuf, float *ubuf);
void uenr_swiglu_fwd_save(UenrGemm *g, int n, int C, int F,
                          const float *X, const float *Wg, const float *Wu,
                          const float *Wd, float *Y, float *gate, float *up,
                          float *gbuf);
void uenr_swiglu_bwd_saved(UenrGemm *g, int n, int C, int F,
                           const float *dY, const float *X,
                           const float *Wg, const float *Wu, const float *Wd,
                           const float *gate, const float *up,
                           float *dX, float *dWg, float *dWu, float *dWd,
                           float *scratch);
/* scratch must hold 5*n*F floats. */
void uenr_swiglu_bwd(UenrGemm *g, int n, int C, int F,
                     const float *dY, const float *X,
                     const float *Wg, const float *Wu, const float *Wd,
                     float *dX, float *dWg, float *dWu, float *dWd,
                     float *scratch);

/* ---- fused softmax + cross-entropy ------------------------------------- */
/* Returns summed loss over non-ignored rows. If dlogits != NULL it is filled. */
float uenr_softmax_ce_fwd(const float *logits, const int *targets,
                          int T, int V, int ignore_index,
                          float *dlogits, int write_grad, UenrPool *pool,
                          float z_coef);

/* ---- optimizer --------------------------------------------------------- */
void uenr_adamw_step(float *p, const float *grad, float *m, float *v, size_t n,
                     int step, float lr, float b1, float b2, float eps,
                     float weight_decay, UenrPool *pool);

#endif /* UENR_OPS_H */
