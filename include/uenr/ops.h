#ifndef UENR_OPS_H
#define UENR_OPS_H

#include <stddef.h>
#include <stdint.h>
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
void uenr_rmsnorm_noscale_fwd(float *y, float *rstd, const float *x,
                              int T, int D, float eps);
void uenr_rmsnorm_noscale_bwd(float *dx, const float *dy, const float *x,
                              const float *rstd, int T, int D);
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
UenrResult uenr_rope_init_ex(UenrRope *r, int max_seq, int half, float theta,
                             float scale, int yarn, float beta_fast,
                             float beta_slow, float attn_factor);
UenrResult uenr_rope_init_partial(UenrRope *r, int max_seq, int half, float theta,
                                  float partial, float scale, int yarn,
                                  float beta_fast, float beta_slow,
                                  float attn_factor);
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
                   int window, float softcap, UenrPool *pool);
/* `scratch` (optional, nullable) must hold nthreads*S floats; when supplied
 * the backward path performs no allocation.  Returns UENR_ERR_ALLOC if a
 * fallback allocation fails, so callers cannot silently lose gradients. */
UenrResult uenr_attn_bwd(float *dq, float *dk, float *dv,
                         const float *dout, const float *q, const float *k,
                         const float *v, const float *probs,
                         int B, int S, int H, int G, int hd, float scale,
                         int window, float softcap, float *scratch,
                         UenrPool *pool);
/* FP16-stored probabilities (half the saved-activation footprint). */
void uenr_attn_fwd16(float *out, uint16_t *probs,
                     const float *q, const float *k, const float *v,
                     int B, int S, int H, int G, int hd, float scale,
                     int window, float softcap, UenrPool *pool);
UenrResult uenr_attn_bwd16(float *dq, float *dk, float *dv,
                           const float *dout, const float *q, const float *k,
                           const float *v, const uint16_t *probs,
                           int B, int S, int H, int G, int hd, float scale,
                           int window, float softcap, float *scratch,
                           UenrPool *pool);

/* ---- FlashAttention-style online softmax (no S^2 buffer) ---------------- */
void uenr_flash_attn_fwd(float *out, float *lse,
                         const float *q, const float *k, const float *v,
                         int B, int S, int H, int G, int hd, float scale,
                         int window, float softcap, UenrPool *pool);
void uenr_flash_attn_bwd(float *dq, float *dk, float *dv,
                         const float *dout, const float *q, const float *k,
                         const float *v, const float *out, const float *lse,
                         int B, int S, int H, int G, int hd, float scale,
                         int window, float softcap, UenrPool *pool);

/* ---- SwiGLU ------------------------------------------------------------ */
/* Optional V4-style clamp: gate <= c, linear (up) in [-c, c]; c=0 disables.
 * Set once at model init; inference and training share the same activation. */
void  uenr_swiglu_set_clamp(float c);
float uenr_swiglu_activation(float gate, float up);
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
/* Blockwise int8 first moment + fp32 second moment (1.6x smaller state). */
#define UENR_ADAM8_BLOCK 64
void uenr_adamw_step_m8(float *p, const float *grad, int8_t *m8, float *ms,
                        float *v, size_t n, int step, float lr, float b1,
                        float b2, float eps, float weight_decay);

#endif /* UENR_OPS_H */
