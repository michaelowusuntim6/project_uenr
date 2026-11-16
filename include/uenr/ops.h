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
void  uenr_add_scaled(float *dst, const float *src, float s, size_t n);

/* ---- RMSNorm ----------------------------------------------------------- */
void uenr_rmsnorm_fwd(float *y, float *rstd, const float *x, const float *w,
                      int T, int D, float eps);
void uenr_rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x,
                      const float *w, const float *rstd, int T, int D);

/* ---- RoPE -------------------------------------------------------------- */
typedef struct {
    float *cos_tab; /* [max_seq][half] */
    float *sin_tab;
    int    max_seq;
    int    half;
} UenrRope;

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
                   int B, int S, int H, int G, int hd, float scale);
void uenr_attn_bwd(float *dq, float *dk, float *dv,
                   const float *dout, const float *q, const float *k,
                   const float *v, const float *probs,
                   int B, int S, int H, int G, int hd, float scale);

/* ---- SwiGLU ------------------------------------------------------------ */
void uenr_swiglu_fwd(UenrGemm *g, int n, int C, int F,
                     const float *X, const float *Wg, const float *Wu,
                     const float *Wd, float *Y, float *gbuf, float *ubuf);
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
                          float *dlogits, int write_grad);

/* ---- optimizer --------------------------------------------------------- */
void uenr_adamw_step(float *p, const float *grad, float *m, float *v, size_t n,
                     int step, float lr, float b1, float b2, float eps,
                     float weight_decay);

#endif /* UENR_OPS_H */
