/* Runtime SIMD dispatch for the non-GEMM hot kernels.
 *
 * Every public op in ops.h is a thin wrapper over a function-pointer table.
 * The table points at either the avx1 or avx2 suffixed implementation (both
 * are always linked).  uenr_simd_configure() refreshes it; a lazy refresh in
 * active() covers tools/tests that never call configure.  GEMM has its own
 * dispatch inside gemm.c because it also carries the scratch/pool state. */
#include "uenr/ops.h"
#include "uenr/simd_dispatch.h"

#define D2(ret, name, args) \
    ret uenr_avx1_##name args; \
    ret uenr_avx2_##name args;

D2(void, zero, (float *x, size_t n))
D2(void, scale, (float *x, size_t n, float s))
D2(void, copy, (float *dst, const float *src, size_t n))
D2(float, l2_norm, (const float *x, size_t n))
D2(float, l2_norm_p, (UenrPool *pool, const float *x, size_t n))
D2(void, scale_p, (UenrPool *pool, float *x, size_t n, float s))
D2(void, add_scaled, (float *dst, const float *src, float s, size_t n))

D2(void, rmsnorm_fwd, (float *y, float *rstd, const float *x, const float *w,
                       int T, int D, float eps))
D2(void, rmsnorm_bwd, (float *dx, float *dw, const float *dy, const float *x,
                       const float *w, const float *rstd, int T, int D))
D2(void, rmsnorm_noscale_fwd, (float *y, float *rstd, const float *x,
                               int T, int D, float eps))
D2(void, rmsnorm_noscale_bwd, (float *dx, const float *dy, const float *x,
                               const float *rstd, int T, int D))
D2(void, rmsnorm_rope_fwd, (float *y, float *rstd, const float *x,
                            const float *w, const UenrRope *r, int T, int S,
                            int nheads, int hd, float eps))
D2(void, rmsnorm_rope_bwd, (float *dx, float *dw, const float *dy,
                            const float *x, const float *w,
                            const float *rstd, const UenrRope *r, int T, int S,
                            int nheads, int hd))
D2(void, residual_rmsnorm_fwd, (float *x2, float *xf, float *rstd,
                                const float *x, const float *proj,
                                const float *w, int T, int D, float eps))
D2(void, residual_rmsnorm_bwd, (float *dx2, float *dw, const float *dproj,
                                const float *dx, const float *x2,
                                const float *w, const float *rstd,
                                int T, int D))

D2(UenrResult, rope_init, (UenrRope *r, int max_seq, int half, float theta))
D2(UenrResult, rope_init_ex, (UenrRope *r, int max_seq, int half, float theta,
                              float scale, int yarn, float beta_fast,
                              float beta_slow, float attn_factor))
D2(UenrResult, rope_init_partial, (UenrRope *r, int max_seq, int half,
                                   float theta, float partial, float scale,
                                   int yarn, float beta_fast, float beta_slow,
                                   float attn_factor))
D2(void, rope_destroy, (UenrRope *r))
D2(void, rope_apply, (const UenrRope *r, float *x, int T, int S,
                      int nheads, int hd))
D2(void, rope_apply_bwd, (const UenrRope *r, float *dx, int T, int S,
                          int nheads, int hd))

D2(void, attn_fwd, (float *out, float *probs, const float *q, const float *k,
                    const float *v, int B, int S, int H, int G, int hd,
                    float scale, int window, float softcap, const int *doc_start,
                    UenrPool *pool))
D2(UenrResult, attn_bwd, (float *dq, float *dk, float *dv, const float *dout,
                          const float *q, const float *k, const float *v,
                          const float *probs, int B, int S, int H, int G,
                          int hd, float scale, int window, float softcap,
                          const int *doc_start, float *scratch, UenrPool *pool))
D2(void, attn_fwd16, (float *out, uint16_t *probs, const float *q,
                      const float *k, const float *v, int B, int S, int H,
                      int G, int hd, float scale, int window, float softcap,
                      const int *doc_start, UenrPool *pool))
D2(UenrResult, attn_bwd16, (float *dq, float *dk, float *dv,
                            const float *dout, const float *q, const float *k,
                            const float *v, const uint16_t *probs,
                            int B, int S, int H, int G, int hd, float scale,
                            int window, float softcap, const int *doc_start,
                            float *scratch, UenrPool *pool))

#define D3(ret, name, args) ret uenr_avx512_##name args;
D3(void, attn_fwd, (float *out, float *probs, const float *q, const float *k,
                    const float *v, int B, int S, int H, int G, int hd,
                    float scale, int window, float softcap, const int *doc_start,
                    UenrPool *pool))
D3(UenrResult, attn_bwd, (float *dq, float *dk, float *dv, const float *dout,
                          const float *q, const float *k, const float *v,
                          const float *probs, int B, int S, int H, int G,
                          int hd, float scale, int window, float softcap,
                          const int *doc_start, float *scratch, UenrPool *pool))
D3(void, attn_fwd16, (float *out, uint16_t *probs, const float *q,
                      const float *k, const float *v, int B, int S, int H,
                      int G, int hd, float scale, int window, float softcap,
                      const int *doc_start, UenrPool *pool))
D3(UenrResult, attn_bwd16, (float *dq, float *dk, float *dv,
                            const float *dout, const float *q, const float *k,
                            const float *v, const uint16_t *probs,
                            int B, int S, int H, int G, int hd, float scale,
                            int window, float softcap, const int *doc_start,
                            float *scratch, UenrPool *pool))

D2(void, swiglu_fwd, (UenrGemm *g, int n, int C, int F, const float *X,
                      const float *Wg, const float *Wu, const float *Wd,
                      float *Y, float *gbuf, float *ubuf))
D2(void, swiglu_fwd_save, (UenrGemm *g, int n, int C, int F, const float *X,
                           const float *Wg, const float *Wu, const float *Wd,
                           float *Y, float *gate, float *up, float *gbuf))
D2(void, swiglu_bwd_saved, (UenrGemm *g, int n, int C, int F,
                            const float *dY, const float *X,
                            const float *Wg, const float *Wu,
                            const float *Wd, const float *gate,
                            const float *up, float *dX, float *dWg,
                            float *dWu, float *dWd, float *scratch))
D2(void, swiglu_bwd, (UenrGemm *g, int n, int C, int F, const float *dY,
                      const float *X, const float *Wg, const float *Wu,
                      const float *Wd, float *dX, float *dWg, float *dWu,
                      float *dWd, float *scratch))

D2(float, softmax_ce_fwd, (const float *logits, const int *targets, int T,
                           int V, int ignore_index, float *dlogits,
                           int write_grad, UenrPool *pool, float z_coef))

D2(void, adamw_step, (float *p, const float *grad, float *m, float *v,
                      size_t n, int step, float lr, float b1, float b2,
                      float eps, float weight_decay, UenrPool *pool))
D2(void, adamw_step_m8, (float *p, const float *grad, int8_t *m8, float *ms,
                         float *v, size_t n, int step, float lr, float b1,
                         float b2, float eps, float weight_decay))

typedef struct {
    void (*zero)(float *, size_t);
    void (*scale)(float *, size_t, float);
    void (*copy)(float *, const float *, size_t);
    float (*l2_norm)(const float *, size_t);
    float (*l2_norm_p)(UenrPool *, const float *, size_t);
    void (*scale_p)(UenrPool *, float *, size_t, float);
    void (*add_scaled)(float *, const float *, float, size_t);
    void (*rmsnorm_fwd)(float *, float *, const float *, const float *, int, int, float);
    void (*rmsnorm_bwd)(float *, float *, const float *, const float *, const float *, const float *, int, int);
    void (*rmsnorm_noscale_fwd)(float *, float *, const float *, int, int, float);
    void (*rmsnorm_noscale_bwd)(float *, const float *, const float *, const float *, int, int);
    void (*rmsnorm_rope_fwd)(float *, float *, const float *, const float *, const UenrRope *, int, int, int, int, float);
    void (*rmsnorm_rope_bwd)(float *, float *, const float *, const float *, const float *, const float *, const UenrRope *, int, int, int, int);
    void (*residual_rmsnorm_fwd)(float *, float *, float *, const float *, const float *, const float *, int, int, float);
    void (*residual_rmsnorm_bwd)(float *, float *, const float *, const float *, const float *, const float *, const float *, int, int);
    UenrResult (*rope_init)(UenrRope *, int, int, float);
    UenrResult (*rope_init_ex)(UenrRope *, int, int, float, float, int, float, float, float);
    UenrResult (*rope_init_partial)(UenrRope *, int, int, float, float, float, int, float, float, float);
    void (*rope_destroy)(UenrRope *);
    void (*rope_apply)(const UenrRope *, float *, int, int, int, int);
    void (*rope_apply_bwd)(const UenrRope *, float *, int, int, int, int);
    void (*attn_fwd)(float *, float *, const float *, const float *, const float *, int, int, int, int, int, float, int, float, const int *, UenrPool *);
    UenrResult (*attn_bwd)(float *, float *, float *, const float *, const float *, const float *, const float *, const float *, int, int, int, int, int, float, int, float, const int *, float *, UenrPool *);
    void (*attn_fwd16)(float *, uint16_t *, const float *, const float *, const float *, int, int, int, int, int, float, int, float, const int *, UenrPool *);
    UenrResult (*attn_bwd16)(float *, float *, float *, const float *, const float *, const float *, const float *, const uint16_t *, int, int, int, int, int, float, int, float, const int *, float *, UenrPool *);
    void (*swiglu_fwd)(UenrGemm *, int, int, int, const float *, const float *, const float *, const float *, float *, float *, float *);
    void (*swiglu_fwd_save)(UenrGemm *, int, int, int, const float *, const float *, const float *, const float *, float *, float *, float *, float *);
    void (*swiglu_bwd_saved)(UenrGemm *, int, int, int, const float *, const float *, const float *, const float *, const float *, const float *, const float *, float *, float *, float *, float *, float *);
    void (*swiglu_bwd)(UenrGemm *, int, int, int, const float *, const float *, const float *, const float *, const float *, float *, float *, float *, float *, float *);
    float (*softmax_ce_fwd)(const float *, const int *, int, int, int, float *, int, UenrPool *, float);
    void (*adamw_step)(float *, const float *, float *, float *, size_t, int, float, float, float, float, float, UenrPool *);
    void (*adamw_step_m8)(float *, const float *, int8_t *, float *, float *, size_t, int, float, float, float, float, float);
} Ops;

#define OP1(f) uenr_avx1_##f
#define OP2(f) uenr_avx2_##f
#define OP3(f) uenr_avx512_##f
#define OPS_LIST(P, A) \
    P(zero), P(scale), P(copy), P(l2_norm), P(l2_norm_p), \
    P(scale_p), P(add_scaled), P(rmsnorm_fwd), P(rmsnorm_bwd), \
    P(rmsnorm_noscale_fwd), P(rmsnorm_noscale_bwd), \
    P(rmsnorm_rope_fwd), P(rmsnorm_rope_bwd), \
    P(residual_rmsnorm_fwd), P(residual_rmsnorm_bwd), \
    P(rope_init), P(rope_init_ex), P(rope_init_partial), \
    P(rope_destroy), P(rope_apply), P(rope_apply_bwd), \
    A(attn_fwd), A(attn_bwd), A(attn_fwd16), A(attn_bwd16), \
    P(swiglu_fwd), P(swiglu_fwd_save), P(swiglu_bwd_saved), \
    P(swiglu_bwd), P(softmax_ce_fwd), P(adamw_step), P(adamw_step_m8)
static const Ops ops_avx1 = { OPS_LIST(OP1, OP1) };
static const Ops ops_avx2 = { OPS_LIST(OP2, OP2) };
static const Ops ops_avx512 = { OPS_LIST(OP2, OP3) };

static const Ops *g_active;
static const Ops *table_for_level(int level) {
    if (level == UENR_SIMD_AVX512) return &ops_avx512;
    if (level == UENR_SIMD_AVX2) return &ops_avx2;
    return &ops_avx1;
}
static const Ops *active(void) {
    if (!g_active) g_active = table_for_level(uenr_simd_level());
    return g_active;
}
void uenr_ops_refresh(void) {
    g_active = table_for_level(uenr_simd_level());
}

/* ---- public wrappers --------------------------------------------------- */
void uenr_zero(float *x, size_t n) { active()->zero(x, n); }
void uenr_scale(float *x, size_t n, float s) { active()->scale(x, n, s); }
void uenr_copy(float *dst, const float *src, size_t n) { active()->copy(dst, src, n); }
float uenr_l2_norm(const float *x, size_t n) { return active()->l2_norm(x, n); }
float uenr_l2_norm_p(UenrPool *p, const float *x, size_t n) { return active()->l2_norm_p(p, x, n); }
void uenr_scale_p(UenrPool *p, float *x, size_t n, float s) { active()->scale_p(p, x, n, s); }
void uenr_add_scaled(float *d, const float *s, float k, size_t n) { active()->add_scaled(d, s, k, n); }

void uenr_rmsnorm_fwd(float *y, float *r, const float *x, const float *w, int T, int D, float e) { active()->rmsnorm_fwd(y, r, x, w, T, D, e); }
void uenr_rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x, const float *w, const float *r, int T, int D) { active()->rmsnorm_bwd(dx, dw, dy, x, w, r, T, D); }
void uenr_rmsnorm_noscale_fwd(float *y, float *r, const float *x, int T, int D, float e) { active()->rmsnorm_noscale_fwd(y, r, x, T, D, e); }
void uenr_rmsnorm_noscale_bwd(float *dx, const float *dy, const float *x, const float *r, int T, int D) { active()->rmsnorm_noscale_bwd(dx, dy, x, r, T, D); }
void uenr_rmsnorm_rope_fwd(float *y, float *rs, const float *x, const float *w, const UenrRope *r, int T, int S, int H, int hd, float e) { active()->rmsnorm_rope_fwd(y, rs, x, w, r, T, S, H, hd, e); }
void uenr_rmsnorm_rope_bwd(float *dx, float *dw, const float *dy, const float *x, const float *w, const float *rs, const UenrRope *r, int T, int S, int H, int hd) { active()->rmsnorm_rope_bwd(dx, dw, dy, x, w, rs, r, T, S, H, hd); }
void uenr_residual_rmsnorm_fwd(float *x2, float *xf, float *rs, const float *x, const float *p, const float *w, int T, int D, float e) { active()->residual_rmsnorm_fwd(x2, xf, rs, x, p, w, T, D, e); }
void uenr_residual_rmsnorm_bwd(float *dx2, float *dw, const float *dp, const float *dx, const float *x2, const float *w, const float *rs, int T, int D) { active()->residual_rmsnorm_bwd(dx2, dw, dp, dx, x2, w, rs, T, D); }

UenrResult uenr_rope_init(UenrRope *r, int m, int h, float t) { return active()->rope_init(r, m, h, t); }
UenrResult uenr_rope_init_ex(UenrRope *r, int m, int h, float t, float sc, int y, float bf, float bs, float af) { return active()->rope_init_ex(r, m, h, t, sc, y, bf, bs, af); }
UenrResult uenr_rope_init_partial(UenrRope *r, int m, int h, float t, float p, float sc, int y, float bf, float bs, float af) { return active()->rope_init_partial(r, m, h, t, p, sc, y, bf, bs, af); }
void uenr_rope_destroy(UenrRope *r) { active()->rope_destroy(r); }
void uenr_rope_apply(const UenrRope *r, float *x, int T, int S, int H, int hd) { active()->rope_apply(r, x, T, S, H, hd); }
void uenr_rope_apply_bwd(const UenrRope *r, float *x, int T, int S, int H, int hd) { active()->rope_apply_bwd(r, x, T, S, H, hd); }

void uenr_attn_fwd(float *o, float *p, const float *q, const float *k, const float *v, int B, int S, int H, int G, int hd, float sc, int w, float cap, const int *ds, UenrPool *pool) { active()->attn_fwd(o, p, q, k, v, B, S, H, G, hd, sc, w, cap, ds, pool); }
UenrResult uenr_attn_bwd(float *dq, float *dk, float *dv, const float *dout, const float *q, const float *k, const float *v, const float *p, int B, int S, int H, int G, int hd, float sc, int w, float cap, const int *ds, float *scratch, UenrPool *pool) { return active()->attn_bwd(dq, dk, dv, dout, q, k, v, p, B, S, H, G, hd, sc, w, cap, ds, scratch, pool); }
void uenr_attn_fwd16(float *o, uint16_t *p, const float *q, const float *k, const float *v, int B, int S, int H, int G, int hd, float sc, int w, float cap, const int *ds, UenrPool *pool) { active()->attn_fwd16(o, p, q, k, v, B, S, H, G, hd, sc, w, cap, ds, pool); }
UenrResult uenr_attn_bwd16(float *dq, float *dk, float *dv, const float *dout, const float *q, const float *k, const float *v, const uint16_t *p, int B, int S, int H, int G, int hd, float sc, int w, float cap, const int *ds, float *scratch, UenrPool *pool) { return active()->attn_bwd16(dq, dk, dv, dout, q, k, v, p, B, S, H, G, hd, sc, w, cap, ds, scratch, pool); }

void uenr_swiglu_fwd(UenrGemm *g, int n, int C, int F, const float *X, const float *Wg, const float *Wu, const float *Wd, float *Y, float *gb, float *ub) { active()->swiglu_fwd(g, n, C, F, X, Wg, Wu, Wd, Y, gb, ub); }
void uenr_swiglu_fwd_save(UenrGemm *g, int n, int C, int F, const float *X, const float *Wg, const float *Wu, const float *Wd, float *Y, float *gate, float *up, float *gb) { active()->swiglu_fwd_save(g, n, C, F, X, Wg, Wu, Wd, Y, gate, up, gb); }
void uenr_swiglu_bwd_saved(UenrGemm *g, int n, int C, int F, const float *dY, const float *X, const float *Wg, const float *Wu, const float *Wd, const float *gate, const float *up, float *dX, float *dWg, float *dWu, float *dWd, float *scratch) { active()->swiglu_bwd_saved(g, n, C, F, dY, X, Wg, Wu, Wd, gate, up, dX, dWg, dWu, dWd, scratch); }
void uenr_swiglu_bwd(UenrGemm *g, int n, int C, int F, const float *dY, const float *X, const float *Wg, const float *Wu, const float *Wd, float *dX, float *dWg, float *dWu, float *dWd, float *scratch) { active()->swiglu_bwd(g, n, C, F, dY, X, Wg, Wu, Wd, dX, dWg, dWu, dWd, scratch); }

float uenr_softmax_ce_fwd(const float *l, const int *t, int T, int V, int ign, float *d, int wg, UenrPool *pool, float z) { return active()->softmax_ce_fwd(l, t, T, V, ign, d, wg, pool, z); }

void uenr_adamw_step(float *p, const float *g, float *m, float *v, size_t n, int st, float lr, float b1, float b2, float eps, float wd, UenrPool *pool) { active()->adamw_step(p, g, m, v, n, st, lr, b1, b2, eps, wd, pool); }
void uenr_adamw_step_m8(float *p, const float *g, int8_t *m8, float *ms, float *v, size_t n, int st, float lr, float b1, float b2, float eps, float wd) { active()->adamw_step_m8(p, g, m8, ms, v, n, st, lr, b1, b2, eps, wd); }
