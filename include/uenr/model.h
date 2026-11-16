#ifndef UENR_MODEL_H
#define UENR_MODEL_H

#include <stddef.h>
#include "uenr/config.h"
#include "uenr/gemm.h"
#include "uenr/ops.h"
#include "uenr/rng.h"
#include "uenr/result.h"

#define UENR_MAX_TENSORS 1024

typedef struct {
    float       *p;
    size_t       off;   /* index into the contiguous parameter block */
    size_t       n;
    const char  *name;
    int          no_decay;
} UenrTensor;

typedef struct {
    float *Aq, *Bq, *Ak, *Bk, *Av, *Bv, *Ao, *Bo;
    int has_q, has_k, has_v, has_o;
} UenrLoraLayer;

typedef struct {
    float *rms_attn;
    float *wq, *wk, *wv, *wo;
    float *q_norm, *k_norm;
    float *rms_ffn;
    float *ls;                 /* per-channel attention LayerScale [C] */
    int    type;               /* 0 = sliding/local, 1 = full/global */
    int    is_shared;
    int    donor;              /* KV donor layer (self when active) */
    int    slot;               /* compact KV cache slot (donor slot when shared) */
    float *router;
    float *expert_wg, *expert_wu, *expert_wd; /* [E][EF,C], [E][EF,C], [E][C,EF] */
    float *shared_wg, *shared_wu, *shared_wd; /* [FF,C], [FF,C], [C,FF] */
} UenrLayer;

typedef struct {
    int8_t *q;
    float  *scale;
    int     rows, cols;
} UenrQMat;

typedef struct {
    UenrQMat wq, wk, wv, wo, router;
    UenrQMat expert_wg, expert_wu, expert_wd;
    UenrQMat shared_wg, shared_wu, shared_wd;
} UenrQLayer;

typedef struct {
    UenrConfig cfg;
    int V, C, L, H, G, hd, KV, E, EF, FF, K, tie;
    int n_slots;               /* number of active KV cache slots */

    float *params;
    size_t alloc_floats;
    UenrTensor tensors[UENR_MAX_TENSORS];
    int n_tensors;

    UenrLayer *layers;
    float *tok_emb;
    float *rms_final;
    float *lm_head;
    UenrRope rope;

    float *grad;
    float *adam_m;
    float *adam_v;
    int    opt_ready;
    float *ema;

    /* INT8 inference weights. */
    UenrQLayer *q_layers;
    UenrQMat    q_lm_head;
    int         q8_on;

    /* Optional LoRA adapters (base weights frozen while active). */
    int    lora_on, freeze_base, lora_rank;
    float  lora_scale;
    float *lora_p, *lora_g, *lora_m, *lora_v;
    size_t lora_n;
    UenrLoraLayer *lora_layers;
} UenrModel;

/* Activations for one micro-batch (or one decode sequence). */
typedef struct UenrActs {
    int T, S, B, C, L, H, G, hd, KV, E, EF, FF, V, K;
    int T_cur, S_cur;

    float *x;          /* [(L+1) T C] */
    float *xa;         /* [L T C] */
    float *q_pre, *q;  /* [L T C] */
    float *k_pre, *k;  /* [L T KV] */
    float *v;          /* [L T KV] */
    float *probs;      /* [L B H S S] (NULL when flash attention) */
    float *lse;        /* [L T H] log-sum-exp per query (flash path) */
    int    flash;
    float *attn_out;   /* [L T C] */
    float *x2;         /* [L T C] */
    float *xf;         /* [L T C] */
    float *rprobs;     /* [L T E] */
    int   *eidx;       /* [L T K] */
    float *egate;      /* [L T K] */
    float *routed_y;   /* [L T K C] */
    float *eout;       /* [L T C] */
    int   *ecount;     /* [L E] */
    int   *eoff, *elist, *cursor; /* [E+1], [T K], [E] */
    const int *tokens_cur;
    float *rstd_attn;  /* [L T] */
    float *rstd_ffn;   /* [L T] */
    float *rstd_q;     /* [L T H] */
    float *rstd_k;     /* [L T G] */
    float *rstd_final; /* [T] */
    float *xfinal;     /* [T C] */
    float *logits;     /* [T V] */
    float *dlogits;    /* [T V] */

    float *dx, *dxa, *dattn, *dq, *dk, *dv, *dxf, *dx2, *drouter, *tmp;
    float *gate_dp;
    float *gxs, *gys, *gdys, *sscratch;
    int    store_swiglu;
    float *swg_r, *swu_r;   /* [L T K EF] */
    float *swg_s, *swu_s;   /* [L T FF] */
    float *ls_proj;         /* [L T C] unscaled attention projection */
    float *swg_tmp, *swu_tmp; /* [T max(EF,FF)] */
    float *dk_acc, *dv_acc;   /* [L T KV] donor gradient accumulation */
    float *lora_h, *lora_dh;

    /* Sampled softmax (target + negatives) for training. */
    int    sampled, sn, sampled_last;
    float *slogits;  /* [T*sn] */
    int   *sids;     /* [T*sn] */
    UenrRng sample_rng;
} UenrActs;

typedef struct {
    int L, KV, cap, len;
    int int8, G, hd;
    int n_slots;
    int *type, *donor, *shared;
    float *k, *v;            /* [L][cap][KV] (fp32 mode) */
    int8_t *k8, *v8;         /* [L][cap][KV] (int8 mode) */
    float *ks, *vs;          /* [L*G] per-head scales */
    float *x, *xa, *q, *krow, *vrow, *attn, *proj, *xf, *eout, *ys;
    float *gbuf, *ubuf, *router_logits, *xfinal, *logits;
    int   *topk_idx;
    float *topk_val;
    float  rstd;
} UenrKVCache;

UenrResult uenr_kv_cache_init(UenrKVCache *c, const UenrConfig *cfg, int cap);
void       uenr_kv_cache_free(UenrKVCache *c);
void       uenr_kv_cache_reset(UenrKVCache *c);
/* Incremental decode for one token at `pos`; writes V logits. */
UenrResult uenr_model_forward_step(UenrModel *m, UenrGemm *g, UenrKVCache *c,
                                   int token, int pos, float *logits_out);

UenrResult uenr_model_init(UenrModel *m, const UenrConfig *cfg, UenrRng *rng);
void       uenr_model_free(UenrModel *m);
UenrResult uenr_model_opt_init(UenrModel *m, int with_adam);
UenrResult uenr_model_ema_init(UenrModel *m);
void       uenr_model_ema_update(UenrModel *m, float decay);
UenrResult uenr_model_quantize_int8(UenrModel *m);
void       uenr_model_free_q8(UenrModel *m);
/* INT8 GEMV: y[o] = sum_i x[i] * q[o*in+i] * scale[o]. */
void uenr_gemv_q8(float *y, const float *x, const int8_t *q,
                  const float *scale, int out, int in);
void uenr_swiglu_fwd_q8_1(float *Y, const float *X,
                          const UenrQMat *Wg, const UenrQMat *Wu,
                          const UenrQMat *Wd, float *gbuf, float *ubuf);
UenrResult uenr_model_lora_init(UenrModel *m, const UenrConfig *cfg, UenrRng *rng);
void       uenr_model_lora_merge(UenrModel *m, UenrGemm *g);

UenrResult uenr_acts_init(UenrActs *a, const UenrConfig *cfg, int T, int S);
void       uenr_acts_free(UenrActs *a);

/* Returns summed cross-entropy over non-ignored targets (0 if targets==NULL).
 * If write_grad, fills a->dlogits (unscaled). *valid_count gets the number of
 * supervised tokens. */
float uenr_model_forward(UenrModel *m, UenrGemm *g, UenrActs *a,
                         const int *tokens, int T, int S,
                         const int *targets, int ignore_index,
                         int write_grad, int *valid_count);

/* Accumulates parameter gradients from a->dlogits. */
void uenr_model_backward(UenrModel *m, UenrGemm *g, UenrActs *a);

void  uenr_model_zero_grad(UenrModel *m);
float uenr_model_grad_norm(UenrModel *m, UenrPool *pool);
float uenr_model_clip_grad(UenrModel *m, float clip, UenrPool *pool);
void  uenr_model_adamw(UenrModel *m, int step, float lr, float b1, float b2,
                       float eps, float weight_decay, UenrPool *pool);

UenrResult uenr_model_save(UenrModel *m, const char *path, int step, int with_opt);
UenrResult uenr_model_load(UenrModel *m, const char *path, int *step, int load_opt);

size_t uenr_model_param_count(const UenrModel *m);
void   uenr_model_report(const UenrModel *m);

#endif /* UENR_MODEL_H */
