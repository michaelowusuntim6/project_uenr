#ifndef UENR_CONFIG_H
#define UENR_CONFIG_H

#include <stddef.h>
#include <stdint.h>
#include "uenr/result.h"

#define UENR_MAX_STR 256
#define UENR_MAX_TARGETS 16

typedef struct {
    char name[UENR_MAX_STR];
    int  seed;
    int  threads;
    int  pin_cpu0;
    int  pin_cpu1;
    int  log_every;
    int  checkpoint_every;
    int  smt_threads;
    char checkpoint_dir[UENR_MAX_STR];
    int  resume;
} UenrRunCfg;

typedef struct {
    char data[UENR_MAX_STR];
    char train_bin[UENR_MAX_STR];
    char val_bin[UENR_MAX_STR];
    char sft_jsonl[UENR_MAX_STR];
    char pairs_jsonl[UENR_MAX_STR];
    char tokenizer[UENR_MAX_STR];
    char vocab_file[UENR_MAX_STR];
    char merges_file[UENR_MAX_STR];
    char model_out[UENR_MAX_STR];
    char ema_out[UENR_MAX_STR];
    char log_csv[UENR_MAX_STR];
    char sft_out[UENR_MAX_STR];
    char lora_out[UENR_MAX_STR];
    char dpo_out[UENR_MAX_STR];
    char ppo_out[UENR_MAX_STR];
    char grpo_out[UENR_MAX_STR];
} UenrPathsCfg;

typedef struct {
    int64_t clock_hz;
    int  avx;
    int  fma;
    int  avx2;
    int  fp16;
    int  align;
    int  micro_m;
    int  micro_n;
    int  l1_block;
    int  l2_block;
    int  huge_pages;
} UenrHwCfg;

typedef struct {
    int   vocab_size;
    int   hidden;
    int   layers;
    int   heads;
    int   kv_heads;
    int   head_dim;
    int   ffn;
    float rms_eps;
    float rope_theta;
    float rope_scale;
    int   rope_yarn;
    float rope_beta_fast;
    float rope_beta_slow;
    float rope_attn_factor;
    float rope_theta_local, rope_partial_local;
    float rope_theta_global, rope_partial_global;
    float attn_scale;
    int   v_norm;
    float attn_softcap;
    float final_softcap;
    float post_norm_init;
    int   kv_shared_layers;
    int   sliding_window;
    char  attn_pattern[32];
    int   tie_embeddings;
    int   max_seq;
    int   flash_attn;
} UenrModelCfg;

typedef struct {
    int   enabled;
    int   num_experts;
    int   experts_per_tok;
    int   expert_ffn;
    int   shared_expert;
    float router_aux_coef;
    float router_jitter;
    float capacity_factor;
} UenrMoeCfg;

typedef struct {
    char kind[UENR_MAX_STR];
    int  lowercase;
    int  normalize_unicode;
    int  strip_control;
    int  train_bytes;
    int  pretokenize;
    int  min_pair_freq;
    int  bos_id;
    int  eos_id;
    int  pad_id;
    int  unk_id;
    int  add_bos;
    int  add_eos;
    int  ignore_index;
    int  special_system;
    int  special_user;
    int  special_assistant;
    int  special_tool_call;
    int  special_tool_result;
    int  special_end;
} UenrTokenizerCfg;

typedef struct {
    int max_bytes;
    int val_percent;
    int sft_stories;
    int sft_prompt_chars;
    int tool_examples;
} UenrPrepareCfg;

typedef struct {
    char  mode[UENR_MAX_STR];
    int   batch_size;
    int   seq_len;
    int   grad_accum;
    int   max_steps;
    float lr;
    float min_lr;
    int   warmup_steps;
    float weight_decay;
    float beta1;
    float beta2;
    float adam_eps;
    float clip;
    float dropout;
    char  dtype[UENR_MAX_STR];
    int   sampled_softmax;
    int   sampled_negatives;
    int   recompute_swiglu;
    float z_loss;
    float ema_decay;
    int   optim8;
} UenrTrainCfg;

typedef struct {
    char  kind[UENR_MAX_STR];
    float tokens_per_param;
    int   min_epochs;
    int   max_epochs;
    float target_loss;
    int   plateau_window;
    float plateau_delta;
    int   min_steps_before_stop;
    float ema_beta;
} UenrScheduleCfg;

typedef struct {
    int   rank;
    float alpha;
    float dropout;
    char  targets[UENR_MAX_TARGETS][32];
    int   n_targets;
} UenrLoraCfg;

typedef struct {
    float beta;
    char  ref_checkpoint[UENR_MAX_STR];
} UenrDpoCfg;

typedef struct {
    float clip_ratio;
    float value_coef;
    float entropy_coef;
    float gamma;
    float lambda;
    int   rollouts;
} UenrPpoCfg;

typedef struct {
    int   group_size;
    float kl_coef;
} UenrGrpoCfg;

typedef struct {
    char  kind[UENR_MAX_STR];
    char  checkpoint[UENR_MAX_STR];
    int   steps;
    float lr;
} UenrRewardCfg;

typedef struct {
    int   max_new_tokens;
    float temperature;
    float reward_format;
    float reward_length;
    int   length_target;
    float reward_repeat;
} UenrRlCfg;

typedef struct {
    int   max_new_tokens;
    char  system[UENR_MAX_STR];
    float temperature;
    int   top_k;
    float top_p;
    int   greedy;
    int   kv_cache;
    int   kv_int8;
    int   weight_int8;
    float repetition_penalty;
    int   repetition_last_n;
    int   no_repeat_ngram;
    float min_p;
    int   json_constrained;
} UenrInferCfg;

typedef struct {
    int gemm_size;
} UenrBenchCfg;

typedef struct {
    char split[32];
    int  batches;
    int  seq_len;
    int  nseq;
} UenrEvalCfg;

typedef struct {
    UenrRunCfg       run;
    UenrPathsCfg     paths;
    UenrHwCfg        hw;
    UenrModelCfg     model;
    UenrMoeCfg       moe;
    UenrTokenizerCfg tok;
    UenrPrepareCfg   prep;
    UenrTrainCfg     train;
    UenrScheduleCfg  sched;
    UenrLoraCfg      lora;
    UenrDpoCfg       dpo;
    UenrPpoCfg       ppo;
    UenrGrpoCfg      grpo;
    UenrRewardCfg    reward;
    UenrRlCfg        rl;
    UenrInferCfg     infer;
    UenrBenchCfg     bench;
    UenrEvalCfg      eval;
} UenrConfig;

/* Loads and validates config. path may be NULL to use UENR_DEFAULT_CONFIG. */
UenrResult uenr_config_load(const char *path, UenrConfig *out);
UenrResult uenr_config_validate(const UenrConfig *c);
void       uenr_config_print(const UenrConfig *c);

/* Derived values (kept here so no code duplicates the arithmetic). */
int uenr_active_params(const UenrConfig *c);
int uenr_layer_is_sliding(const UenrConfig *c, int l);
int uenr_layer_is_shared(const UenrConfig *c, int l);
int uenr_layer_donor(const UenrConfig *c, int l);
int uenr_active_kv_layers(const UenrConfig *c);
int uenr_total_params(const UenrConfig *c);
int uenr_micro_batch(const UenrConfig *c);
int uenr_tokens_per_step(const UenrConfig *c);
int uenr_tokenizer_base(void); /* first mergeable id = 16 specials + 256 bytes */

#define UENR_DEFAULT_CONFIG "config/uenr.toml"
#define UENR_SPECIAL_SLOTS 16
#define UENR_BYTE_BASE     (UENR_SPECIAL_SLOTS)
#define UENR_MERGE_BASE    (UENR_SPECIAL_SLOTS + 256)

#endif /* UENR_CONFIG_H */
