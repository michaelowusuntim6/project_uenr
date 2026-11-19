#include "uenr/config.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Supported configuration syntax (deliberately a strict TOML subset):
 *   - one section header per line: [name]
 *   - one key = value per line
 *   - '#' starts a comment outside double-quoted strings
 *   - strings are double-quoted, support \\ \" \n \t
 *   - integers are decimal and must fit in int/long long
 *   - floats are decimal, finite, and must consume the whole value
 *   - booleans are exactly true or false
 *   - arrays are only supported for [lora].targets: ["a", "b"]
 * Unknown sections/keys, trailing junk, overflow and malformed values are
 * hard errors rather than silently falling back to defaults. */

static char *trim(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = '\0';
    return s;
}

static void strip_comment(char *s) {
    int in_str = 0, esc = 0;
    for (char *p = s; *p; p++) {
        if (in_str && esc) { esc = 0; continue; }
        if (in_str && *p == '\\') { esc = 1; continue; }
        if (*p == '"') in_str = !in_str;
        else if (*p == '#' && !in_str) { *p = '\0'; return; }
    }
}

static int parse_bool_strict(const char *v, int *out) {
    if (!strcmp(v, "true")) { *out = 1; return 1; }
    if (!strcmp(v, "false")) { *out = 0; return 1; }
    return 0;
}

static int parse_int_strict(const char *v, int *out) {
    char *end = NULL;
    errno = 0;
    long x = strtol(v, &end, 10);
    if (end == v || *end != '\0' || errno == ERANGE ||
        x < INT_MIN || x > INT_MAX) return 0;
    *out = (int)x;
    return 1;
}

static int parse_float_strict(const char *v, float *out) {
    char *end = NULL;
    errno = 0;
    float x = strtof(v, &end);
    if (end == v || *end != '\0' || errno == ERANGE || !isfinite(x)) return 0;
    *out = x;
    return 1;
}

/* Strict double-quoted string with a small escape set.  An unterminated or
 * trailing-garbage value, an unescaped control character or an unknown escape
 * is rejected. */
static int parse_string_strict(const char *v, char *dst, size_t n) {
    size_t len = strlen(v);
    if (len < 2 || v[0] != '"' || v[len - 1] != '"') return 0;
    size_t o = 0;
    for (size_t i = 1; i + 1 < len; i++) {
        unsigned char ch = (unsigned char)v[i];
        if (ch == '\\') {
            if (++i >= len - 1) return 0;
            switch (v[i]) {
            case 'n': ch = '\n'; break;
            case 't': ch = '\t'; break;
            case 'r': ch = '\r'; break;
            case '"': ch = '"'; break;
            case '\\': ch = '\\'; break;
            default: return 0;
            }
        } else if (ch < 0x20) {
            return 0;
        } else if (ch == '"') {
            return 0; /* unescaped quote inside the value */
        }
        if (o + 1 >= n) return 0; /* would truncate: treat as malformed */
        dst[o++] = (char)ch;
    }
    dst[o] = '\0';
    return 1;
}

/* Strict array of quoted strings, used by [lora].targets. */
static int parse_quoted_list(const char *v, char *dst, size_t stride,
                             int max, int *n) {
    const char *p = v;
    while (isspace((unsigned char)*p)) p++;
    if (*p != '[') return 0;
    p++;
    int count = 0;
    for (;;) {
        while (isspace((unsigned char)*p)) p++;
        if (*p == ']') { p++; break; }
        if (*p != '"') return 0;
        const char *s = ++p;
        while (*p && *p != '"') {
            if (*p == '\\' || (unsigned char)*p < 0x20) return 0;
            p++;
        }
        if (*p != '"') return 0;
        size_t len = (size_t)(p - s);
        p++;
        if (count >= max || len == 0 || len >= stride) return 0;
        memcpy(dst + (size_t)count * stride, s, len);
        dst[(size_t)count * stride + len] = '\0';
        count++;
        while (isspace((unsigned char)*p)) p++;
        if (*p == ',') { p++; continue; }
        if (*p == ']') { p++; break; }
        return 0;
    }
    while (isspace((unsigned char)*p)) p++;
    if (*p != '\0') return 0;
    *n = count;
    return 1;
}

static int parse_string_array(const char *v, char dst[][32], int max, int *n) {
    return parse_quoted_list(v, &dst[0][0], 32, max, n);
}

/* Strict array of quoted path strings for [datasets.*]. */
static int parse_path_list(const char *v, char dst[][UENR_MAX_STR], int max, int *n) {
    return parse_quoted_list(v, &dst[0][0], UENR_MAX_STR, max, n);
}

static const char *section_name(int sec) {
    switch (sec) {
    case 1: return "run";        case 2: return "paths";
    case 3: return "hardware";   case 4: return "model";
    case 5: return "moe";        case 6: return "tokenizer";
    case 7: return "prepare";    case 8: return "training";
    case 9: return "schedule";   case 10: return "lora";
    case 11: return "dpo";       case 12: return "ppo";
    case 13: return "grpo";      case 14: return "reward";
    case 15: return "rl";        case 16: return "inference";
    case 17: return "bench";     case 18: return "eval";
    case 19: return "datasets.pretrain"; case 20: return "datasets.sft";
    case 21: return "datasets.lora";     case 22: return "datasets.dpo";
    case 23: return "datasets.rl";       case 24: return "datasets.reward";
    case 25: return "datasets.eval";
    default: return "?";
    }
}

typedef enum {
    SEC_NONE = 0, SEC_RUN, SEC_PATHS, SEC_HW, SEC_MODEL, SEC_MOE,
    SEC_TOKENIZER, SEC_PREPARE, SEC_TRAINING, SEC_SCHEDULE, SEC_LORA,
    SEC_DPO, SEC_PPO, SEC_GRPO, SEC_REWARD, SEC_RL, SEC_INFERENCE, SEC_BENCH, SEC_EVAL,
    SEC_DS_PRETRAIN, SEC_DS_SFT, SEC_DS_LORA, SEC_DS_DPO, SEC_DS_RL,
    SEC_DS_REWARD, SEC_DS_EVAL
} Section;

static Section section_from(const char *name) {
    if (!strcmp(name, "run")) return SEC_RUN;
    if (!strcmp(name, "paths")) return SEC_PATHS;
    if (!strcmp(name, "hardware")) return SEC_HW;
    if (!strcmp(name, "model")) return SEC_MODEL;
    if (!strcmp(name, "moe")) return SEC_MOE;
    if (!strcmp(name, "tokenizer")) return SEC_TOKENIZER;
    if (!strcmp(name, "prepare")) return SEC_PREPARE;
    if (!strcmp(name, "training")) return SEC_TRAINING;
    if (!strcmp(name, "schedule")) return SEC_SCHEDULE;
    if (!strcmp(name, "lora")) return SEC_LORA;
    if (!strcmp(name, "dpo")) return SEC_DPO;
    if (!strcmp(name, "ppo")) return SEC_PPO;
    if (!strcmp(name, "grpo")) return SEC_GRPO;
    if (!strcmp(name, "reward")) return SEC_REWARD;
    if (!strcmp(name, "rl")) return SEC_RL;
    if (!strcmp(name, "inference")) return SEC_INFERENCE;
    if (!strcmp(name, "bench")) return SEC_BENCH;
    if (!strcmp(name, "eval")) return SEC_EVAL;
    if (!strcmp(name, "datasets.pretrain")) return SEC_DS_PRETRAIN;
    if (!strcmp(name, "datasets.sft")) return SEC_DS_SFT;
    if (!strcmp(name, "datasets.lora")) return SEC_DS_LORA;
    if (!strcmp(name, "datasets.dpo")) return SEC_DS_DPO;
    if (!strcmp(name, "datasets.rl")) return SEC_DS_RL;
    if (!strcmp(name, "datasets.reward")) return SEC_DS_REWARD;
    if (!strcmp(name, "datasets.eval")) return SEC_DS_EVAL;
    return SEC_NONE;
}

static int set_dataset_kv(UenrDatasetCfg *d, const char *k, const char *v) {
    if (!strcmp(k, "paths"))
        return parse_path_list(v, d->paths, UENR_MAX_DATASET_PATHS, &d->n_paths) ? 0 : -1;
    if (!strcmp(k, "format"))
        return parse_string_strict(v, d->format, sizeof(d->format)) ? 0 : -1;
    if (!strcmp(k, "max_bytes")) {
        int x; if (!parse_int_strict(v, &x)) return -1; d->max_bytes = x; return 0;
    }
    if (!strcmp(k, "shuffle_seed")) {
        int x; if (!parse_int_strict(v, &x)) return -1; d->shuffle_seed = x; return 0;
    }
    if (!strcmp(k, "max_samples")) {
        int x; if (!parse_int_strict(v, &x)) return -1; d->max_samples = x; return 0;
    }
    if (!strcmp(k, "ref_path"))
        return parse_string_strict(v, d->ref_path, sizeof(d->ref_path)) ? 0 : -1;
    if (!strcmp(k, "release_tag"))
        return parse_string_strict(v, d->release_tag, sizeof(d->release_tag)) ? 0 : -1;
    if (!strcmp(k, "release_asset"))
        return parse_string_strict(v, d->release_asset, sizeof(d->release_asset)) ? 0 : -1;
    if (!strcmp(k, "url"))
        return parse_string_strict(v, d->url, sizeof(d->url)) ? 0 : -1;
    if (!strcmp(k, "derive_from"))
        return parse_string_strict(v, d->derive_from, sizeof(d->derive_from)) ? 0 : -1;
    if (!strcmp(k, "convert_from"))
        return parse_string_strict(v, d->convert_from, sizeof(d->convert_from)) ? 0 : -1;
    if (!strcmp(k, "convert_to"))
        return parse_string_strict(v, d->convert_to, sizeof(d->convert_to)) ? 0 : -1;
    if (!strcmp(k, "limit")) {
        int x; if (!parse_int_strict(v, &x)) return -1; d->limit = x; return 0;
    }
    if (!strcmp(k, "bootstrap")) {
        int x; if (!parse_bool_strict(v, &x)) return -1; d->bootstrap = x; return 0;
    }
    return 1;
}

/* set_kv returns 0 on success, 1 for an unknown key and -1 for a malformed or
 * out-of-type value.  The caller reports the section and key. */
#define AS_INT(f) do { int _x; if (!parse_int_strict(v, &_x)) return -1; c->f = _x; } while (0)
#define AS_F(f)   do { float _x; if (!parse_float_strict(v, &_x)) return -1; c->f = _x; } while (0)
#define AS_B(f)   do { int _x; if (!parse_bool_strict(v, &_x)) return -1; c->f = _x; } while (0)
#define AS_S(f)   do { if (!parse_string_strict(v, c->f, sizeof(c->f))) return -1; } while (0)

static int set_kv(UenrConfig *c, Section sec, const char *k, const char *v) {
    switch (sec) {
    case SEC_RUN:
        if (!strcmp(k, "name")) AS_S(run.name);
        else if (!strcmp(k, "seed")) AS_INT(run.seed);
        else if (!strcmp(k, "threads")) AS_INT(run.threads);
        else if (!strcmp(k, "pin_cpu0")) AS_INT(run.pin_cpu0);
        else if (!strcmp(k, "pin_cpu1")) AS_INT(run.pin_cpu1);
        else if (!strcmp(k, "log_every")) AS_INT(run.log_every);
        else if (!strcmp(k, "checkpoint_every")) AS_INT(run.checkpoint_every);
        else if (!strcmp(k, "smt_threads")) AS_INT(run.smt_threads);
        else if (!strcmp(k, "simd")) AS_S(run.simd);
        else if (!strcmp(k, "checkpoint_dir")) AS_S(run.checkpoint_dir);
        else if (!strcmp(k, "resume")) AS_B(run.resume);
        else return 1;
        break;
    case SEC_PATHS:
        if (!strcmp(k, "data")) AS_S(paths.data);
        else if (!strcmp(k, "train_bin")) AS_S(paths.train_bin);
        else if (!strcmp(k, "val_bin")) AS_S(paths.val_bin);
        else if (!strcmp(k, "sft_jsonl")) AS_S(paths.sft_jsonl);
        else if (!strcmp(k, "lora_jsonl")) AS_S(paths.lora_jsonl);
        else if (!strcmp(k, "pairs_jsonl")) AS_S(paths.pairs_jsonl);
        else if (!strcmp(k, "rl_jsonl")) AS_S(paths.rl_jsonl);
        else if (!strcmp(k, "reward_jsonl")) AS_S(paths.reward_jsonl);
        else if (!strcmp(k, "tokenizer")) AS_S(paths.tokenizer);
        else if (!strcmp(k, "vocab_file")) AS_S(paths.vocab_file);
        else if (!strcmp(k, "merges_file")) AS_S(paths.merges_file);
        else if (!strcmp(k, "model_out")) AS_S(paths.model_out);
        else if (!strcmp(k, "ema_out")) AS_S(paths.ema_out);
        else if (!strcmp(k, "log_csv")) AS_S(paths.log_csv);
        else if (!strcmp(k, "sft_out")) AS_S(paths.sft_out);
        else if (!strcmp(k, "lora_out")) AS_S(paths.lora_out);
        else if (!strcmp(k, "dpo_out")) AS_S(paths.dpo_out);
        else if (!strcmp(k, "ppo_out")) AS_S(paths.ppo_out);
        else if (!strcmp(k, "grpo_out")) AS_S(paths.grpo_out);
        else return 1;
        break;
    case SEC_HW:
        if (!strcmp(k, "huge_pages")) AS_B(hw.huge_pages);
        else return 1;
        break;
    case SEC_MODEL:
        if (!strcmp(k, "vocab_size")) AS_INT(model.vocab_size);
        else if (!strcmp(k, "hidden")) AS_INT(model.hidden);
        else if (!strcmp(k, "layers")) AS_INT(model.layers);
        else if (!strcmp(k, "heads")) AS_INT(model.heads);
        else if (!strcmp(k, "kv_heads")) AS_INT(model.kv_heads);
        else if (!strcmp(k, "head_dim")) AS_INT(model.head_dim);
        else if (!strcmp(k, "ffn")) AS_INT(model.ffn);
        else if (!strcmp(k, "rms_eps")) AS_F(model.rms_eps);
        else if (!strcmp(k, "rope_theta")) AS_F(model.rope_theta);
        else if (!strcmp(k, "rope_scale")) AS_F(model.rope_scale);
        else if (!strcmp(k, "rope_yarn")) AS_B(model.rope_yarn);
        else if (!strcmp(k, "rope_beta_fast")) AS_F(model.rope_beta_fast);
        else if (!strcmp(k, "rope_beta_slow")) AS_F(model.rope_beta_slow);
        else if (!strcmp(k, "rope_attn_factor")) AS_F(model.rope_attn_factor);
        else if (!strcmp(k, "rope_theta_local")) AS_F(model.rope_theta_local);
        else if (!strcmp(k, "rope_partial_local")) AS_F(model.rope_partial_local);
        else if (!strcmp(k, "rope_theta_global")) AS_F(model.rope_theta_global);
        else if (!strcmp(k, "rope_partial_global")) AS_F(model.rope_partial_global);
        else if (!strcmp(k, "attn_scale")) AS_F(model.attn_scale);
        else if (!strcmp(k, "v_norm")) AS_B(model.v_norm);
        else if (!strcmp(k, "kv_adapter")) AS_B(model.kv_adapter);
        else if (!strcmp(k, "attn_softcap")) AS_F(model.attn_softcap);
        else if (!strcmp(k, "final_softcap")) AS_F(model.final_softcap);
        else if (!strcmp(k, "post_norm_init")) AS_F(model.post_norm_init);
        else if (!strcmp(k, "kv_shared_layers")) AS_INT(model.kv_shared_layers);
        else if (!strcmp(k, "sliding_window")) AS_INT(model.sliding_window);
        else if (!strcmp(k, "attn_pattern")) AS_S(model.attn_pattern);
        else if (!strcmp(k, "tie_embeddings")) AS_B(model.tie_embeddings);
        else if (!strcmp(k, "max_seq")) AS_INT(model.max_seq);
        else if (!strcmp(k, "flash_attn")) AS_B(model.flash_attn);
        else return 1;
        break;
    case SEC_MOE:
        if (!strcmp(k, "enabled")) AS_B(moe.enabled);
        else if (!strcmp(k, "num_experts")) AS_INT(moe.num_experts);
        else if (!strcmp(k, "experts_per_tok")) AS_INT(moe.experts_per_tok);
        else if (!strcmp(k, "expert_ffn")) AS_INT(moe.expert_ffn);
        else if (!strcmp(k, "shared_expert")) AS_B(moe.shared_expert);
        else if (!strcmp(k, "router_aux_coef")) AS_F(moe.router_aux_coef);
        else return 1;
        break;
    case SEC_TOKENIZER:
        if (!strcmp(k, "kind")) AS_S(tok.kind);
        else if (!strcmp(k, "lowercase")) AS_B(tok.lowercase);
        else if (!strcmp(k, "normalize_unicode")) AS_B(tok.normalize_unicode);
        else if (!strcmp(k, "strip_control")) AS_B(tok.strip_control);
        else if (!strcmp(k, "train_bytes")) AS_INT(tok.train_bytes);
        else if (!strcmp(k, "pretokenize")) AS_B(tok.pretokenize);
        else if (!strcmp(k, "min_pair_freq")) AS_INT(tok.min_pair_freq);
        else if (!strcmp(k, "bos_id")) AS_INT(tok.bos_id);
        else if (!strcmp(k, "eos_id")) AS_INT(tok.eos_id);
        else if (!strcmp(k, "pad_id")) AS_INT(tok.pad_id);
        else if (!strcmp(k, "unk_id")) AS_INT(tok.unk_id);
        else if (!strcmp(k, "add_bos")) AS_B(tok.add_bos);
        else if (!strcmp(k, "add_eos")) AS_B(tok.add_eos);
        else if (!strcmp(k, "ignore_index")) AS_INT(tok.ignore_index);
        else if (!strcmp(k, "special_system")) AS_INT(tok.special_system);
        else if (!strcmp(k, "special_user")) AS_INT(tok.special_user);
        else if (!strcmp(k, "special_assistant")) AS_INT(tok.special_assistant);
        else if (!strcmp(k, "special_tool_call")) AS_INT(tok.special_tool_call);
        else if (!strcmp(k, "special_tool_result")) AS_INT(tok.special_tool_result);
        else if (!strcmp(k, "special_end")) AS_INT(tok.special_end);
        else return 1;
        break;
    case SEC_PREPARE:
        if (!strcmp(k, "max_bytes")) AS_INT(prep.max_bytes);
        else if (!strcmp(k, "val_percent")) AS_INT(prep.val_percent);
        else if (!strcmp(k, "dedup")) AS_B(prep.dedup);
        else if (!strcmp(k, "dedup_slots")) AS_INT(prep.dedup_slots);
        else if (!strcmp(k, "sft_stories")) AS_INT(prep.sft_stories);
        else if (!strcmp(k, "sft_prompt_chars")) AS_INT(prep.sft_prompt_chars);
        else if (!strcmp(k, "tool_examples")) AS_INT(prep.tool_examples);
        else return 1;
        break;
    case SEC_TRAINING:
        if (!strcmp(k, "batch_size")) AS_INT(train.batch_size);
        else if (!strcmp(k, "seq_len")) AS_INT(train.seq_len);
        else if (!strcmp(k, "grad_accum")) AS_INT(train.grad_accum);
        else if (!strcmp(k, "max_steps")) AS_INT(train.max_steps);
        else if (!strcmp(k, "lr")) AS_F(train.lr);
        else if (!strcmp(k, "min_lr")) AS_F(train.min_lr);
        else if (!strcmp(k, "warmup_steps")) AS_INT(train.warmup_steps);
        else if (!strcmp(k, "weight_decay")) AS_F(train.weight_decay);
        else if (!strcmp(k, "beta1")) AS_F(train.beta1);
        else if (!strcmp(k, "beta2")) AS_F(train.beta2);
        else if (!strcmp(k, "adam_eps")) AS_F(train.adam_eps);
        else if (!strcmp(k, "clip")) AS_F(train.clip);
        else if (!strcmp(k, "sampled_softmax")) AS_B(train.sampled_softmax);
        else if (!strcmp(k, "sampled_negatives")) AS_INT(train.sampled_negatives);
        else if (!strcmp(k, "recompute_swiglu")) AS_B(train.recompute_swiglu);
        else if (!strcmp(k, "z_loss")) AS_F(train.z_loss);
        else if (!strcmp(k, "ema_decay")) AS_F(train.ema_decay);
        else if (!strcmp(k, "optim8")) AS_B(train.optim8);
        else if (!strcmp(k, "recompute_attn")) AS_B(train.recompute_attn);
        else if (!strcmp(k, "block_checkpoint")) AS_B(train.block_checkpoint);
        else if (!strcmp(k, "fp16_activations")) AS_B(train.fp16_activations);
        else if (!strcmp(k, "kv_qat_bits")) AS_INT(train.kv_qat_bits);
        else if (!strcmp(k, "kv_lowrank")) AS_INT(train.kv_lowrank);
        else if (!strcmp(k, "adafactor")) AS_B(train.adafactor);
        else if (!strcmp(k, "seq_ramp_steps")) AS_INT(train.seq_ramp_steps);
        else if (!strcmp(k, "muon")) AS_B(train.muon);
        else if (!strcmp(k, "muon_ns_iters")) AS_INT(train.muon_ns_iters);
        else if (!strcmp(k, "sft_pack")) AS_B(train.sft_pack);
        else if (!strcmp(k, "weight_qat_bits")) AS_INT(train.weight_qat_bits);
        else if (!strcmp(k, "swiglu_clamp")) AS_F(train.swiglu_clamp);
        else if (!strcmp(k, "qlora")) AS_B(train.qlora);
        else return 1;
        break;
    case SEC_SCHEDULE:
        if (!strcmp(k, "kind")) AS_S(sched.kind);
        else if (!strcmp(k, "tokens_per_param")) AS_F(sched.tokens_per_param);
        else if (!strcmp(k, "min_epochs")) AS_INT(sched.min_epochs);
        else if (!strcmp(k, "max_epochs")) AS_INT(sched.max_epochs);
        else if (!strcmp(k, "target_loss")) AS_F(sched.target_loss);
        else if (!strcmp(k, "plateau_window")) AS_INT(sched.plateau_window);
        else if (!strcmp(k, "plateau_delta")) AS_F(sched.plateau_delta);
        else if (!strcmp(k, "min_steps_before_stop")) AS_INT(sched.min_steps_before_stop);
        else if (!strcmp(k, "ema_beta")) AS_F(sched.ema_beta);
        else return 1;
        break;
    case SEC_LORA:
        if (!strcmp(k, "rank")) AS_INT(lora.rank);
        else if (!strcmp(k, "alpha")) AS_F(lora.alpha);
        else if (!strcmp(k, "targets")) {
            if (!parse_string_array(v, c->lora.targets, UENR_MAX_TARGETS,
                                    &c->lora.n_targets)) return -1;
        }
        else return 1;
        break;
    case SEC_DPO:
        if (!strcmp(k, "beta")) AS_F(dpo.beta);
        else if (!strcmp(k, "ref_checkpoint")) AS_S(dpo.ref_checkpoint);
        else return 1;
        break;
    case SEC_PPO:
        if (!strcmp(k, "clip_ratio")) AS_F(ppo.clip_ratio);
        else if (!strcmp(k, "kl_coef")) AS_F(ppo.kl_coef);
        else if (!strcmp(k, "rollouts")) AS_INT(ppo.rollouts);
        else return 1;
        break;
    case SEC_GRPO:
        if (!strcmp(k, "group_size")) AS_INT(grpo.group_size);
        else if (!strcmp(k, "kl_coef")) AS_F(grpo.kl_coef);
        else if (!strcmp(k, "ref_refresh")) AS_INT(grpo.ref_refresh);
        else return 1;
        break;
    case SEC_REWARD:
        if (!strcmp(k, "kind")) AS_S(reward.kind);
        else if (!strcmp(k, "checkpoint")) AS_S(reward.checkpoint);
        else if (!strcmp(k, "steps")) AS_INT(reward.steps);
        else if (!strcmp(k, "lr")) AS_F(reward.lr);
        else return 1;
        break;
    case SEC_RL:
        if (!strcmp(k, "max_new_tokens")) AS_INT(rl.max_new_tokens);
        else if (!strcmp(k, "temperature")) AS_F(rl.temperature);
        else if (!strcmp(k, "reward_format")) AS_F(rl.reward_format);
        else if (!strcmp(k, "reward_length")) AS_F(rl.reward_length);
        else if (!strcmp(k, "length_target")) AS_INT(rl.length_target);
        else if (!strcmp(k, "reward_repeat")) AS_F(rl.reward_repeat);
        else if (!strcmp(k, "reward_verify")) AS_F(rl.reward_verify);
        else return 1;
        break;
    case SEC_INFERENCE:
        if (!strcmp(k, "max_new_tokens")) AS_INT(infer.max_new_tokens);
        else if (!strcmp(k, "system")) AS_S(infer.system);
        else if (!strcmp(k, "temperature")) AS_F(infer.temperature);
        else if (!strcmp(k, "top_k")) AS_INT(infer.top_k);
        else if (!strcmp(k, "top_p")) AS_F(infer.top_p);
        else if (!strcmp(k, "greedy")) AS_B(infer.greedy);
        else if (!strcmp(k, "kv_cache")) AS_B(infer.kv_cache);
        else if (!strcmp(k, "kv_int8")) AS_B(infer.kv_int8);
        else if (!strcmp(k, "weight_int8")) AS_B(infer.weight_int8);
        else if (!strcmp(k, "weight_int4")) AS_B(infer.weight_int4);
        else if (!strcmp(k, "kv_fp16")) AS_B(infer.kv_fp16);
        else if (!strcmp(k, "kv_q4")) AS_B(infer.kv_q4);
        else if (!strcmp(k, "kv_rotate")) AS_B(infer.kv_rotate);
        else if (!strcmp(k, "kv_q2")) AS_B(infer.kv_q2);
        else if (!strcmp(k, "kv_recent")) AS_INT(infer.kv_recent);
        else if (!strcmp(k, "kv_sinks")) AS_INT(infer.kv_sinks);
        else if (!strcmp(k, "kv_evict")) AS_B(infer.kv_evict);
        else if (!strcmp(k, "kv_evict_window")) AS_INT(infer.kv_evict_window);
        else if (!strcmp(k, "kv_sparse_k")) AS_INT(infer.kv_sparse_k);
        else if (!strcmp(k, "kv_lowrank")) AS_INT(infer.kv_lowrank);
        else if (!strcmp(k, "early_exit")) AS_B(infer.early_exit);
        else if (!strcmp(k, "early_exit_delta")) AS_F(infer.early_exit_delta);
        else if (!strcmp(k, "kv_offload")) AS_B(infer.kv_offload);
        else if (!strcmp(k, "prefix_cache")) AS_B(infer.prefix_cache);
        else if (!strcmp(k, "draft_model")) AS_S(infer.draft_model);
        else if (!strcmp(k, "draft_config")) AS_S(infer.draft_config);
        else if (!strcmp(k, "draft_k")) AS_INT(infer.draft_k);
        else if (!strcmp(k, "kv_offload_file")) AS_S(infer.kv_offload_file);
        else if (!strcmp(k, "repetition_penalty")) AS_F(infer.repetition_penalty);
        else if (!strcmp(k, "repetition_last_n")) AS_INT(infer.repetition_last_n);
        else if (!strcmp(k, "no_repeat_ngram")) AS_INT(infer.no_repeat_ngram);
        else if (!strcmp(k, "min_p")) AS_F(infer.min_p);
        else if (!strcmp(k, "json_constrained")) AS_B(infer.json_constrained);
        else if (!strcmp(k, "speculative")) AS_B(infer.speculative);
        else if (!strcmp(k, "spec_k")) AS_INT(infer.spec_k);
        else if (!strcmp(k, "spec_ngram")) AS_INT(infer.spec_ngram);
        else if (!strcmp(k, "batch_max")) AS_INT(infer.batch_max);
        else if (!strcmp(k, "continuous")) AS_B(infer.continuous);
        else return 1;
        break;
    case SEC_BENCH:
        if (!strcmp(k, "gemm_size")) AS_INT(bench.gemm_size);
        else return 1;
        break;
    case SEC_EVAL:
        if (!strcmp(k, "split")) AS_S(eval.split);
        else if (!strcmp(k, "batches")) AS_INT(eval.batches);
        else if (!strcmp(k, "seq_len")) AS_INT(eval.seq_len);
        else if (!strcmp(k, "nseq")) AS_INT(eval.nseq);
        else if (!strcmp(k, "gen_tokens")) AS_INT(eval.gen_tokens);
        else if (!strcmp(k, "prompt")) AS_S(eval.prompt);
        else return 1;
        break;
    case SEC_DS_PRETRAIN: return set_dataset_kv(&c->datasets.pretrain, k, v);
    case SEC_DS_SFT:      return set_dataset_kv(&c->datasets.sft, k, v);
    case SEC_DS_LORA:     return set_dataset_kv(&c->datasets.lora, k, v);
    case SEC_DS_DPO:      return set_dataset_kv(&c->datasets.dpo, k, v);
    case SEC_DS_RL:       return set_dataset_kv(&c->datasets.rl, k, v);
    case SEC_DS_REWARD:   return set_dataset_kv(&c->datasets.reward, k, v);
    case SEC_DS_EVAL:     return set_dataset_kv(&c->datasets.eval, k, v);
    case SEC_NONE:
    default:
        return 1;
    }
    return 0;
}

UenrResult uenr_config_load(const char *path, UenrConfig *out) {
    if (out == NULL) return UENR_ERR_INVALID;
    memset(out, 0, sizeof(*out));
    const char *p = (path == NULL) ? UENR_DEFAULT_CONFIG : path;
    FILE *f = fopen(p, "r");
    if (f == NULL) return UENR_ERR_IO;

    char line[1024];
    Section sec = SEC_NONE;
    int lineno = 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        lineno++;
        size_t ll = strlen(line);
        if (ll == sizeof(line) - 1 && line[ll - 1] != '\n' && !feof(f)) {
            fprintf(stderr, "config: %s:%d: line too long\n", p, lineno);
            fclose(f);
            return UENR_ERR_INVALID;
        }
        strip_comment(line);
        char *s = trim(line);
        if (*s == '\0') continue;
        if (*s == '[') {
            char *end = strchr(s, ']');
            if (end == NULL || *trim(end + 1) != '\0') {
                fprintf(stderr, "config: %s:%d: malformed section header\n", p, lineno);
                fclose(f);
                return UENR_ERR_INVALID;
            }
            *end = '\0';
            const char *name = trim(s + 1);
            sec = section_from(name);
            if (sec == SEC_NONE) {
                fprintf(stderr, "config: %s:%d: unknown section [%s]\n", p, lineno, name);
                fclose(f);
                return UENR_ERR_INVALID;
            }
            continue;
        }
        char *eq = strchr(s, '=');
        if (eq == NULL) {
            fprintf(stderr, "config: %s:%d: expected key = value\n", p, lineno);
            fclose(f);
            return UENR_ERR_INVALID;
        }
        *eq = '\0';
        char *key = trim(s);
        char *val = trim(eq + 1);
        if (*key == '\0') {
            fprintf(stderr, "config: %s:%d: empty key\n", p, lineno);
            fclose(f);
            return UENR_ERR_INVALID;
        }
        if (sec == SEC_NONE) {
            fprintf(stderr, "config: %s:%d: key '%s' outside a section\n", p, lineno, key);
            fclose(f);
            return UENR_ERR_INVALID;
        }
        int rc = set_kv(out, sec, key, val);
        if (rc != 0) {
            if (rc > 0)
                fprintf(stderr, "config: %s:%d: unknown key '%s' in [%s]\n",
                        p, lineno, key, section_name(sec));
            else
                fprintf(stderr, "config: %s:%d: invalid value '%s' for [%s].%s\n",
                        p, lineno, val, section_name(sec), key);
            fclose(f);
            return UENR_ERR_INVALID;
        }
    }
    fclose(f);
    /* Backward compatibility: the old flat [paths].data + [prepare].max_bytes
     * describe the pretrain phase.  New [datasets.pretrain] keys win. */
    if (out->datasets.pretrain.n_paths == 0 && out->paths.data[0]) {
        snprintf(out->datasets.pretrain.paths[0], UENR_MAX_STR, "%s", out->paths.data);
        out->datasets.pretrain.n_paths = 1;
        if (!out->datasets.pretrain.format[0])
            snprintf(out->datasets.pretrain.format, sizeof(out->datasets.pretrain.format),
                     "text_lines");
        if (out->datasets.pretrain.max_bytes == 0)
            out->datasets.pretrain.max_bytes = out->prep.max_bytes;
        fprintf(stderr, "config: note: using legacy [paths].data/[prepare].max_bytes "
                        "for [datasets.pretrain]\n");
    }
    if (out->datasets.dpo.ref_path[0])
        snprintf(out->dpo.ref_checkpoint, sizeof(out->dpo.ref_checkpoint), "%s",
                 out->datasets.dpo.ref_path);
    /* Resolve prepared-output fallbacks once so trainers do not repeat them. */
    if (!out->paths.lora_jsonl[0])
        snprintf(out->paths.lora_jsonl, sizeof(out->paths.lora_jsonl), "%s",
                 out->paths.sft_jsonl);
    if (!out->paths.rl_jsonl[0])
        snprintf(out->paths.rl_jsonl, sizeof(out->paths.rl_jsonl), "%s",
                 out->paths.pairs_jsonl);
    if (!out->paths.reward_jsonl[0])
        snprintf(out->paths.reward_jsonl, sizeof(out->paths.reward_jsonl), "%s",
                 out->paths.pairs_jsonl);
    UenrResult vr = uenr_config_validate(out);
    if (vr != UENR_OK)
        fprintf(stderr, "config: %s: semantic validation failed; "
                        "check ranges in docs/CONFIG.md\n", p);
    return vr;
}

static int unique_specials(const UenrTokenizerCfg *t) {
    int ids[8] = { t->bos_id, t->eos_id, t->pad_id, t->unk_id,
                   t->special_system, t->special_user, t->special_assistant,
                   t->special_end };
    for (int i = 0; i < 8; i++) {
        if (ids[i] < 0 || ids[i] >= UENR_SPECIAL_SLOTS) return 0;
        for (int j = i + 1; j < 8; j++)
            if (ids[i] == ids[j]) return 0;
    }
    if (t->special_tool_call < 0 || t->special_tool_call >= UENR_SPECIAL_SLOTS) return 0;
    if (t->special_tool_result < 0 || t->special_tool_result >= UENR_SPECIAL_SLOTS) return 0;
    if (t->special_tool_call == t->special_tool_result) return 0;
    return 1;
}

UenrResult uenr_config_validate(const UenrConfig *c) {
    if (c == NULL) return UENR_ERR_INVALID;
    if (c->model.vocab_size < UENR_MERGE_BASE || c->model.vocab_size > 65535)
        return UENR_ERR_INVALID;
    if (c->model.hidden <= 0 || c->model.layers <= 0) return UENR_ERR_INVALID;
    if (c->model.heads <= 0 || c->model.kv_heads <= 0 || c->model.head_dim <= 0)
        return UENR_ERR_INVALID;
    if (c->model.heads % c->model.kv_heads != 0) return UENR_ERR_INVALID;
    if (c->model.heads * c->model.head_dim != c->model.hidden) return UENR_ERR_INVALID;
    if (c->model.head_dim % 2 != 0) return UENR_ERR_INVALID; /* RoPE pairs */
    if (c->model.ffn <= 0) return UENR_ERR_INVALID;
    if (c->model.max_seq <= 0 || c->model.max_seq > 1048576) return UENR_ERR_INVALID;
    if (c->model.rms_eps <= 0.0f || c->model.rope_theta <= 0.0f ||
        c->model.rope_scale <= 0.0f) return UENR_ERR_INVALID;
    if (c->model.rope_theta_local <= 0.0f || c->model.rope_theta_global <= 0.0f)
        return UENR_ERR_INVALID;
    if (c->model.rope_partial_local < 0.0f || c->model.rope_partial_local > 1.0f ||
        c->model.rope_partial_global < 0.0f || c->model.rope_partial_global > 1.0f)
        return UENR_ERR_INVALID;
    if (c->model.attn_scale < 0.0f || c->model.attn_softcap < 0.0f ||
        c->model.final_softcap < 0.0f) return UENR_ERR_INVALID;
    if (c->model.post_norm_init < 0.0f || c->model.post_norm_init > 1.0f)
        return UENR_ERR_INVALID;
    if (c->model.kv_shared_layers < 0 || c->model.kv_shared_layers >= c->model.layers)
        return UENR_ERR_INVALID;
    if (c->model.sliding_window < 0 || c->model.sliding_window > c->model.max_seq)
        return UENR_ERR_INVALID;
    /* Every shared layer borrows K/V storage from a donor of the same
     * attention type: sliding borrows a windowed slot, global borrows a full
     * (possibly evicted) slot.  Without a same-type donor the borrower would
     * index past the donor's shorter cache. */
    {
        int boundary = c->model.layers - c->model.kv_shared_layers;
        for (int l = boundary; l < c->model.layers; l++) {
            int want = uenr_layer_is_sliding(c, l);
            int found = 0;
            for (int j = 0; j < boundary; j++)
                if (uenr_layer_is_sliding(c, j) == want) found = 1;
            if (!found) return UENR_ERR_INVALID;
        }
    }
    if (c->train.seq_len <= 0 || c->train.seq_len > c->model.max_seq)
        return UENR_ERR_INVALID;
    if (c->run.threads <= 0 || c->run.threads > 64) return UENR_ERR_INVALID;
    if (c->run.smt_threads < 1 || c->run.smt_threads > 64) return UENR_ERR_INVALID;
    if (c->run.simd[0] && strcmp(c->run.simd, "auto") &&
        strcmp(c->run.simd, "avx1") && strcmp(c->run.simd, "avx2") && strcmp(c->run.simd, "avx512"))
        return UENR_ERR_INVALID;
    if (c->moe.enabled) {
        if (c->moe.num_experts <= 0) return UENR_ERR_INVALID;
        if (c->moe.experts_per_tok <= 0 || c->moe.experts_per_tok > c->moe.num_experts)
            return UENR_ERR_INVALID;
        if (c->moe.expert_ffn <= 0) return UENR_ERR_INVALID;
    }
    if (c->train.batch_size <= 0 || c->train.grad_accum <= 0) return UENR_ERR_INVALID;
    if (c->train.batch_size % c->train.grad_accum != 0) return UENR_ERR_INVALID;
    if (c->train.lr <= 0.0f) return UENR_ERR_INVALID;
    if (c->train.swiglu_clamp < 0.0f || c->train.swiglu_clamp > 100.0f)
        return UENR_ERR_INVALID;
    if (c->train.min_lr < 0.0f || c->train.min_lr > c->train.lr) return UENR_ERR_INVALID;
    if (c->train.beta1 < 0.0f || c->train.beta1 >= 1.0f ||
        c->train.beta2 < 0.0f || c->train.beta2 >= 1.0f) return UENR_ERR_INVALID;
    if (c->train.adam_eps <= 0.0f || c->train.clip <= 0.0f) return UENR_ERR_INVALID;
    if (c->train.weight_decay < 0.0f) return UENR_ERR_INVALID;
    if (c->train.warmup_steps < 0 || c->train.max_steps < 0) return UENR_ERR_INVALID;
    if (c->train.z_loss < 0.0f) return UENR_ERR_INVALID;
    if (c->grpo.ref_refresh < 0 || c->rl.reward_verify < 0.0f ||
        c->rl.reward_verify > 10.0f)
        return UENR_ERR_INVALID;
    if (c->train.sft_pack != 0 && c->train.sft_pack != 1) return UENR_ERR_INVALID;
    {
        const UenrDatasetCfg *ds[] = { &c->datasets.pretrain, &c->datasets.sft,
                                       &c->datasets.lora, &c->datasets.dpo,
                                       &c->datasets.rl, &c->datasets.reward,
                                       &c->datasets.eval };
        for (size_t i = 0; i < sizeof(ds) / sizeof(ds[0]); i++) {
            if (ds[i]->limit < 0) return UENR_ERR_INVALID;
            if (ds[i]->bootstrap != 0 && ds[i]->bootstrap != 1)
                return UENR_ERR_INVALID;
            if (ds[i]->derive_from[0]) {
                const char *df = ds[i]->derive_from;
                if (strcmp(df, "pretrain") && strcmp(df, "sft") &&
                    strcmp(df, "lora") && strcmp(df, "dpo") &&
                    strcmp(df, "rl") && strcmp(df, "reward"))
                    return UENR_ERR_INVALID;
            }
        }
    }
    if (c->train.muon_ns_iters < 0 || c->train.muon_ns_iters > 10)
        return UENR_ERR_INVALID;
    if (c->train.kv_qat_bits != 0 && c->train.kv_qat_bits != 2 &&
        c->train.kv_qat_bits != 4) return UENR_ERR_INVALID;
    if (c->train.kv_lowrank < 0 || c->train.kv_lowrank > 1024) return UENR_ERR_INVALID;
    if (c->train.seq_ramp_steps < 0) return UENR_ERR_INVALID;
    if (c->train.weight_qat_bits != 0 && c->train.weight_qat_bits != 4 &&
        c->train.weight_qat_bits != 8) return UENR_ERR_INVALID;
    if (c->train.ema_decay < 0.0f || c->train.ema_decay >= 1.0f)
        return UENR_ERR_INVALID;
    if (c->train.sampled_softmax &&
        (c->train.sampled_negatives < 1 ||
         c->train.sampled_negatives >= c->model.vocab_size))
        return UENR_ERR_INVALID;
    if (c->prep.max_bytes <= 0) return UENR_ERR_INVALID;
    if (c->datasets.pretrain.n_paths < 1) return UENR_ERR_INVALID;
    if (c->datasets.pretrain.max_bytes <= 0) return UENR_ERR_INVALID;
    if (strcmp(c->datasets.pretrain.format, "text_lines")) return UENR_ERR_INVALID;
    if (c->datasets.sft.n_paths > 0 &&
        strcmp(c->datasets.sft.format, "jsonl_prompt_completion") &&
        strcmp(c->datasets.sft.format, "jsonl_messages")) return UENR_ERR_INVALID;
    if (c->datasets.lora.n_paths > 0 &&
        strcmp(c->datasets.lora.format, "inherit") &&
        strcmp(c->datasets.lora.format, "jsonl_prompt_completion") &&
        strcmp(c->datasets.lora.format, "jsonl_messages")) return UENR_ERR_INVALID;
    if (c->datasets.dpo.n_paths > 0 &&
        strcmp(c->datasets.dpo.format, "jsonl_prompt_chosen_rejected"))
        return UENR_ERR_INVALID;
    if (c->datasets.rl.n_paths > 0 &&
        strcmp(c->datasets.rl.format, "jsonl_prompt_only")) return UENR_ERR_INVALID;
    if (c->datasets.reward.n_paths > 0 &&
        strcmp(c->datasets.reward.format, "jsonl_prompt_response_label"))
        return UENR_ERR_INVALID;
    if (c->datasets.eval.n_paths > 0 && strcmp(c->datasets.eval.format, "bin"))
        return UENR_ERR_INVALID;
    if (c->datasets.sft.max_samples < 0 || c->datasets.dpo.max_samples < 0 ||
        c->datasets.rl.max_samples < 0 || c->datasets.reward.max_samples < 0)
        return UENR_ERR_INVALID;
    if (c->datasets.sft.shuffle_seed < 0 || c->datasets.dpo.shuffle_seed < 0 ||
        c->datasets.rl.shuffle_seed < 0 || c->datasets.reward.shuffle_seed < 0)
        return UENR_ERR_INVALID;
    if (c->prep.val_percent < 0 || c->prep.val_percent > 50) return UENR_ERR_INVALID;
    if (c->prep.dedup_slots < 0) return UENR_ERR_INVALID;
    if (c->prep.sft_stories < 0) return UENR_ERR_INVALID;
    if (c->prep.tool_examples < 0) return UENR_ERR_INVALID;
    if (c->tok.train_bytes <= 0) return UENR_ERR_INVALID;
    if (c->tok.min_pair_freq < 1) return UENR_ERR_INVALID;
    if (!unique_specials(&c->tok)) return UENR_ERR_INVALID;
    if (c->sched.min_epochs <= 0 || c->sched.max_epochs < c->sched.min_epochs)
        return UENR_ERR_INVALID;
    if (c->sched.plateau_window <= 0) return UENR_ERR_INVALID;
    if (c->sched.tokens_per_param <= 0.0f) return UENR_ERR_INVALID;
    if (c->sched.target_loss < 0.0f || c->sched.plateau_delta < 0.0f ||
        c->sched.min_steps_before_stop < 0) return UENR_ERR_INVALID;
    if (c->sched.ema_beta < 0.0f || c->sched.ema_beta >= 1.0f) return UENR_ERR_INVALID;
    if (c->lora.rank <= 0 || c->lora.alpha <= 0.0f || c->lora.n_targets < 1)
        return UENR_ERR_INVALID;
    if (c->moe.router_aux_coef < 0.0f) return UENR_ERR_INVALID;
    if (c->prep.sft_prompt_chars <= 0 || c->prep.val_percent < 0 ||
        c->prep.val_percent > 50) return UENR_ERR_INVALID;
    if (c->reward.steps <= 0 || c->reward.lr <= 0.0f) return UENR_ERR_INVALID;
    if (c->infer.max_new_tokens <= 0) return UENR_ERR_INVALID;
    if (c->infer.temperature <= 0.0f) return UENR_ERR_INVALID;
    if (c->infer.top_k < 0 || c->infer.top_p < 0.0f || c->infer.top_p > 1.0f)
        return UENR_ERR_INVALID;
    if (c->infer.repetition_last_n < 0) return UENR_ERR_INVALID;
    if (c->infer.repetition_penalty <= 0.0f) return UENR_ERR_INVALID;
    if (c->infer.no_repeat_ngram < 0) return UENR_ERR_INVALID;
    if (c->infer.min_p < 0.0f || c->infer.min_p > 1.0f) return UENR_ERR_INVALID;
    if (c->infer.kv_recent < 0 || c->infer.kv_sinks < 0) return UENR_ERR_INVALID;
    if (c->infer.kv_evict && c->infer.kv_evict_window <= 0) return UENR_ERR_INVALID;
    if (c->infer.kv_lowrank < 0 || c->infer.kv_lowrank > 1024) return UENR_ERR_INVALID;
    if (c->infer.kv_sparse_k < 0) return UENR_ERR_INVALID;
    if (c->infer.early_exit_delta < 0.0f) return UENR_ERR_INVALID;
    if (c->infer.batch_max < 0 || c->infer.batch_max > 64) return UENR_ERR_INVALID;
    if (c->infer.draft_k < 0 || c->infer.draft_k > 8) return UENR_ERR_INVALID;
    if (c->infer.speculative &&
        (c->infer.spec_k < 1 || c->infer.spec_k > 8 ||
         c->infer.spec_ngram < 1 || c->infer.spec_ngram > 64))
        return UENR_ERR_INVALID;
    if (c->bench.gemm_size <= 0) return UENR_ERR_INVALID;
    if (c->eval.batches <= 0 || c->eval.seq_len <= 0 || c->eval.nseq <= 0)
        return UENR_ERR_INVALID;
    if (c->eval.seq_len > c->model.max_seq) return UENR_ERR_INVALID;
    if (c->eval.gen_tokens < 0) return UENR_ERR_INVALID;
    if (c->rl.max_new_tokens <= 0 || c->rl.temperature <= 0.0f)
        return UENR_ERR_INVALID;
    if (c->ppo.clip_ratio <= 0.0f || c->ppo.clip_ratio >= 1.0f) return UENR_ERR_INVALID;
    if (c->ppo.kl_coef < 0.0f || c->ppo.rollouts < 1 || c->ppo.rollouts > 32)
        return UENR_ERR_INVALID;
    if (c->grpo.group_size < 1 || c->grpo.group_size > 32 ||
        c->grpo.kl_coef < 0.0f) return UENR_ERR_INVALID;
    if (c->dpo.beta <= 0.0f) return UENR_ERR_INVALID;
    if (c->reward.steps <= 0 || c->reward.lr <= 0.0f)
        return UENR_ERR_INVALID;
    return UENR_OK;
}

int uenr_layer_is_sliding(const UenrConfig *c, int l) {
    const char *p = c->model.attn_pattern;
    int n = (int)strlen(p);
    if (n <= 0) return 0; /* default: full attention */
    char ch = p[l % n];
    return ch == 'S' || ch == 's' || ch == 'L' || ch == 'l';
}

int uenr_active_kv_layers(const UenrConfig *c) {
    int boundary = c->model.layers - c->model.kv_shared_layers;
    return boundary < 1 ? 1 : boundary;
}

int uenr_layer_is_shared(const UenrConfig *c, int l) {
    return l >= uenr_active_kv_layers(c);
}

int uenr_layer_donor(const UenrConfig *c, int l) {
    int boundary = uenr_active_kv_layers(c);
    if (l < boundary) return l;
    int want = uenr_layer_is_sliding(c, l);
    int best = -1;
    for (int j = 0; j < boundary; j++)
        if (uenr_layer_is_sliding(c, j) == want) best = j;
    if (best < 0) best = boundary - 1;
    return best;
}

int uenr_total_params(const UenrConfig *c) {
    const UenrModelCfg *m = &c->model;
    long long n = 0;
    n += (long long)m->vocab_size * m->hidden;      /* token embedding */
    if (!m->tie_embeddings) n += (long long)m->vocab_size * m->hidden;
    n += m->hidden;                                  /* final rmsnorm */
    for (int l = 0; l < m->layers; l++) {
        n += 4LL * m->hidden;                        /* pre/post attn+ffn norms */
        n += m->hidden;                              /* attention LayerScale */
        n += (long long)m->head_dim;                 /* q_norm */
        if (!uenr_layer_is_shared(c, l)) n += (long long)m->head_dim; /* k_norm */
        n += (long long)m->hidden * m->hidden;       /* wq */
        if (!uenr_layer_is_shared(c, l)) {
            n += (long long)m->hidden * (m->kv_heads * m->head_dim); /* wk */
            n += (long long)m->hidden * (m->kv_heads * m->head_dim); /* wv */
        } else if (m->kv_adapter) {
            n += 4LL * (long long)(m->kv_heads * m->head_dim);
        }
        n += (long long)m->hidden * m->hidden;       /* wo */
        n += (long long)c->moe.num_experts * m->hidden;  /* router */
        if (c->moe.enabled) {
            n += (long long)c->moe.num_experts * 3LL * m->hidden * c->moe.expert_ffn;
            if (c->moe.shared_expert)
                n += 3LL * m->hidden * m->ffn;
        } else {
            n += 3LL * m->hidden * m->ffn;
        }
    }
    return (int)n;
}

int uenr_active_params(const UenrConfig *c) {
    const UenrModelCfg *m = &c->model;
    long long n = 0;
    for (int l = 0; l < m->layers; l++) {
        n += 5LL * m->hidden + (long long)m->head_dim;
        n += (long long)m->hidden * m->hidden;
        if (!uenr_layer_is_shared(c, l)) {
            n += (long long)m->head_dim;
            n += 2LL * (long long)m->hidden * (m->kv_heads * m->head_dim);
        } else if (m->kv_adapter) {
            n += 4LL * (long long)(m->kv_heads * m->head_dim);
        }
        n += (long long)m->hidden * m->hidden;
        n += (long long)c->moe.num_experts * m->hidden;
        if (c->moe.enabled) {
            n += (long long)c->moe.experts_per_tok * 3LL * m->hidden * c->moe.expert_ffn;
            if (c->moe.shared_expert) n += 3LL * m->hidden * m->ffn;
        } else {
            n += 3LL * m->hidden * m->ffn;
        }
    }
    n += m->hidden;
    return (int)n;
}

int uenr_micro_batch(const UenrConfig *c) {
    return c->train.batch_size / c->train.grad_accum;
}

int uenr_tokens_per_step(const UenrConfig *c) {
    return c->train.batch_size * c->train.seq_len;
}

int uenr_tokenizer_base(void) {
    return UENR_MERGE_BASE;
}

void uenr_config_print_datasets(const UenrConfig *c) {
    if (!c) return;
    struct { const char *name; const UenrDatasetCfg *d; } list[] = {
        { "pretrain", &c->datasets.pretrain }, { "sft", &c->datasets.sft },
        { "lora", &c->datasets.lora }, { "dpo", &c->datasets.dpo },
        { "rl", &c->datasets.rl }, { "reward", &c->datasets.reward },
        { "eval", &c->datasets.eval },
    };
    for (size_t i = 0; i < sizeof(list) / sizeof(list[0]); i++) {
        const UenrDatasetCfg *d = list[i].d;
        const char *derive_path = "";
        if (!strcmp(d->derive_from, "pretrain"))
            derive_path = c->datasets.pretrain.paths[0];
        else if (!strcmp(d->derive_from, "sft"))
            derive_path = c->datasets.sft.paths[0];
        else if (!strcmp(d->derive_from, "lora"))
            derive_path = c->datasets.lora.paths[0];
        else if (!strcmp(d->derive_from, "dpo"))
            derive_path = c->datasets.dpo.paths[0];
        else if (!strcmp(d->derive_from, "rl"))
            derive_path = c->datasets.rl.paths[0];
        else if (!strcmp(d->derive_from, "reward"))
            derive_path = c->datasets.reward.paths[0];
        int n = d->n_paths > 0 ? d->n_paths : 1;
        for (int p = 0; p < n; p++) {
            printf("%s\x1f%d\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%d\x1f%d\x1f%s\n",
                   list[i].name, p + 1, d->n_paths > 0 ? d->paths[p] : "",
                   d->release_tag, d->release_asset, d->url, d->derive_from,
                   derive_path, d->convert_from, d->convert_to, d->limit,
                   d->bootstrap, d->format);
        }
    }
}

void uenr_config_print(const UenrConfig *c) {
    if (c == NULL) return;
    printf("[config] name=%s seed=%d threads=%d simd=%s\n", c->run.name,
           c->run.seed, c->run.threads,
           c->run.simd[0] ? c->run.simd : "auto");
    printf("[config] model V=%d C=%d L=%d H=%d KV=%d hd=%d ffn=%d moe=%d E=%d top=%d e_ffn=%d\n",
           c->model.vocab_size, c->model.hidden, c->model.layers,
           c->model.heads, c->model.kv_heads, c->model.head_dim, c->model.ffn,
           c->moe.enabled, c->moe.num_experts, c->moe.experts_per_tok, c->moe.expert_ffn);
    printf("[config] params total=%d active=%d\n",
           uenr_total_params(c), uenr_active_params(c));
    printf("[config] train bs=%d seq=%d accum=%d micro=%d lr=%g schedule=%s\n",
           c->train.batch_size, c->train.seq_len, c->train.grad_accum,
           uenr_micro_batch(c), (double)c->train.lr, c->sched.kind);
    printf("[config] tokenizer V=%d train_bytes=%d bos=%d eos=%d\n",
           c->model.vocab_size, c->tok.train_bytes, c->tok.bos_id, c->tok.eos_id);
    printf("[config] datasets:\n");
    const struct { const char *name; const UenrDatasetCfg *d; } ds[] = {
        {"pretrain", &c->datasets.pretrain}, {"sft", &c->datasets.sft},
        {"lora", &c->datasets.lora}, {"dpo", &c->datasets.dpo},
        {"rl", &c->datasets.rl}, {"reward", &c->datasets.reward},
        {"eval", &c->datasets.eval},
    };
    for (size_t i = 0; i < sizeof(ds) / sizeof(ds[0]); i++) {
        printf("  %-8s format=%-32s paths(%d):", ds[i].name,
               ds[i].d->format[0] ? ds[i].d->format : "(inherit/none)",
               ds[i].d->n_paths);
        for (int j = 0; j < ds[i].d->n_paths; j++)
            printf(" %s", ds[i].d->paths[j]);
        printf("\n");
    }
    printf("[config] memory adafactor=%d muon=%d ns_iters=%d block_checkpoint=%d "
           "recompute_attn=%d fp16_activations=%d checkpoint_every=%d resume=%d\n",
           c->train.adafactor, c->train.muon, c->train.muon_ns_iters,
           c->train.block_checkpoint, c->train.recompute_attn,
           c->train.fp16_activations, c->run.checkpoint_every, c->run.resume);
    printf("[config] sft_pack=%d\n", c->train.sft_pack);
    printf("[config] swiglu_clamp=%.1f kv=%s evict=%d\n",
           (double)c->train.swiglu_clamp,
           c->infer.kv_int8 ? "int8" : c->infer.kv_fp16 ? "fp16" :
           c->infer.kv_q4 ? "q4" : c->infer.kv_q2 ? "q2" : "fp32",
           c->infer.kv_evict);
}
