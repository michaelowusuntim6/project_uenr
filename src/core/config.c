#include "uenr/config.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *trim(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = '\0';
    return s;
}

static void strip_comment(char *s) {
    int in_str = 0;
    for (char *p = s; *p; p++) {
        if (*p == '"') in_str = !in_str;
        else if (*p == '#' && !in_str) { *p = '\0'; return; }
    }
}

static int parse_bool(const char *v) {
    return (strcmp(v, "true") == 0 || strcmp(v, "1") == 0);
}

static void parse_string(const char *v, char *dst, size_t n) {
    size_t len = strlen(v);
    if (len >= 2 && v[0] == '"' && v[len - 1] == '"') {
        size_t inner = len - 2;
        if (inner >= n) inner = n - 1;
        memcpy(dst, v + 1, inner);
        dst[inner] = '\0';
    } else {
        snprintf(dst, n, "%s", v);
    }
}

typedef enum {
    SEC_NONE = 0, SEC_RUN, SEC_PATHS, SEC_HW, SEC_MODEL, SEC_MOE,
    SEC_TOKENIZER, SEC_PREPARE, SEC_TRAINING, SEC_SCHEDULE, SEC_LORA,
    SEC_DPO, SEC_PPO, SEC_GRPO, SEC_RL, SEC_INFERENCE, SEC_BENCH
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
    if (!strcmp(name, "rl")) return SEC_RL;
    if (!strcmp(name, "inference")) return SEC_INFERENCE;
    if (!strcmp(name, "bench")) return SEC_BENCH;
    return SEC_NONE;
}

static void set_kv(UenrConfig *c, Section sec, const char *k, const char *v) {
    switch (sec) {
    case SEC_RUN:
        if (!strcmp(k, "name")) parse_string(v, c->run.name, UENR_MAX_STR);
        else if (!strcmp(k, "seed")) c->run.seed = atoi(v);
        else if (!strcmp(k, "threads")) c->run.threads = atoi(v);
        else if (!strcmp(k, "pin_cpu0")) c->run.pin_cpu0 = atoi(v);
        else if (!strcmp(k, "pin_cpu1")) c->run.pin_cpu1 = atoi(v);
        else if (!strcmp(k, "log_every")) c->run.log_every = atoi(v);
        else if (!strcmp(k, "checkpoint_every")) c->run.checkpoint_every = atoi(v);
        else if (!strcmp(k, "checkpoint_dir")) parse_string(v, c->run.checkpoint_dir, UENR_MAX_STR);
        else if (!strcmp(k, "resume")) c->run.resume = parse_bool(v);
        break;
    case SEC_PATHS:
        if (!strcmp(k, "data")) parse_string(v, c->paths.data, UENR_MAX_STR);
        else if (!strcmp(k, "train_bin")) parse_string(v, c->paths.train_bin, UENR_MAX_STR);
        else if (!strcmp(k, "val_bin")) parse_string(v, c->paths.val_bin, UENR_MAX_STR);
        else if (!strcmp(k, "sft_jsonl")) parse_string(v, c->paths.sft_jsonl, UENR_MAX_STR);
        else if (!strcmp(k, "pairs_jsonl")) parse_string(v, c->paths.pairs_jsonl, UENR_MAX_STR);
        else if (!strcmp(k, "tokenizer")) parse_string(v, c->paths.tokenizer, UENR_MAX_STR);
        else if (!strcmp(k, "vocab_file")) parse_string(v, c->paths.vocab_file, UENR_MAX_STR);
        else if (!strcmp(k, "merges_file")) parse_string(v, c->paths.merges_file, UENR_MAX_STR);
        else if (!strcmp(k, "model_out")) parse_string(v, c->paths.model_out, UENR_MAX_STR);
        else if (!strcmp(k, "log_csv")) parse_string(v, c->paths.log_csv, UENR_MAX_STR);
        else if (!strcmp(k, "sft_out")) parse_string(v, c->paths.sft_out, UENR_MAX_STR);
        else if (!strcmp(k, "lora_out")) parse_string(v, c->paths.lora_out, UENR_MAX_STR);
        else if (!strcmp(k, "dpo_out")) parse_string(v, c->paths.dpo_out, UENR_MAX_STR);
        else if (!strcmp(k, "ppo_out")) parse_string(v, c->paths.ppo_out, UENR_MAX_STR);
        else if (!strcmp(k, "grpo_out")) parse_string(v, c->paths.grpo_out, UENR_MAX_STR);
        break;
    case SEC_HW:
        if (!strcmp(k, "clock_hz")) c->hw.clock_hz = atoll(v);
        else if (!strcmp(k, "avx")) c->hw.avx = parse_bool(v);
        else if (!strcmp(k, "fma")) c->hw.fma = parse_bool(v);
        else if (!strcmp(k, "avx2")) c->hw.avx2 = parse_bool(v);
        else if (!strcmp(k, "fp16")) c->hw.fp16 = parse_bool(v);
        else if (!strcmp(k, "align")) c->hw.align = atoi(v);
        else if (!strcmp(k, "micro_m")) c->hw.micro_m = atoi(v);
        else if (!strcmp(k, "micro_n")) c->hw.micro_n = atoi(v);
        else if (!strcmp(k, "l1_block")) c->hw.l1_block = atoi(v);
        else if (!strcmp(k, "l2_block")) c->hw.l2_block = atoi(v);
        break;
    case SEC_MODEL:
        if (!strcmp(k, "vocab_size")) c->model.vocab_size = atoi(v);
        else if (!strcmp(k, "hidden")) c->model.hidden = atoi(v);
        else if (!strcmp(k, "layers")) c->model.layers = atoi(v);
        else if (!strcmp(k, "heads")) c->model.heads = atoi(v);
        else if (!strcmp(k, "kv_heads")) c->model.kv_heads = atoi(v);
        else if (!strcmp(k, "head_dim")) c->model.head_dim = atoi(v);
        else if (!strcmp(k, "ffn")) c->model.ffn = atoi(v);
        else if (!strcmp(k, "rms_eps")) c->model.rms_eps = strtof(v, NULL);
        else if (!strcmp(k, "rope_theta")) c->model.rope_theta = strtof(v, NULL);
        else if (!strcmp(k, "tie_embeddings")) c->model.tie_embeddings = parse_bool(v);
        else if (!strcmp(k, "max_seq")) c->model.max_seq = atoi(v);
        else if (!strcmp(k, "flash_attn")) c->model.flash_attn = parse_bool(v);
        break;
    case SEC_MOE:
        if (!strcmp(k, "enabled")) c->moe.enabled = parse_bool(v);
        else if (!strcmp(k, "num_experts")) c->moe.num_experts = atoi(v);
        else if (!strcmp(k, "experts_per_tok")) c->moe.experts_per_tok = atoi(v);
        else if (!strcmp(k, "expert_ffn")) c->moe.expert_ffn = atoi(v);
        else if (!strcmp(k, "shared_expert")) c->moe.shared_expert = parse_bool(v);
        else if (!strcmp(k, "router_aux_coef")) c->moe.router_aux_coef = strtof(v, NULL);
        else if (!strcmp(k, "router_jitter")) c->moe.router_jitter = strtof(v, NULL);
        else if (!strcmp(k, "capacity_factor")) c->moe.capacity_factor = strtof(v, NULL);
        break;
    case SEC_TOKENIZER:
        if (!strcmp(k, "kind")) parse_string(v, c->tok.kind, UENR_MAX_STR);
        else if (!strcmp(k, "lowercase")) c->tok.lowercase = parse_bool(v);
        else if (!strcmp(k, "normalize_unicode")) c->tok.normalize_unicode = parse_bool(v);
        else if (!strcmp(k, "strip_control")) c->tok.strip_control = parse_bool(v);
        else if (!strcmp(k, "train_bytes")) c->tok.train_bytes = atoi(v);
        else if (!strcmp(k, "bos_id")) c->tok.bos_id = atoi(v);
        else if (!strcmp(k, "eos_id")) c->tok.eos_id = atoi(v);
        else if (!strcmp(k, "pad_id")) c->tok.pad_id = atoi(v);
        else if (!strcmp(k, "unk_id")) c->tok.unk_id = atoi(v);
        else if (!strcmp(k, "add_bos")) c->tok.add_bos = parse_bool(v);
        else if (!strcmp(k, "add_eos")) c->tok.add_eos = parse_bool(v);
        else if (!strcmp(k, "ignore_index")) c->tok.ignore_index = atoi(v);
        else if (!strcmp(k, "special_system")) c->tok.special_system = atoi(v);
        else if (!strcmp(k, "special_user")) c->tok.special_user = atoi(v);
        else if (!strcmp(k, "special_assistant")) c->tok.special_assistant = atoi(v);
        else if (!strcmp(k, "special_tool_call")) c->tok.special_tool_call = atoi(v);
        else if (!strcmp(k, "special_tool_result")) c->tok.special_tool_result = atoi(v);
        else if (!strcmp(k, "special_end")) c->tok.special_end = atoi(v);
        break;
    case SEC_PREPARE:
        if (!strcmp(k, "max_bytes")) c->prep.max_bytes = atoi(v);
        else if (!strcmp(k, "val_percent")) c->prep.val_percent = atoi(v);
        else if (!strcmp(k, "sft_stories")) c->prep.sft_stories = atoi(v);
        else if (!strcmp(k, "sft_prompt_chars")) c->prep.sft_prompt_chars = atoi(v);
        break;
    case SEC_TRAINING:
        if (!strcmp(k, "mode")) parse_string(v, c->train.mode, UENR_MAX_STR);
        else if (!strcmp(k, "batch_size")) c->train.batch_size = atoi(v);
        else if (!strcmp(k, "seq_len")) c->train.seq_len = atoi(v);
        else if (!strcmp(k, "grad_accum")) c->train.grad_accum = atoi(v);
        else if (!strcmp(k, "max_steps")) c->train.max_steps = atoi(v);
        else if (!strcmp(k, "lr")) c->train.lr = strtof(v, NULL);
        else if (!strcmp(k, "min_lr")) c->train.min_lr = strtof(v, NULL);
        else if (!strcmp(k, "warmup_steps")) c->train.warmup_steps = atoi(v);
        else if (!strcmp(k, "weight_decay")) c->train.weight_decay = strtof(v, NULL);
        else if (!strcmp(k, "beta1")) c->train.beta1 = strtof(v, NULL);
        else if (!strcmp(k, "beta2")) c->train.beta2 = strtof(v, NULL);
        else if (!strcmp(k, "adam_eps")) c->train.adam_eps = strtof(v, NULL);
        else if (!strcmp(k, "clip")) c->train.clip = strtof(v, NULL);
        else if (!strcmp(k, "dropout")) c->train.dropout = strtof(v, NULL);
        else if (!strcmp(k, "dtype")) parse_string(v, c->train.dtype, UENR_MAX_STR);
        else if (!strcmp(k, "sampled_softmax")) c->train.sampled_softmax = parse_bool(v);
        else if (!strcmp(k, "sampled_negatives")) c->train.sampled_negatives = atoi(v);
        break;
    case SEC_SCHEDULE:
        if (!strcmp(k, "kind")) parse_string(v, c->sched.kind, UENR_MAX_STR);
        else if (!strcmp(k, "tokens_per_param")) c->sched.tokens_per_param = strtof(v, NULL);
        else if (!strcmp(k, "min_epochs")) c->sched.min_epochs = atoi(v);
        else if (!strcmp(k, "max_epochs")) c->sched.max_epochs = atoi(v);
        else if (!strcmp(k, "target_loss")) c->sched.target_loss = strtof(v, NULL);
        else if (!strcmp(k, "plateau_window")) c->sched.plateau_window = atoi(v);
        else if (!strcmp(k, "plateau_delta")) c->sched.plateau_delta = strtof(v, NULL);
        else if (!strcmp(k, "min_steps_before_stop")) c->sched.min_steps_before_stop = atoi(v);
        else if (!strcmp(k, "ema_beta")) c->sched.ema_beta = strtof(v, NULL);
        break;
    case SEC_LORA:
        if (!strcmp(k, "rank")) c->lora.rank = atoi(v);
        else if (!strcmp(k, "alpha")) c->lora.alpha = strtof(v, NULL);
        else if (!strcmp(k, "dropout")) c->lora.dropout = strtof(v, NULL);
        else if (!strcmp(k, "targets")) {
            char *p = (char *)v;
            while (*p && c->lora.n_targets < UENR_MAX_TARGETS) {
                while (*p && *p != '"') p++;
                if (!*p) break;
                p++;
                char *start = p;
                while (*p && *p != '"') p++;
                size_t len = (size_t)(p - start);
                if (len >= 32) len = 31;
                memcpy(c->lora.targets[c->lora.n_targets], start, len);
                c->lora.targets[c->lora.n_targets][len] = '\0';
                c->lora.n_targets++;
                if (*p) p++;
            }
        }
        break;
    case SEC_DPO:
        if (!strcmp(k, "beta")) c->dpo.beta = strtof(v, NULL);
        else if (!strcmp(k, "ref_checkpoint")) parse_string(v, c->dpo.ref_checkpoint, UENR_MAX_STR);
        break;
    case SEC_PPO:
        if (!strcmp(k, "clip_ratio")) c->ppo.clip_ratio = strtof(v, NULL);
        else if (!strcmp(k, "value_coef")) c->ppo.value_coef = strtof(v, NULL);
        else if (!strcmp(k, "entropy_coef")) c->ppo.entropy_coef = strtof(v, NULL);
        else if (!strcmp(k, "gamma")) c->ppo.gamma = strtof(v, NULL);
        else if (!strcmp(k, "lambda")) c->ppo.lambda = strtof(v, NULL);
        else if (!strcmp(k, "rollouts")) c->ppo.rollouts = atoi(v);
        break;
    case SEC_GRPO:
        if (!strcmp(k, "group_size")) c->grpo.group_size = atoi(v);
        else if (!strcmp(k, "kl_coef")) c->grpo.kl_coef = strtof(v, NULL);
        break;
    case SEC_RL:
        if (!strcmp(k, "max_new_tokens")) c->rl.max_new_tokens = atoi(v);
        else if (!strcmp(k, "temperature")) c->rl.temperature = strtof(v, NULL);
        else if (!strcmp(k, "reward_format")) c->rl.reward_format = strtof(v, NULL);
        else if (!strcmp(k, "reward_length")) c->rl.reward_length = strtof(v, NULL);
        else if (!strcmp(k, "length_target")) c->rl.length_target = atoi(v);
        else if (!strcmp(k, "reward_repeat")) c->rl.reward_repeat = strtof(v, NULL);
        break;
    case SEC_INFERENCE:
        if (!strcmp(k, "max_new_tokens")) c->infer.max_new_tokens = atoi(v);
        else if (!strcmp(k, "system")) parse_string(v, c->infer.system, UENR_MAX_STR);
        else if (!strcmp(k, "temperature")) c->infer.temperature = strtof(v, NULL);
        else if (!strcmp(k, "top_k")) c->infer.top_k = atoi(v);
        else if (!strcmp(k, "top_p")) c->infer.top_p = strtof(v, NULL);
        else if (!strcmp(k, "greedy")) c->infer.greedy = parse_bool(v);
        else if (!strcmp(k, "kv_cache")) c->infer.kv_cache = parse_bool(v);
        break;
    case SEC_BENCH:
        if (!strcmp(k, "gemm_size")) c->bench.gemm_size = atoi(v);
        break;
    default:
        break;
    }
}

UenrResult uenr_config_load(const char *path, UenrConfig *out) {
    if (out == NULL) return UENR_ERR_INVALID;
    memset(out, 0, sizeof(*out));
    const char *p = (path == NULL) ? UENR_DEFAULT_CONFIG : path;
    FILE *f = fopen(p, "r");
    if (f == NULL) return UENR_ERR_IO;

    char line[1024];
    Section sec = SEC_NONE;
    while (fgets(line, sizeof(line), f) != NULL) {
        strip_comment(line);
        char *s = trim(line);
        if (*s == '\0') continue;
        if (*s == '[') {
            char *end = strchr(s, ']');
            if (end == NULL) { fclose(f); return UENR_ERR_INVALID; }
            *end = '\0';
            sec = section_from(trim(s + 1));
            continue;
        }
        char *eq = strchr(s, '=');
        if (eq == NULL) { fclose(f); return UENR_ERR_INVALID; }
        *eq = '\0';
        char *key = trim(s);
        char *val = trim(eq + 1);
        if (sec == SEC_NONE) { fclose(f); return UENR_ERR_INVALID; }
        set_kv(out, sec, key, val);
    }
    fclose(f);
    return uenr_config_validate(out);
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
    if (c->train.seq_len <= 0 || c->train.seq_len > c->model.max_seq)
        return UENR_ERR_INVALID;
    if (c->run.threads <= 0 || c->run.threads > 64) return UENR_ERR_INVALID;
    if (c->moe.enabled) {
        if (c->moe.num_experts <= 0) return UENR_ERR_INVALID;
        if (c->moe.experts_per_tok <= 0 || c->moe.experts_per_tok > c->moe.num_experts)
            return UENR_ERR_INVALID;
        if (c->moe.expert_ffn <= 0) return UENR_ERR_INVALID;
    }
    if (c->train.batch_size <= 0 || c->train.grad_accum <= 0) return UENR_ERR_INVALID;
    if (c->train.batch_size % c->train.grad_accum != 0) return UENR_ERR_INVALID;
    if (c->train.lr <= 0.0f) return UENR_ERR_INVALID;
    if (c->train.sampled_softmax &&
        (c->train.sampled_negatives < 1 ||
         c->train.sampled_negatives >= c->model.vocab_size))
        return UENR_ERR_INVALID;
    if (c->hw.align <= 0 || (c->hw.align & (c->hw.align - 1)) != 0)
        return UENR_ERR_INVALID;
    if (c->hw.micro_m <= 0 || c->hw.micro_n <= 0) return UENR_ERR_INVALID;
    if (c->prep.max_bytes <= 0) return UENR_ERR_INVALID;
    if (c->prep.val_percent < 0 || c->prep.val_percent > 50) return UENR_ERR_INVALID;
    if (c->prep.sft_stories < 0) return UENR_ERR_INVALID;
    if (c->tok.train_bytes <= 0) return UENR_ERR_INVALID;
    if (!unique_specials(&c->tok)) return UENR_ERR_INVALID;
    if (c->sched.min_epochs <= 0 || c->sched.max_epochs < c->sched.min_epochs)
        return UENR_ERR_INVALID;
    if (c->sched.plateau_window <= 0) return UENR_ERR_INVALID;
    if (c->sched.tokens_per_param <= 0.0f) return UENR_ERR_INVALID;
    if (c->lora.rank <= 0) return UENR_ERR_INVALID;
    if (c->infer.max_new_tokens <= 0) return UENR_ERR_INVALID;
    if (c->bench.gemm_size <= 0) return UENR_ERR_INVALID;
    if (c->rl.max_new_tokens <= 0 || c->rl.temperature <= 0.0f)
        return UENR_ERR_INVALID;
    return UENR_OK;
}

int uenr_total_params(const UenrConfig *c) {
    const UenrModelCfg *m = &c->model;
    long long n = 0;
    n += (long long)m->vocab_size * m->hidden;      /* token embedding */
    if (!m->tie_embeddings) n += (long long)m->vocab_size * m->hidden;
    n += m->hidden;                                  /* final rmsnorm */
    for (int l = 0; l < m->layers; l++) {
        n += 2LL * m->hidden;                        /* two rmsnorms */
        n += 2LL * m->head_dim;                      /* q_norm, k_norm */
        n += (long long)m->hidden * m->hidden;       /* wq */
        n += (long long)m->hidden * (m->kv_heads * m->head_dim); /* wk */
        n += (long long)m->hidden * (m->kv_heads * m->head_dim); /* wv */
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
        n += 2LL * m->hidden + 2LL * m->head_dim;
        n += (long long)m->hidden * m->hidden;
        n += 2LL * (long long)m->hidden * (m->kv_heads * m->head_dim);
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

void uenr_config_print(const UenrConfig *c) {
    if (c == NULL) return;
    printf("[config] name=%s seed=%d threads=%d\n", c->run.name, c->run.seed, c->run.threads);
    printf("[config] model V=%d C=%d L=%d H=%d KV=%d hd=%d ffn=%d moe=%d E=%d top=%d e_ffn=%d\n",
           c->model.vocab_size, c->model.hidden, c->model.layers,
           c->model.heads, c->model.kv_heads, c->model.head_dim, c->model.ffn,
           c->moe.enabled, c->moe.num_experts, c->moe.experts_per_tok, c->moe.expert_ffn);
    printf("[config] params total=%d active=%d\n",
           uenr_total_params(c), uenr_active_params(c));
    printf("[config] train mode=%s bs=%d seq=%d accum=%d micro=%d lr=%g schedule=%s\n",
           c->train.mode, c->train.batch_size, c->train.seq_len,
           c->train.grad_accum, uenr_micro_batch(c), (double)c->train.lr, c->sched.kind);
    printf("[config] tokenizer V=%d train_bytes=%d bos=%d eos=%d\n",
           c->model.vocab_size, c->tok.train_bytes, c->tok.bos_id, c->tok.eos_id);
}
