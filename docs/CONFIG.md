# Configuration Reference

`config/uenr.toml` is the **only** place training and model settings live.
There are no hardcoded hyperparameters in the C sources. The parser is a
handwritten TOML subset: sections, `key = value`, strings, integers, floats,
booleans, and string arrays. Comments use `#`.

## `[run]`

| Key | Meaning |
|---|---|
| `name` | Run label (printed in logs). |
| `seed` | xorshift128+ seed for init and data order. |
| `threads` | Thread pool size (1 = single-threaded; 2 recommended). |
| `pin_cpu0`, `pin_cpu1` | Physical CPU ids for the main thread / worker. |
| `log_every` | Steps between log/CSV rows. |
| `checkpoint_every` | Steps between periodic atomic checkpoints (default 500; newest 3 kept; 0 disables). |
| `checkpoint_dir` | Directory for periodic checkpoints. |
| `resume` | Resume from the newest of `checkpoint_dir/step_*.bin` and `paths.model_out`, restoring optimizer state. Also `uenr train|sft --resume`. |

## `[paths]`

`data` (raw corpus), `train_bin`, `val_bin` (prepared uint16 streams),
`sft_jsonl`, `pairs_jsonl`, tokenizer artifacts (`tokenizer`, `vocab_file`,
`merges_file`), and output checkpoints (`model_out`, `sft_out`, `lora_out`,
`dpo_out`, `ppo_out`, `grpo_out`, `log_csv`).

## `[hardware]`

Tuning constants measured on this machine: `clock_hz`, ISA flags, `align`
(32/64), and GEMM tile knobs (`micro_m`, `micro_n`, `l1_block`, `l2_block`).
Changing these does not alter model math; they exist so the kernel can be
re-tuned if the machine or measurements change.

## `[model]`

| Key | Default | Meaning |
|---|---|---|
| `vocab_size` | 2048 | Must match the tokenizer artifact. |
| `hidden` | 192 | Residual width `C`. |
| `layers` | 4 | Transformer blocks. |
| `heads` | 6 | Query heads. |
| `kv_heads` | 1 | KV heads. 1 = MQA (shipped, with exact compensation). |
| `head_dim` | 32 | Must satisfy `heads·head_dim == hidden`. |
| `ffn` | 277 | Shared/dense expert width (compensates the KV-head reduction). |
| `rms_eps` | 1e-6 | RMSNorm epsilon. |
| `rope_theta` | 1e6 | Fallback RoPE base. |
| `rope_theta_local/global` | 1e4 / 1e6 | Per-attention-type bases. |
| `rope_partial_local/global` | 1.0 / 1.0 | p-RoPE factor per type (DOCUMENT mode uses 0.5 global). |
| `rope_scale`, `rope_yarn`, `rope_beta_fast/slow`, `rope_attn_factor` | 1.0,false,32,1,1 | Long-context RoPE scaling. |
| `kv_shared_layers` | 2 | Trailing layers that borrow donor K/V. |
| `attn_pattern` | `SGSG` | Cycled layer types (S=sliding, G=global). |
| `sliding_window` | 128 | Ring/window size for S layers (0 = full). |
| `attn_scale` | 0 | 0 = 1/√head_dim; 1.0 enables the DOCUMENT unit-scale mode. |
| `v_norm` | false | Scale-free V-norm (DOCUMENT mode; smoke-scale A/B favored off). |
| `attn_softcap`, `final_softcap` | 50, 30 | tanh logit caps (0 disables). |
| `post_norm_init` | 0 | Init scale for the two post-sub-layer norms. |
| `tie_embeddings` | true | Reuse `tok_emb` as the LM head. |
| `max_seq` | 512 | Maximum context. |
| `flash_attn` | false | Online-softmax attention; removes the O(S²) probability buffer. Slower at short S on this CPU, better for long context/memory. |

## `[moe]`

`enabled`, `num_experts`, `experts_per_tok` (top-k), `expert_ffn`,
`shared_expert`, `router_aux_coef` (load-balance loss), `router_jitter`
(reserved), `capacity_factor` (0 = no capacity drop).

## `[tokenizer]`

`kind` (`bpe`), `lowercase`, `normalize_unicode` (UTF-8 validation and line
normalisation), `strip_control`, `train_bytes` (sample size for BPE training),
special ids (`bos_id`, `eos_id`, `pad_id`, `unk_id`, `special_system`,
`special_user`, `special_assistant`, `special_tool_call`,
`special_tool_result`, `special_end`), `add_bos`/`add_eos`, `ignore_index`.
Ids 0–15 are reserved for specials; bytes are 16–271; merges start at 272.

## `[prepare]`

`max_bytes` (raw bytes encoded by `uenr prepare`), `val_percent`, `dedup`/`dedup_slots`, and the
synthetic SFT/pair generator settings (`sft_stories`, `sft_prompt_chars`).

## `[training]`

`mode` (label), `batch_size` (sequences per optimizer step), `grad_accum`
(micro-batches per step; `batch_size % grad_accum == 0`), `seq_len`,
`max_steps`, `lr` (default 1e-3 from the sweep), `min_lr`, `warmup_steps` (20), `weight_decay`, `beta1`, `beta2`,
`adam_eps`, `clip`, `dropout` (reserved), `dtype`, `sampled_softmax`
(negative-sampled training; default off), `sampled_negatives`, `z_loss`,
`ema_decay` (0 off), `optim8` (blockwise INT8 first moment), and
`block_checkpoint` (full block gradient checkpointing: one layer's saved
activations, each layer forward re-run in backward), `fp16_activations`
(store attention probabilities as IEEE FP16), `recompute_attn` (regenerate
attention probs in backward), plus
`adafactor` (factored optimizer state), `seq_ramp_steps` (seq-length curriculum), `weight_qat_bits` (0/4/8 weight QAT), `qlora` (INT4-quantized frozen base for LoRA), `recompute_swiglu` (recompute the FFN gate/up cache), and `kv_qat_bits`
(0 off, 4=int4, 2=ternary KV quantization-aware training, straight-through),
`kv_lowrank` (QAT-style Hadamard truncation rank for training).

## `[schedule]`

`kind` (`adaptive` or `cosine`), `tokens_per_param` (epoch formula),
`min_epochs`, `max_epochs`, `target_loss`, `plateau_window`, `plateau_delta`,
`min_steps_before_stop`, `ema_beta`. See `docs/FORMULAS.md` §12.

## `[lora]`

`rank`, `alpha` (scale = alpha/rank), `dropout` (reserved), `targets`
(array of `q_proj`, `k_proj`, `v_proj`, `o_proj`).

## `[dpo]`

`beta`, `ref_checkpoint` (frozen reference). If the configured reference is
missing, the base `paths.model_out` is used explicitly and reported; if neither
is valid the run fails rather than using random reference weights.

## `[ppo]`

`clip_ratio` (symmetric PPO epsilon), `kl_coef` (reference KL penalty),
`rollouts` (number of completions per prompt). There is no value head, entropy
bonus or GAE: unsupported `value_coef`/`entropy_coef`/`gamma`/`lambda` keys
were removed rather than silently ignored.

## `[grpo]`

`group_size`, `kl_coef`.

## `[reward]`

`kind` (`heuristic` or `model`), `checkpoint`, `steps`, `lr`. The model reward
is a linear pairwise head over the frozen base model's final hidden state.

## `[rl]`

`max_new_tokens` per rollout, `temperature` (scales the rollout sampling
distribution only; policy log-probabilities and PPO ratios are evaluated at
T=1 so the importance ratio stays valid), reward shaping: `reward_format`,
`reward_length`, `length_target`, `reward_repeat`.

## `[inference]`

`max_new_tokens`, `system` prompt, `temperature`, `top_k`, `top_p`, `greedy`,
`kv_cache` (incremental decode), `kv_int8` (per-head quantized KV),
`kv_fp16` (F16C FP16 KV), `kv_q2` (ternary KV with `kv_recent`/`kv_sinks` FP32 mixed precision), `kv_q4` (packed 4-bit KV with per-head scales), `kv_rotate` (Hadamard-rotate K/V before q4), `weight_int8` (per-row
quantized inference weights), `repetition_penalty` + `repetition_last_n`,
`no_repeat_ngram`, `min_p`, `json_constrained`, `speculative` + `spec_k` +
`spec_ngram`, and `batch_max` and `continuous` for `infer --batch`; `seq_ramp_steps` in training; `muon`, and `draft_model`/`draft_config`/`draft_k`
for draft-model speculative decoding.

## `[eval]`

`split`, `batches`, `seq_len`, `nseq`, plus `gen_tokens` and `prompt` for the
greedy generation metrics (end rate, repeated token-3grams, tool-JSON check).

## `[bench]`

`gemm_size` used by `uenr bench` when no `--size` is given.

## Validation

`uenr_config_validate` enforces: shape consistency
(`heads·head_dim == hidden`), `kv_heads | heads`, vocab range, sequence length,
thread bounds, positive widths, unique special ids in 0–15,
`batch_size % grad_accum == 0`, and positive schedule/reward values. A config
that fails prints an error and exits non-zero before any work starts.
