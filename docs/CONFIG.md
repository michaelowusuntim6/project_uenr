# Configuration Reference

`config/uenr.toml` is the **only** place training and model settings live.
There are no hardcoded hyperparameters in the C sources. The parser is a
handwritten, strict TOML subset:

- `[section]` headers and `key = value` lines; `#` starts a comment outside a
  double-quoted string.
- Strings are double-quoted and support `\"`, `\\`, `\n`, `\r`, `\t`.
- Integers are decimal and must fit in `int`; floats must be finite.
- Booleans are exactly `true` or `false`.
- String arrays are supported for `[lora].targets` only: `["q", "k"]`.
- Unknown sections/keys, malformed values, trailing junk, overflow, invalid
  escapes and unterminated strings are reported with the file, line, section
  and key, and the load fails. Nothing is silently replaced by a default.
- Semantic ranges are validated after parsing; the error points to this file.

## `[run]`

| Key | Meaning |
|---|---|
| `name` | Run label (printed in logs). |
| `seed` | xorshift128+ seed for init and data order. |
| `threads` | Thread pool size (1 = single-threaded; 2 on the laptop, 4 on a 2C/4T runner). |
| `pin_cpu0`, `pin_cpu1` | CPU ids for the main thread / worker; remaining workers take the lowest unused ids. On a 2C/4T host with siblings `0-1` and `2-3`, 4-thread runs may pin 0/1 (the pool fills 2/3); 2-thread runs must pin 0/2. |
| `simd` | `auto` (default), `avx1`, `avx2`, or `avx512`. `auto` selects AVX-512 when AVX512F/VL/BW are present, else AVX2+FMA, else AVX1. Forcing an unsupported level is an error. |
| `smt_threads` | Optional second pool for memory-bound phases (1 = disabled; measured 0.99x on the laptop). |
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

`huge_pages` enables the transparent-huge-page allocation hint (1 = on).
Instruction-set selection is a runtime option in `[run] simd` (and the
`--simd` CLI flag). Normal objects are compiled for Ivy Bridge (`-march=
ivybridge`: AVX1 + F16C); the runtime-dispatched `*_avx2.c` objects are built
with `-march=x86-64-v3 -mavx2 -mfma` and are only called when the CPU has
both features. The old no-op `clock_hz`, `avx`, `fma`, `avx2`, `fp16`,
`align`, `micro_m`, `micro_n`, `l1_block` and `l2_block` keys were removed
because nothing consumed them; unknown keys are now rejected.

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
`shared_expert`, `router_aux_coef` (load-balance loss). Router jitter and
capacity dropping are not implemented; their configuration keys were removed
rather than silently ignored.

## `[tokenizer]`

`kind` (`bpe`), `lowercase`, `normalize_unicode` (UTF-8 validation and line
normalisation), `strip_control`, `train_bytes` (sample size for BPE training),
special ids (`bos_id`, `eos_id`, `pad_id`, `unk_id`, `special_system`,
`special_user`, `special_assistant`, `special_tool_call`,
`special_tool_result`, `special_end`), `add_bos`/`add_eos`, `ignore_index`.
Ids 0–15 are reserved for specials; bytes are 16–271; merges start at 272.

## `[prepare]`

`max_bytes` (raw bytes encoded by `uenr prepare`; shipped default 209715200 =
200 MiB, `0` means the whole file), `val_percent`, `dedup`/`dedup_slots`, and
the synthetic SFT/pair generator settings (`sft_stories`, `sft_prompt_chars`).
Tokenizer training has a separate 8 MiB sample limit in
`[tokenizer].train_bytes`; the two are independent.

## `[training]`

`batch_size` (sequences per optimizer step; cloud 64, laptop 16),
`grad_accum` (micro-batches per step; `batch_size % grad_accum == 0`; cloud 1,
laptop 2), `seq_len` (256; tokens/step = batch_size × seq_len),
`max_steps`, `lr` (default 1e-3 from the sweep), `min_lr`, `warmup_steps` (20), `weight_decay`, `beta1`, `beta2`,
`adam_eps`, `clip`, `sampled_softmax`
(negative-sampled training; default off), `sampled_negatives`, `z_loss`,
`ema_decay` (0 off), `optim8` (blockwise INT8 first moment), and
`block_checkpoint` (full block gradient checkpointing: one layer's saved
activations, each layer forward re-run in backward), `fp16_activations`
(store attention probabilities as IEEE FP16), `recompute_attn` (regenerate
attention probs in backward), plus
`adafactor` (factored optimizer state), `seq_ramp_steps` (seq-length curriculum), `weight_qat_bits` (0/4/8 weight QAT), `swiglu_clamp` (V4-style gate/linear clamp; 10 default, 0 disables), legacy `qlora`, `recompute_swiglu` (recompute the FFN gate/up cache), and `kv_qat_bits`
(0 off, 4=int4, 2=ternary KV quantization-aware training, straight-through),
`kv_lowrank` (QAT-style Hadamard truncation rank for training).

## `[schedule]`

`kind` (`adaptive` or `cosine`), `tokens_per_param` (epoch formula),
`min_epochs`, `max_epochs`, `target_loss`, `plateau_window`, `plateau_delta`,
`min_steps_before_stop`, `ema_beta`. See `docs/FORMULAS.md` §12.

## `[lora]` (legacy, not in the default pipeline)

`rank`, `alpha` (scale = alpha/rank), `targets`
(array of `q_proj`, `k_proj`, `v_proj`, `o_proj`). The CLI/menu no longer
expose LoRA; this section remains for research and tests.

## `[dpo]`

`beta`, `ref_checkpoint` (frozen reference). If the configured reference is
missing, the base `paths.model_out` is used explicitly and reported; if neither
is valid the run fails rather than using random reference weights.

## `[ppo]` (legacy, not in the default pipeline)

`clip_ratio`, `kl_coef`, `rollouts`. PPO is no longer exposed; the supported
post-training objective is GRPO (sequence-level group baseline).

## `[grpo]`

`group_size`, `kl_coef`.

## `[reward]` (legacy, not in the default pipeline)

`kind` (`heuristic` or `model`), `checkpoint`, `steps`, `lr`. The learned
reward model is no longer trained by the CLI; GRPO runs with the heuristic
reward and the R1-style verifier path is a future experiment.

## `[rl]`

`max_new_tokens` per rollout, `temperature` (scales the rollout sampling
distribution only; policy log-probabilities and PPO ratios are evaluated at
T=1 so the importance ratio stays valid), reward shaping: `reward_format`,
`reward_length`, `length_target`, `reward_repeat`.

## `[inference]`

`max_new_tokens`, `system` prompt, `temperature`, `top_k`, `top_p`, `greedy`,
`kv_cache` (incremental decode), `kv_int8` (**default true**: 64 B/token,
~1.04× decode vs FP32, max logit error 4.5e-3), `kv_fp16` (F16C FP16 KV), `kv_q2` (ternary KV with `kv_recent`/`kv_sinks` FP32 mixed precision), `kv_q4` (packed 4-bit KV with per-head scales), `kv_rotate` (Hadamard-rotate K/V before q4), `weight_int8` (per-row
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
