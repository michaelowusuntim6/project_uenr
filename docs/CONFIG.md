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
| `checkpoint_every` | Steps between periodic checkpoints (0 disables). |
| `checkpoint_dir` | Directory for periodic checkpoints. |
| `resume` | Resume pretraining from `paths.model_out` and its optimizer state. |

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
| `kv_heads` | 2 | KV heads (GQA group = heads/kv_heads). |
| `head_dim` | 32 | Must satisfy `heads·head_dim == hidden`. |
| `ffn` | 256 | Shared/dense expert width. |
| `rms_eps` | 1e-6 | RMSNorm epsilon. |
| `rope_theta` | 1e6 | RoPE base. |
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

`max_bytes` (raw bytes encoded by `uenr prepare`), `val_percent`, and the
synthetic SFT/pair generator settings (`sft_stories`, `sft_prompt_chars`).

## `[training]`

`mode` (label), `batch_size` (sequences per optimizer step), `grad_accum`
(micro-batches per step; `batch_size % grad_accum == 0`), `seq_len`,
`max_steps`, `lr`, `min_lr`, `warmup_steps`, `weight_decay`, `beta1`, `beta2`,
`adam_eps`, `clip`, `dropout` (reserved), `dtype` (storage is fp32 today).

## `[schedule]`

`kind` (`adaptive` or `cosine`), `tokens_per_param` (epoch formula),
`min_epochs`, `max_epochs`, `target_loss`, `plateau_window`, `plateau_delta`,
`min_steps_before_stop`, `ema_beta`. See `docs/FORMULAS.md` §12.

## `[lora]`

`rank`, `alpha` (scale = alpha/rank), `dropout` (reserved), `targets`
(array of `q_proj`, `k_proj`, `v_proj`, `o_proj`).

## `[dpo]`

`beta`, `ref_checkpoint` (frozen reference; falls back to `model_out`).

## `[ppo]`

`clip_ratio`, `value_coef` (reserved), `entropy_coef` (reserved), `gamma`,
`lambda` (reserved), `rollouts` (group size for PPO).

## `[grpo]`

`group_size`, `kl_coef`.

## `[rl]`

`max_new_tokens` per rollout, `temperature` (sampling uses 1.0 for ratio
correctness), reward shaping: `reward_format`, `reward_length`,
`length_target`, `reward_repeat`.

## `[inference]`

`max_new_tokens`, `system` prompt, `temperature`, `top_k`, `top_p`, `greedy`,
`kv_cache` (reserved; generation currently recomputes the full context).

## `[bench]`

`gemm_size` used by `uenr bench` when no `--size` is given.

## Validation

`uenr_config_validate` enforces: shape consistency
(`heads·head_dim == hidden`), `kv_heads | heads`, vocab range, sequence length,
thread bounds, positive widths, unique special ids in 0–15,
`batch_size % grad_accum == 0`, and positive schedule/reward values. A config
that fails prints an error and exits non-zero before any work starts.
