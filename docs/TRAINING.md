# Training

All training is plain C, one process, no framework. The loop lives in
`src/train/train.c` (pretrain/SFT/LoRA) and `src/train/align.c`
(DPO/PPO/GRPO).

## Quickstart

```bash
make -j2
./uenr tok-train          # ~ seconds; writes artifacts/
./uenr prepare            # tokenizes 16 MiB -> 4.05M tokens + SFT/pairs
./uenr train --steps 100  # pretrain
./uenr sft   --steps 50
./uenr infer --prompt "Once upon a time"
```

Or run `./uenr-menu` and choose the numbered action.

## Pretraining

- Data: `data/train.bin`, a `UDAT` file (magic, version, uint64 count, vocab,
  dtype, then uint16 ids). `val.bin` is held out by `prepare.val_percent`.
- Each optimizer step processes `grad_accum` micro-batches of
  `micro_batch = batch_size / grad_accum` sequences each.
- For each micro-batch: forward, fused softmax+CE, scale `dlogits` by
  `1/(grad_accum · supervised_tokens)`, backward. Gradients accumulate over
  the whole step.
- Global L2 norm is computed, clipped to `training.clip`, then AdamW updates
  every non-norm tensor. Norm gains use zero weight decay.
- The step budget comes from the epoch formula in `docs/FORMULAS.md` §12.

Observed on the target laptop (default config, 2 threads): ~870 tokens/s,
~8.8 GFLOP/s end-to-end, ~13% of the measured 2-core roof. A 100-step run
moved the loss from 7.52 to 6.61 (EMA 7.04); the first 20 steps were still
inside warmup, so most of the movement comes as the LR reaches 5e-4.

## SFT

- Data: `data/sft.jsonl`; `uenr_sft_load` builds chat-templated ids and
  assistant-only labels.
- Batch packing pads to `seq_len` with `pad_id` and `ignore_index` labels, so
  padding never contributes to the loss.
- The optimizer is the same; the model is initialized from
  `paths.model_out` (pretrained) when it exists.

## LoRA

`uenr lora` loads the base model, freezes every base tensor, and injects
low-rank adapters into the projections listed in `[lora].targets` (default
q/k/v/o):

```
y = x W^T + (alpha/rank) · (x A^T) B^T,   A: [r, in], B: [out, r]
```

`A` is initialized `N(0, 0.02)`, `B` is zero, so the adapted model starts
identical to the base. Backward computes `dA`, `dB`, and the adapter's `dx`;
base gradients are still formed but the optimizer skips them. On completion
the adapters are **merged into the base weights** and saved as an ordinary
checkpoint (`paths.lora_out`), so inference needs no adapter code.

## DPO

`uenr dpo` loads preference pairs and a frozen reference
(`dpo.ref_checkpoint`, falling back to `model_out`). For each pair it builds
`chat(prompt) + completion + <end>` for chosen and rejected, computes sequence
log-probs under policy and reference, and applies the DPO loss and gradient in
`docs/FORMULAS.md` §14. Only the policy is updated.

## PPO and GRPO

`uenr ppo` / `uenr grpo` sample completions from the current policy for a
prompt, score them with the configurable heuristic reward in `[rl]`, and update
with sequence-level policy gradients:

- **GRPO**: advantages are normalized inside the group of `group_size`
  completions; coefficient `-A + kl_coef`.
- **PPO**: advantage is reward minus the group mean; the probability ratio
  against the sampling policy is clipped by `ppo.clip_ratio`; coefficient
  `-A·ratio + kl_coef` inside the trust region, `kl_coef` outside.

Both use the frozen reference for the KL direction. This is deliberately the
smallest honest implementation: there is no learned value head, and the ratio
is sequence-level, not token-level. The formulas are documented so the next
iteration can extend them.

## Checkpoints and resume

- Every `run.checkpoint_every` steps: `checkpoints/step_XXXXXX.bin`, with
  parameters and Adam state.
- Final: `paths.model_out` (pretrain), `sft_out`, `lora_out`, `dpo_out`,
  `ppo_out`, `grpo_out`.
- `run.resume = true` resumes pretraining from `model_out` and the saved step.
- Writes are atomic (`.tmp` + `rename`).
- Format: `UMDL` magic, version, model dims, step, optimizer flag, parameter
  count, then the aligned parameter block and optional `m`/`v`.

## Logging

Every `run.log_every` steps the trainer prints and appends to
`checkpoints/train.csv`:

```
step, lr, loss, ema, tokens_per_s, flops_per_s, mfu_pct
```

FLOPs use `6·(active_params + V·C)` per token (forward + backward, tied head).
MFU is relative to the measured `33.9 GFLOP/s × threads` roof.

## Gradient-checked invariants

`tests/test_gradcheck.c` finite-differences every parameter family on dense and
single-expert models, including RMSNorm, RoPE, GQA attention, SwiGLU,
tied-head logits, and the shared/routed experts. Top-1 routing is piecewise
constant, so router gradients are derived analytically and the numerical check
uses smooth configurations.
