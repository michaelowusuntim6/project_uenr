# UENR Architecture

UENR is a dependency-free C implementation of a tiny sparse-MoE transformer
trained from scratch on a laptop CPU. It does not link BLAS, PyTorch, Python,
or any third-party library. The only external calls are POSIX/Linux system
interfaces (`mmap`, `pthread`, file I/O) and the C runtime's allocator.

## Directory map

```
uenr/
├── Makefile                 # all / test / bench / check / clean
├── config/uenr.toml         # single source of truth for every setting
├── include/uenr/
│   ├── result.h             # Result-style error codes, UENR_TRY
│   ├── arena.h              # bump allocator
│   ├── simd.h               # AVX1-safe vector wrappers + F16C
│   ├── math.h               # FMA-free transcendental formulas
│   ├── rng.h                # xorshift128+
│   ├── io.h                 # read/write/mmap/atomic save/mkdirs
│   ├── thread.h             # persistent pool, CPU pinning
│   ├── config.h             # typed configuration + derived shapes
│   ├── gemm.h               # blocked SGEMM + linear wrappers
│   ├── ops.h                # RMSNorm, RoPE, attention, SwiGLU, CE, AdamW
│   ├── schedule.h           # epoch formula, LR curve, plateau stop
│   ├── tokenizer.h          # byte-level BPE + chat/tool template
│   ├── dataset.h            # uint16 corpus, SFT JSONL, preference pairs
│   ├── model.h              # parameter layout, activations, fwd/bwd
│   ├── train.h              # pretrain/SFT/DPO/GRPO entry points (LoRA/PPO legacy)
│   ├── inference.h          # generation/chat
│   └── uenr.h               # umbrella include
├── src/
│   ├── core/                # arena, math, config, rng, io, thread
│   ├── ops/                 # gemm, norm, rope, attn, swiglu, softmax_ce, optim
│   ├── model/               # model.c (params, forward, backward, checkpoints)
│   ├── data/                # tokenizer.c, dataset.c
│   ├── train/               # schedule.c, train.c, align.c (DPO/GRPO; PPO legacy)
│   ├── infer/               # inference.c
│   ├── main.c               # CLI
│   └── menu.c               # numbered interactive launcher
├── tests/                   # math/config/gemm/ops/tokenizer/gradcheck
├── bench/                   # cpubench, threadbench, gemmbench
├── docs/                    # this documentation set
├── artifacts/               # tokenizer.bin, vocab.txt, merges.txt
├── data/                    # corpus + train.bin/val.bin/SFT/pairs
└── checkpoints/             # model checkpoints + CSV log
```

## Parameter layout

All model weights live in **one contiguous, 64-byte-aligned block** (`params`).
A descriptor table (`UenrTensor`) records each tensor's offset, size, name, and
whether weight decay applies. Gradients, Adam moments, and any checkpoint all
reuse the same offsets, so `grad_of(p) = grad + (p - params)`.

Per layer:

| Tensor | Shape | Notes |
|---|---|---|
| `rms_attn` | `[C]` | RMSNorm gain, no decay |
| `wq` | `[C, C]` | row-major `[out, in]` |
| `wk`, `wv` | `[KV, C]` | `KV = kv_heads · head_dim` |
| `wo` | `[C, C]` | |
| `q_norm`, `k_norm` | `[hd]` | per-head RMSNorm gains (`k_norm` absent on borrowed layers) |
| `rms_ffn` | `[C]` | |
| `post_attn`, `post_ffn` | `[C]` | four-norm post-sub-layer gains (zero init) |
| `ls` | `[C]` | attention-branch LayerScale |
| `router` | `[E, C]` | |
| `expert_wg`, `expert_wu` | `[E, EF, C]` | each expert `[EF, C]` |
| `expert_wd` | `[E, C, EF]` | each expert `[C, EF]` |
| `shared_wg`, `shared_wu` | `[FF, C]` | shared/dense expert |
| `shared_wd` | `[C, FF]` | |

Plus `tok_emb [V, C]`, `rms_final [C]`, and (only if untied) `lm_head [V, C]`.
With the shipped `config/uenr.toml`: **68 tensors, 8,451,456 parameters**
(32.2 MiB in fp32), and **1,275,264 active parameters per token**.

Initialization: `N(0, 0.02)` for matrices, `1.0` for norm gains, residual output
projections (`wo`, expert/shared `down`) scaled by `1/sqrt(2L)`. A seeded
xorshift128+ makes runs reproducible.

## Activation layout

`UenrActs` is one set of buffers sized for a micro-batch `T = micro_batch·S`:
residual stream per layer, normed inputs, pre/post-RoPE Q/K, V, attention
probabilities, router probabilities, top-k indices/gates, per-token expert
outputs, logits, dlogits, and reusable backward scratch. Everything is
allocated once and reused for every step; there are no per-step allocations in
the training loop.

For the default config (`T = 1024`, `S = 256`, `L = 4`, `H = 6`, `V = 2048`)
the activation set is roughly 80 MiB. The attention probability buffer is the
largest term because it is `L·B·H·S²` (needed for exact backward).

Two memory modes trim that footprint without changing the math:

- `training.block_checkpoint = true` keeps saved activations for **one layer
  only** and re-runs `layer_forward()` for each layer immediately before its
  backward. At `seq_len=512` peak RSS falls 382→248 MB (1.54× less) at 0.81×
  training speed; the residual stream (`x[L+1]`) and donor K/V stay resident
  because later layers borrow them.
- `training.fp16_activations = true` stores attention probabilities as IEEE
  half (`uint16_t`, F16C conversions). Combined with block checkpointing at
  `seq_len=512`: 248→236 MB (12.6 MB less), 0.96× speed, and identical loss to
  four decimals. The FP16 forward stores capped scores, exponentiates in a
  second pass and normalizes in a third, so it performs the same number of
  dot products as the FP32 kernel.

## Forward pass

1. Gather token embeddings into `x[0]`.
2. For each layer:
   - `xa = RMSNorm(x, rms_attn)`;
   - `q_pre = xa Wq^T`, `k_pre = xa Wk^T`, `v = xa Wv^T`;
   - per-head RMSNorm on Q and K (rows `T·H` and `T·G`, width `hd`);
   - RoPE on Q and K;
   - causal GQA attention;
   - `x2 = x + attn_out Wo^T`;
   - `xf = RMSNorm(x2, rms_ffn)`;
   - router logits + softmax + top-k;
   - grouped SwiGLU over routed experts + shared expert;
   - `x = x2 + expert_out`.
3. Final RMSNorm, tied logits `xfinal · tok_emb^T`.

## Backward pass

Reverse order, reusing saved norm statistics and activations:

- fused softmax+CE writes `dlogits`;
- tied head: `dxfinal = dlogits · tok_emb`, `d_tok_emb += dlogits^T · xfinal`;
- final RMSNorm backward;
- per layer: expert/SwiGLU backward (recomputing gate/up), router softmax and
  auxiliary gradient, FFN RMSNorm backward, residual, output projection,
  attention backward, RoPE inverse, QK-norm backward, Q/K/V projection
  gradients, attention RMSNorm backward;
- embedding scatter for every token id in the batch.

Gradient accumulation averages micro-batches by scaling `dlogits` with
`1/(grad_accum · supervised_tokens)`. Global L2 clipping is applied before
AdamW.

## GEMM

`uenr_gemm` computes `C = alpha·op(A)·op(B) + beta·C` for all four
transpose combinations. The microkernel is a 4×8 AVX tile with four independent
accumulator chains (no FMA; each `c += a·b` is a mul and an add). Transposed-B
operands are packed into K-major panels (`[K][8]`) once per N-block. Rows of C
are split across the thread pool; the default pool uses the two physical cores
(CPU 0 and CPU 2) from the config. A GEMV path handles batch-1 decode.

Measured: ~15–25 GFLOP/s on 64–384 GEMMs (2 threads), versus ~6 GFLOP/s for the
naive reference and a measured 67.8 GFLOP/s two-core roof.

## Threading policy

Compute scales ×2 with the second physical core; DRAM bandwidth scales only
×1.30 on this machine. UENR therefore uses two threads **only for GEMM** (and
therefore attention projections, expert MLPs, and logits). Elementwise work,
RMSNorm, RoPE, softmax, and optimizers are single-threaded and vectorized. The
main thread is pinned to `run.pin_cpu0`; worker 1 is pinned to `run.pin_cpu1`.

## Checkpoints

`UMDL` format v2: 20-word header (magic, version, model dims, step, optimizer
kind, parameter count, extra-byte count) followed by the aligned parameter
block and optional optimizer state. Optimizer kinds: 1 = AdamW `m`/`v`,
2 = Adafactor factored state, 3 = Muon momentum, 4 = INT8-Adam `m8`/`ms`/`v`.
Writes go to `<path>.tmp` then `rename()`, so a crash never leaves a
half-written checkpoint. On load, a kind that no longer matches the config is
ignored and weights alone are restored.

Training writes `checkpoints/step_XXXXXX.bin` every
`run.checkpoint_every` steps and keeps the newest three. `SIGINT`/`SIGTERM`
finishes the current step and writes `paths.model_out` before exiting.
`run.resume=true` or `uenr train|sft --resume` reads the newest step file and
compares it with `model_out`, then continues from whichever has the larger
step. Optimizer state resumes too (warm Adam/Adafactor/INT8-Adam/Muon).


## Long-context / KV architecture (DOCUMENT.md items)

The shipped defaults implement the Gemma-4-style KV reduction stack in pure C,
scaled to this model (4 layers, C=192, hd=32, 6 query heads):

- **MQA + exact compensation.** `kv_heads=1`; `ffn` is widened to 277 and each
  layer gains a per-channel LayerScale so the item-49 checkpoint is exactly
  8,474,560 total / 1,298,368 active. Later mechanisms change the totals to
  8,451,456 / 1,275,264 (shared layers drop `wk/wv/k_norm`; four-norm adds two
  norms).
- **Layer plan.** `attn_pattern="SGSG"` cycles sliding/global types;
  `kv_shared_layers=2` makes layers 2–3 borrowers. `donor(l)` is the last
  active layer of the same type (0 for sliding, 1 for global). Borrowers have
  no `wk/wv/k_norm` tensors, predict only Q, and read the donor's cached
  post-norm/post-RoPE K and normalized V.
- **Hybrid attention.** Sliding layers use a ring buffer of `sliding_window`
  tokens (128 here); global layers cover the full context. Full-sequence
  training applies the `[max(0,t-W+1), t]` mask in the attention forward and
  backward; cached decode uses `pos % W` writes and `t % W` reads.
- **p-RoPE / norms.** NeoX split-half rotation, per-type local/global tables
  with independent theta and partial-rotary factors; the inverse-frequency
  denominator is the full head dimension. `RMSNormNoScale` is available for V
  (`v_norm`); QK-norm runs before RoPE. Measured smoke-scale A/B favored
  scaled attention (1/√d), no V-norm, and full global rotary, so those are the
  shipped defaults — flipping the config switches to the full DOCUMENT mode.
- **Softcaps.** Attention logits can be capped at 50 and final logits at 30
  with exact `1-tanh²` backward derivatives and a skip-small fast path.
- **Four-norm residual.** `x2 = x + ls·post_attn(W_o·attn)`,
  `x = x2 + post_ffn(FFN)`; post-norm gains start at zero and learn their
  scales. This was the largest quality win at smoke scale (val ppl 129.5→81.0
  at 100 steps).
- **Donor gradient accumulation.** Backward visits borrowers first; their dK/dV
  are summed into per-donor accumulators and applied once to the donor's
  K-norm/RoPE and `wk/wv` gradients. Covered by finite differences.
- **Compact KV cache.** Only donor layers own slots; local slots are `W` rows
  and global slots `max_seq`. Shared layers alias the donor's offset, capacity
  and INT8 scales. FP16 and INT8 caches are optional. End-to-end inference
  allocates `T=1` scratch when the KV path is active.

At the shipped shape the cache is 640 rows at S=512 (160 KiB FP32 versus
512 KiB for the old full GQA cache) and 1,000,128 rows at 1M tokens: 256 MB
FP32 / 128 MB FP16 / 64 MB INT8 / 40 MB packed Q4, versus 2.048 GB for the
original design. Eviction is off by default; optional StreamingLLM eviction
(sinks 4 + newest 256) keeps the 1M-context global slots at 388 rows (~24 KiB
INT8, ~0.1 MB FP32) and makes long-context decode 1.4–2× faster. With eviction
off, `kv_sparse_k=32` reduces global keys at long context (3.19× at 8192). Q4 uses per-head/per-row scales and AVX1 nibble unpacking;
it is an accuracy-preserving memory mode (max logit error 0.061, argmax 64/64)
but not a speed win on this CPU. The fixed 4096-entry score stack in decode was
replaced by a `cap`-sized cache buffer, so contexts above 4096 are safe.
