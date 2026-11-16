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
│   ├── train.h              # pretrain/SFT/LoRA/DPO/PPO/GRPO entry points
│   ├── inference.h          # generation/chat
│   └── uenr.h               # umbrella include
├── src/
│   ├── core/                # arena, math, config, rng, io, thread
│   ├── ops/                 # gemm, norm, rope, attn, swiglu, softmax_ce, optim
│   ├── model/               # model.c (params, forward, backward, checkpoints)
│   ├── data/                # tokenizer.c, dataset.c
│   ├── train/               # schedule.c, train.c, align.c (DPO/PPO/GRPO)
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
| `q_norm`, `k_norm` | `[hd]` | per-head RMSNorm gains |
| `rms_ffn` | `[C]` | |
| `router` | `[E, C]` | |
| `expert_wg`, `expert_wu` | `[E, EF, C]` | each expert `[EF, C]` |
| `expert_wd` | `[E, C, EF]` | each expert `[C, EF]` |
| `shared_wg`, `shared_wu` | `[FF, C]` | shared/dense expert |
| `shared_wd` | `[C, FF]` | |

Plus `tok_emb [V, C]`, `rms_final [C]`, and (only if untied) `lm_head [V, C]`.
With the shipped `config/uenr.toml`: **62 tensors, 8,474,560 parameters**
(32.3 MiB in fp32), and **1,298,368 active parameters per token**.

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

`UMDL` format: 20-word header (magic, version, model dims, step, optimizer
flag, parameter count) followed by the full aligned parameter block and
optionally Adam `m`/`v`. Writes go to `<path>.tmp` then `rename()`, so a crash
never leaves a half-written checkpoint. Resume loads parameters and optimizer
state and continues from the saved step.
