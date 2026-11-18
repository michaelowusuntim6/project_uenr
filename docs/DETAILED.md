# UENR — The Complete Technical and Operational Guide

> One file, ordered for a new collaborator. It explains what this project is,
> how it differs from a conventional transformer stack, every architectural
> mechanism, the measured results, how to build/train/serve it, and how to
> continue safely. It supersedes the overlapping introductions in the other
> docs; those files remain as focused references.
>
> Target machine: Intel i5-3210M (Ivy Bridge), 2 cores / 4 threads, AVX1 +
> F16C, **no FMA, no AVX2**, ~2.2 GHz sustained, 16 GB RAM class laptop.
>
> Language/runtime: C11 only. No PyTorch, Python runtime, BLAS, NumPy, CUDA,
> tokenizer library, or third-party dependency. POSIX/Linux (`mmap`,
> `pthread`, file I/O) and libc are the only external interfaces.

---

## 1. Read this first: where the project stands

UENR is a complete, dependency-free training and inference system for a tiny
sparse mixture-of-experts language model:

```
8,451,456 total parameters
1,275,264 active parameters per token
4 transformer layers, C = 192, 6 query heads, 1 KV head (MQA), head_dim 32
24 routed experts, top-1, expert width 128, plus a 277-wide shared expert
tied input/output embeddings, vocabulary 2,048
```

It includes:

- a handwritten byte-level BPE tokenizer with chat/tool-call markup;
- streaming corpus preparation into a compact `uint16` token format;
- pretraining, SFT, LoRA, QLoRA, DPO, PPO and GRPO trainers;
- a full KV-cached inference engine, batched and continuous serving, prefix
  caching, draft-model speculation, constrained JSON decoding;
- a numbered interactive launcher;
- unit, gradient, parity, and end-to-end tests;
- atomic checkpoints with optimizer state, SIGINT-safe saves, and resume.

Verified on the target laptop. Rows marked “earlier” below were measured in
earlier development sessions with their own checkpoints and eval windows; do
not combine them with the v0.1.0 validation rows.

| Metric | Value |
|---|---|
| Pretrain throughput | ~1.15–1.18k tok/s, Adafactor (v0.1.0 validation; an earlier A/B saw 1.23k with AdamW) |
| 100-step pretrain (v0.1.0 validation) | val loss **4.1572**, perplexity **63.89** (8×512 eval, 16,384 tokens) |
| Default training RSS | **215,748 KiB** one step with Adafactor (v0.1.0 validation); AdamW measured separately at 349.7 MB |
| 300-step stability run | loss 7.694→3.706, val ppl 36.98, no NaN |
| 128-token cached decode | ~1.0–1.1k tok/s |
| Batched decode, B=8 | 1,718 tok/s vs 538 sequential (**3.19×**) |
| Full serving stack | 0.77 s vs 2.56 s (**3.32×**) on 16 prompts with a 120-token shared prefix |
| KV at 1M tokens, default eviction | 388 rows ≈ **0.1 MB** FP32, 1.4–2× faster long decode |
| KV at 1M, no eviction | 256 MB FP32 / 128 MB FP16 / 64 MB INT8 / 40 MB Q4 |
| Inference RSS at 64k context | 150.8 MB (was 985.8 MB) |
| Block checkpointing | seq 512: RSS 382→248 MB (1.54× less), 0.81× speed |
| FP16 activation storage | with block: 236 MB (1.62× less than default), 0.96× block speed, identical loss |
| QLoRA low-RAM | 225→194 MB (1.16× less), speed unchanged |
| GEMM | ~15–25 GFLOP/s vs 67.8 two-core AVX roof |

The numbers are reproduced with commands in §13 and §14. The rest of this
document explains how each number is achieved and what its trade-offs are.

### 1.1 Recommended reading order for a collaborator

1. This document, §1–§4: project state, design philosophy, hardware ceilings.
2. §5–§8: model, attention/KV, MoE, training.
3. §9–§12: data/tokenizer, inference/serving, config, measured results.
4. §13–§16: operating instructions, tests, limitations, future work.
5. Source map, if you are changing code:
   - `src/model/model.c` — parameter layout, forward/backward, optimizer,
     checkpoints.
   - `src/model/decode.c` — KV cache, incremental decode, quantized KV.
   - `src/ops/gemm.c` — AVX1 SGEMM/GEMV kernels.
   - `src/ops/attn.c` — training attention and its FP16-prob variant.
   - `src/train/train.c` — generic training loop, checkpoints, resume, signals.
   - `src/infer/inference.c` — generation, speculative decoding, batching.
   - `src/menu.c` — interactive launcher.
   - `config/uenr.toml` — the single source of truth for every hyperparameter.

### 1.2 One-paragraph summary of the engineering bet

A conventional implementation assumes a GPU, a framework, dense attention and
dense FFNs, and accepts whatever memory the framework allocates. UENR starts
from the measured limits of one 2012 laptop CPU: 33.9 GFLOP/s per core,
67.8 GFLOP/s for two cores, and 13.4 GB/s DRAM. It then removes work rather
than adding hardware: one KV head with cross-layer sharing, a 128-token sliding
window for half the layers, top-1 MoE with a tiny shared expert, block-level
activation checkpointing, FP16 storage for the largest activation, eviction for
unbounded contexts, and hand-written AVX1 kernels. The result is a complete
LLM stack that trains in ~218 MB of RAM and serves a million-token context
with a cache smaller than this document's PDF.

---

## 2. What is different from a standard transformer/training stack

This section is the short answer to “why is this more efficient than the usual
implementation?” Each row is a concrete mechanism in this tree, not a
hypothetical.

| Area | Standard implementation | UENR | Effect |
|---|---|---|---|
| Runtime | Python + PyTorch/JAX, CUDA kernels | Single static C binary + libc/pthreads | No interpreter, no framework allocator, no dispatch overhead; RSS and latency are explicit |
| Linear algebra | BLAS/cuBLAS | 4×8 AVX1 microkernel, packed B panels, 2-thread pool pinned to physical cores | ~15–25 GFLOP/s on 64–384 GEMMs, 1.3× the first blocked kernel; no FMA required |
| Attention KV | Multi-head or GQA with 2·H·hd floats/token/layer | MQA (1 KV head) + cross-layer KV sharing (2 active slots for 4 layers) | 8× smaller than the original GQA layout, 256 B/token FP32 |
| Local attention | Dense causal mask | Sliding-window ring buffer on `S` layers, global on `G` layers | Local layers read 128 rows, not the whole context |
| Long context | KV grows without bound | StreamingLLM eviction: 4 sinks + newest 256 rows | 388 rows (~0.1 MB) at any length; 1.4–2× long decode |
| KV precision | FP16/FP8 | FP32 / FP16 / INT8 / packed Q4 / ternary, with QAT | Up to 10.7× smaller than FP32; Q4 is lossless on argmax in measurement |
| FFN | Dense FFN in every layer | Top-1 MoE (24 experts) + 277-wide shared expert | ~1.28M active params/token out of 8.45M total |
| Training memory | Store every layer's activations | Full block checkpointing: one layer's saved activations, layer forward re-run in backward | seq 512: 382→248 MB (1.54× less) |
| Activation precision | FP32 or BF16 everywhere | Attention probabilities stored as IEEE FP16 | 12.6 MB less at seq 512, 0.96× block speed, unchanged loss |
| Optimizer | AdamW (2 FP32 moments) | Adafactor factored state default; optional INT8 Adam moments, Muon | Default RSS 349.7→217.9 MB (1.60× less) and ppl 81.0→67.7, at 0.92× AdamW speed |
| LoRA/QLoRA | Frozen base still allocates grads + optimizer state | Frozen base gets a bounded grad sink and no optimizer state; QLoRA quantize-dequantizes the base to INT4 | 225→194 MB (1.16× less) at unchanged speed |
| Checkpoints | Whole-state blob; crash may lose the run | Atomic v2 files with optimizer kind, newest-3 rotation, SIGINT save, newest-by-step resume | A laptop shutdown loses at most one step, not the run |
| Serving | One sequence per forward | Lockstep batch + continuous batching + LCP prefix cache + draft speculation | 3.32× the lockstep baseline with a shared long prefix |
| Tokenizer | Python `tokenizers`/SentencePiece | Handwritten BPE trainer/encoder/decoder in C | 3.64 bytes/token; 16k-vocab recipe reaches 4.19 bytes/token |
| Math | libm transcendentals | Hand-written exp/log/sin/cos/sigmoid/SiLU/tanh/GELU/rcp/rsqrt | No libm calls in hot paths; AVX1-safe (128-bit integer halves) |
| Config | Constants in code | One TOML file; no hardcoded hyperparameters in C | Change the file, rebuild nothing |

The important point is not that any single row is novel. It is that the whole
stack is co-designed for one machine: every memory saving is paired with a
kernel that makes it useful, and every optional feature is measured and kept
off by default unless it wins.

---

## 3. Hardware audit and ceilings

Measured with `bench/cpubench` and `bench/gemmbench` on the target laptop:

| Quantity | Measured |
|---|---|
| Cores / threads | 2 physical / 4 logical |
| ISA | AVX1, F16C, SSE4.2; no FMA, no AVX2 |
| Sustained clock | ~2.2 GHz |
| AVX1 FP32 issue | 8 floats/instruction, 2 vector instructions/cycle theoretical |
| Measured roof, 1 core | **33.9 GFLOP/s** |
| Measured roof, 2 cores | **67.8 GFLOP/s** |
| DRAM bandwidth, 2 threads | **13.4 GB/s** |
| GEMM achieved, 64–384 | ~15–25 GFLOP/s |
| Decode regime | memory-bound |

Consequences that shaped the design:

- The Ivy Bridge path has no FMA, so every `a*b+c` is two instructions and
  kernels use independent accumulator chains to hide multiply/add latency.
  AVX2+FMA copies of the hot kernels are selected at runtime when available.
- In the AVX1 path 256-bit integer shifts are unavailable, so `exp` constructs
  `2^k` from two 128-bit halves and recombines with `vinsertf128`; the AVX2
  copy uses one 256-bit shift.
- The second core helps compute ~2× but DRAM only ~1.3×. Threading is used for
  GEMM only; norm/RoPE/softmax/optimizer stay single-threaded and vectorized.
- At decode, arithmetic intensity is ~1 FLOP/byte, so KV bytes dominate. This
  is why KV sharing, eviction, INT8/FP16/Q4 storage, and prefix reuse matter
  more than a wider GEMM tile.

The original audit of `karpathy/llm.c` and `michaelowusuntim6/minimind` is in
`docs/AUDIT.md`; the conclusion was that neither targets a two-core AVX1 CPU
with a hard no-dependency constraint, which motivated building this tree.

---

## 4. Design philosophy and invariants

1. **One source of truth.** Every shape, path, toggle and hyperparameter is in
   `config/uenr.toml` (plus alternate profiles `config/scale.toml`,
   `config/draft.toml`, `config/tokenizer16k.toml`). The C sources contain no
   model hyperparameters.
2. **Measure, then default.** Optional mechanisms are implemented behind a
   config flag, A/B measured, and left off if they lose. The negative results
   are documented in `docs/CHECK_LIST.md` and `docs/CHECK_LIST02.md` (for example, the
   6×16 GEMM tile and Muon-lite were reverted).
3. **Atomic state.** Checkpoints and artifacts write to `.tmp` then `rename()`.
   A crash never leaves a half-written file.
4. **Bounded memory at any context.** KV eviction keeps long-context memory
   constant; inference uses `T=1` scratch when the KV path is active.
5. **Exactness where it matters.** Cached decode is tested against full
   recompute; FP32 paths are bit-identical. Quantized paths are tested by logit
   error and greedy argmax agreement.
6. **Gradient-checked mechanisms.** Every architecture change is covered by
   finite differences (`tests/test_gradcheck.c`), including shared-KV donor
   accumulation, donor adapters, block checkpointing and FP16 probabilities.

---

## 5. Model architecture

### 5.1 Tensor layout

All weights live in one contiguous, 64-byte-aligned `float` block. A descriptor
table records each tensor's offset, length, name, and decay flag; gradients,
optimizer moments and checkpoint payloads reuse the same offsets, so
`grad_of(p) = grad + (p - params)` is O(1) and the checkpoint is a single
aligned copy.

Per layer:

| Tensor | Shape | Notes |
|---|---|---|
| `rms_attn` | `[C]` | pre-attention RMSNorm gain |
| `wq` | `[C,C]` | row-major `[out,in]` |
| `wk`, `wv` | `[KV,C]` | absent on shared (borrower) layers |
| `wo` | `[C,C]` | attention output projection |
| `q_norm`, `k_norm` | `[hd]` | per-head RMSNorm gains; `k_norm` absent on borrowers |
| `rms_ffn` | `[C]` | pre-FFN RMSNorm gain |
| `post_attn`, `post_ffn` | `[C]` | post-sub-layer norms, zero-initialized |
| `ls` | `[C]` | attention-branch LayerScale |
| `router` | `[E,C]` | |
| `expert_wg`, `expert_wu` | `[E,EF,C]` | SwiGLU gate/up |
| `expert_wd` | `[E,C,EF]` | down projection |
| `shared_wg`, `shared_wu` | `[FF,C]` | shared expert |
| `shared_wd` | `[C,FF]` | |

Plus `tok_emb [V,C]`, `rms_final [C]`, and `lm_head [V,C]` only when embeddings
are untied. The shipped config has 68 tensors, 8,451,456 parameters (32.2 MiB
FP32) and 1,275,264 active parameters per token.

### 5.2 The four-norm residual block

The shipped block is:

```
xa  = RMSNorm(x, rms_attn)
q,k,v = projections(xa) with QK-norm and RoPE
ao  = attention(q,k,v)
x2  = x + ls · post_attn(wo · ao)
xf  = RMSNorm(x2, rms_ffn)
e   = MoE(xf)
x'  = x2 + post_ffn(e)
```

`ls` is a learned per-channel LayerScale; `post_attn` and `post_ffn` are RMSNorm
gains initialized to zero, so each sub-layer starts as a no-op and learns its
own scale. This was the largest quality win at smoke scale: 100-step val ppl
129.5→81.0 (1.60× better) at ~1.00× speed.

Why four norms rather than the usual two? The extra post-sub-layer norm bounds
the branch output before it is added to the residual; with zero init, training
is stable without warmup tricks, and at this tiny scale it converges faster.
The cost is two extra `[C]` gain vectors per layer and one RMSNorm each — small
relative to the projections.

QK-norm is applied per head before RoPE. It keeps attention logits bounded and
was part of the MQA compensation package. V-norm (`v_norm`) is implemented but
off by default because the smoke-scale A/B did not favor it.

### 5.3 Attention

The shipped plan is `attn_pattern = "SGSG"`: layers 0 and 2 are sliding-window
local, layers 1 and 3 are global. `sliding_window = 128`. Attention is causal
GQA with a single KV head (`kv_heads=1`, `hd=32`, `KV=32`), six query heads
(group size 6).

- Attention scale is `1/sqrt(head_dim)` by default (`model.attn_scale=0`
  selects it); the config can override.
- Attention logits can be soft-capped at 50 (`attn_softcap`) and final logits at
  30 (`final_softcap`) with exact `1-tanh²` backward derivatives and a
  skip-small fast path. Measured effect at smoke scale: ppl 132.53→132.09,
  0.97× training / 0.95× evaluation. It is kept because it improves long-run
  numerical robustness at negligible cost.
- RoPE is split-half (NeoX style) with per-type local/global tables
  (`rope_theta_local=1e4`, `rope_theta_global=1e6`), independent partial-rotary
  factors, optional YaRN scaling and a sequence-length curriculum.
- A flash/online-softmax forward-backward pair exists
  (`model.flash_attn=true`) and avoids the `S²` probability buffer, but the
  saved-probability path is used by default because at these sizes it is
  faster and simpler to checkpoint.

### 5.4 Cross-layer KV sharing

`kv_shared_layers=2` makes layers 2 and 3 **borrowers**. Each borrower has no
`wk`, `wv`, or `k_norm`; it computes only Q and reads the donor's cached
post-norm/post-RoPE K and normalized V. `donor(l)` is the last active layer of
the same attention type (layer 0 for sliding, layer 1 for global in SGSG).

Two consequences:

- Active parameter count drops and the KV cache has two slots instead of four.
- Backward must accumulate borrower dK/dV into the donor. The implementation
  sums them in `dk_acc`/`dv_acc` and applies them once to the donor's K-norm,
  RoPE and `wk`/`wv` gradients. This is finite-difference tested
  (`donor-accum`, `donor-adapter`).

Optional donor adapters (`model.kv_adapter=true`) give each borrower a learned
per-dimension scale/shift on the borrowed K/V (`ka,kb,va,vb`). They cost a few
hundred parameters and were measured to recover part of the quality lost by
sharing (ppl 78.16 vs 78.84 without adapters in the aggressive-sharing test).
They are off by default.

A more aggressive configuration (share three layers, one KV slot) is
implemented and measured (100-step ppl 78.16 with adapters vs 81.03 baseline),
but long-context behavior is mixed, so it ships as an option.

### 5.5 Sparse MoE

Default: 24 experts, top-1, expert width 128, plus a 277-wide dense shared
expert. For each token:

```
r = xf · router^T
p = softmax(r)
idx, gate = top-1(p)
y = Σ gate · expert_idx(xf) + shared_expert(xf)
```

Dispatch groups tokens by selected expert (count → prefix sum → per-expert token
list), runs one contiguous SwiGLU per expert, and scatters `gate · y` back.
Backward recomputes the grouping from saved indices, computes expert gradients
from `gate · dy`, and derives the router softmax gradient analytically.
An optional Switch-style load-balance auxiliary loss
(`router_aux_coef=5e-4`, gentle at this scale) is included.

At this scale top-1 is the right choice: active compute is ~1.28M
parameters/token, grouping is deterministic and cache-friendly, and total
capacity scales by adding experts with almost no per-token FLOPs. Top-k > 1 is
implemented and validated but is not the shipped configuration.

Why is the shared expert 277 wide, an odd number? It is the width that makes
the post-MQA-compensation model hit the exact target parameter count while
keeping the KV head at one; it is not a hardware constraint. Every dimension is
a config value.

---

## 6. Long-context and KV engineering

This is the part of the system with the largest measured wins.

### 6.1 Compact donor-slot cache

Only donor layers own cache slots. A local slot has `sliding_window` rows; a
global slot has `max_seq` rows. Borrowers alias the donor's offset, capacity,
and quantization scales. At the shipped shape:

| Context | Rows | FP32 | FP16 | INT8 | Q4 |
|---|---|---|---|---|---|
| S=512 | 640 | 160 KiB | 80 KiB | 40 KiB | 20 KiB |
| 1M, no eviction | 1,000,128 | 256 MB | 128 MB | 64 MB | 40 MB |
| 1M, default eviction | **388** | **~0.1 MB** | ~0.05 MB | ~0.025 MB | ~0.012 MB |

The original GQA design (6 KV heads, 4 active layers) needs 2,048 B/token =
2.048 GB at 1M. The shipped FP32 design is 256 B/token, an 8× reduction;
INT8/Q4 reach 32×/51×. Eviction makes the cache constant-size.

### 6.2 Sliding-window ring buffers

Local layers write at `pos % window` and read the newest `window` rows. This is
exact for a causal sliding-window layer and needs no shifting or copying.
Global layers keep the full `max_seq`. During training, the mask is
`[max(0,t-W+1), t]`; forward and backward apply it directly.

### 6.3 Eviction and attention sinks

`inference.kv_evict = true` is the default. For global slots, the cache keeps:

- the first `kv_sinks = 4` rows (attention sinks),
- the newest `kv_evict_window = 256` rows,
- and discards everything between them.

Because position is tracked separately, eviction does not corrupt RoPE. The
measured effect on held-out perplexity was neutral-to-positive:

| Context | Without eviction | With eviction | Decode speed |
|---|---|---|---|
| 512 | 34.25 | 34.15 | 1.00× |
| 2048 | 33.60 | 33.11 | 1.41× |
| 4096 | 52.88 | 52.17 | 1.96× |

The cache is 388 rows at 1M versus 2,600× more without eviction. This is why
the user-visible “KV at 1M is less than half a megabyte” result holds.

### 6.4 KV precision modes

| Mode | Storage | Measured accuracy | Decode speed | Status |
|---|---|---|---|---|
| FP32 | 100% | exact / bit-parity | 1.00× | default |
| FP16 | 50% | within FP16 tolerance | ~1.0× | option |
| INT8 | 25% | max logit err 4.5e-3 | 1.04× | option |
| Q4 + Hadamard | 15.6% | QAT-4 mean err 0.0136, argmax 64/64 | 0.77–0.83× | memory mode |
| Ternary QAT | 9.4% | mean err 0.039, argmax 63/64 | 0.14–0.32× | prototype |

Q4 stores four signed bits with one FP32 scale per KV head per row; decode
unpacks nibbles with AVX1 integer/float operations. It is an accuracy-preserving
memory mode but not a speed win on this CPU, whose memory-bound decode does not
benefit from the extra unpack instructions. FP16 is the best bandwidth
compromise; FP32 remains fastest at short context.

QAT matters: quantization-aware training (`training.kv_qat_bits = 4` or `2`)
quantize-dequantizes K/V in the forward with a straight-through gradient,
bringing Q4 from argmax 61/64 to 64/64 and roughly halving mean logit error.
The Hadamard rotation is orthogonal (add/sub only) and reduces outliers.

### 6.5 Low-rank latent KV

`inference.kv_lowrank` and `training.kv_lowrank` keep only the first N Hadamard
coefficients of each K/V row. Rank 8 gives a 4× smaller cache (64 MB vs 256 MB
at 1M) with cached perplexity 83.17 vs 83.60 at 2048 and 121.10 vs 121.41 at
4096 on the same adapted model, and ~1.02× decode. It requires low-rank-QAT
training and is off by default. Rank 16 gives 2× compression with ppl 85.4.

### 6.6 Sparse decode

With eviction disabled, `inference.kv_sparse_k = 32` adds 32 deterministic
random global keys per query on top of sinks and recent rows. Measured 1.38×
at 2048 and 3.19× at 8192 with slightly better perplexity. With eviction on
(the default) it is inert, because the global slot already has only 388 rows.

### 6.7 Block checkpointing and FP16 activations

Two training-memory mechanisms complete the memory story:

- **Full block gradient checkpointing** (`training.block_checkpoint=true`):
  only the residual stream `x[L+1]` and donor K/V are kept per layer. The
  layer forward is extracted into `layer_forward()` and re-run immediately
  before each layer's backward, filling one shared activation slot. At
  `seq_len=512`, peak training RSS falls 382→248 MB (1.54× less) at 0.81×
  speed. Donor K/V stays per-layer because borrowers read it.
- **FP16 activation storage** (`training.fp16_activations=true`): attention
  probabilities — the largest saved activation, `slots·B·H·S²` — are stored as
  IEEE half. The FP16 forward writes capped scores, exponentiates and
  normalizes in separate passes, so it performs the same number of dot
  products as the FP32 kernel. Combined with block checkpointing at seq 512:
  236 MB (1.62× less than default), 0.96× of block speed, loss identical to
  four decimal places.

---

## 7. Training system

### 7.1 Loop

The generic loop (`src/train/train.c`) serves pretraining, SFT, LoRA, and the
alignment methods:

1. Fetch a micro-batch of `micro = batch_size / grad_accum` sequences.
2. Forward, cross-entropy (optionally sampled softmax, z-loss), write
   `dlogits`.
3. Scale `dlogits` by `1/(grad_accum · supervised_tokens)` and backward.
4. After all micro-batches: global L2 gradient norm/clipping, optimizer step,
   optional weight QAT, EMA.
5. Log/CSV every `run.log_every`; checkpoint every `run.checkpoint_every`.

Loss and gradient math use hand-written AVX1 kernels; there is no per-step
allocation.

### 7.2 Schedule

- Adaptive cosine LR with warmup (`training.lr`, `min_lr`, `warmup_steps`).
- Custom epoch formula: `epoch = tokens_per_param · active_params / corpus_tokens`
  (`[schedule]`), clamped by `min_epochs`/`max_epochs`.
- Plateau early stop on an EMA of loss (`ema_beta`, `plateau_window`,
  `plateau_delta`, `min_steps_before_stop`).
- Optional sequence-length curriculum: `seq_ramp_steps` ramps from
  `training.seq_len` to `model.max_seq`, with YaRN/RoPE scaling. Measured
  1024-context ppl 34.45→33.93 and 2048 37.72→36.82, enabling 4× context.
- Optional EMA weights saved separately (`paths.ema_out`).

### 7.3 Optimizers

- **Adafactor** (`training.adafactor=true`, default): factored row/column
  second-moment state for matrices and full state for vectors. In an earlier
  development A/B at 100 steps: ppl **67.7** vs 81.0 AdamW, peak RSS
  **217.9 MB** vs 349.7 MB (1.60× less), speed **1130 vs 1229 tok/s (0.92×)**.
  These predate the v0.1.0 validation run and use a different eval window. This is the default because the
  memory and quality wins dominate a 8% throughput loss on a laptop.
- **AdamW** (`adafactor=false`, `optim8=false`): full FP32 `m`/`v`.
- **INT8 Adam** (`optim8=true`): blockwise INT8 first moment plus FP32 second
  moment. Optimizer state 64→40 MB (1.6×), RSS 349.6→259.4 MB (1.35×), ppl
  81.27.
- **Muon-lite** (`muon=true`): normalized momentum; measured ppl 413 vs 67.7,
  so it is off and kept only as a documented negative result.
- Weight QAT (`weight_qat_bits=4/8`): quantize-dequantize master weights each
  step. INT4 QAT at this scale degraded ppl to 169, so it is off; INT8-era
  results are preserved in the checklists.

### 7.4 LoRA and QLoRA

LoRA freezes the base and trains low-rank adapters on Q/K/V/O. Rank, alpha and
target list are in `[lora]`; adapters merge into the base at the end
(`uenr_model_lora_merge`), and the merge correctly skips `wk`/`wv` on shared
layers (a real bug found by crash testing).

The low-RAM change: when the base is frozen, UENR does **not** allocate the
full base gradient buffer or any base optimizer state. Every base-weight
gradient write is routed by `grad_of()` to a single bounded `grad_sink` that is
never read; `zero_grad` clears it. LoRA parameters keep their own small
`lora_m`/`lora_v`. Result: LoRA/QLoRA RSS 225→194 MB (1.16× less) at unchanged
speed. QLoRA additionally quantize-dequantizes the frozen base to INT4 before
adapter training (`training.qlora=true`), simulating a quantized base; in a
50-step A/B its EMA loss was 3.8067 vs 3.8380 for plain LoRA. A future step is
to store the base itself in packed INT4 and free the FP32 copy, which would
save another ~27 MB but needs a batched quantized training forward.

### 7.5 Alignment methods

DPO, PPO and GRPO are implemented with a trainable heuristic reward model:

- **DPO**: logistic preference loss against a frozen reference checkpoint.
- **GRPO**: group-normalized advantages over `group_size` sampled completions.
- **PPO**: sequence-level ratio `r = exp(logπ_new − logπ_old)` and the clipped
  surrogate `−min(r·A, clamp(r,1−ε,1+ε)·A)`, with ε = `ppo.clip_ratio`. The
  unclipped term is the minimum for A>0 when r ≤ 1+ε and for A<0 when
  r ≥ 1−ε; gradient is zero only where clipping binds.
- **KL**: both PPO and GRPO use the frozen reference and the single-sample k1
  estimator `KL ≈ logπθ − logπref`; its policy-log-prob gradient is +kl_coef.

These are deliberately minimal but real implementations (no value head,
entropy bonus or GAE; sequence-level rather than token-level ratios). The
exact objective, sign conventions and clipping regions are documented in
`docs/FORMULAS.md` §15 and regression-tested in `tests/test_align.c`.

### 7.6 Checkpoints, crash safety and resume

`UMDL` v2 format:

```
header: magic, version, model dims, step, optimizer kind, param count,
        extra-byte count
body:   aligned parameter block
extra:  optimizer payload (kind 1 AdamW m/v; 2 Adafactor; 3 Muon;
        kind 4 INT8-Adam m8/ms/v)
```

Behavior:

- Every `run.checkpoint_every` steps (default 500) writes
  `checkpoints/step_XXXXXX.bin`; the newest three are kept.
- `SIGINT`/`SIGTERM` sets a flag; training finishes the current step, then
  writes `paths.model_out` and exits cleanly.
- `run.resume=true` or `uenr train|sft --resume` reads the newest
  `step_*.bin` and compares its header step with `paths.model_out`, then loads
  whichever is newer. Optimizer state is restored when the kind matches.
- All writes are atomic (`.tmp` + `rename`).
- Verified: a run interrupted at step 3 saved `model.bin` at step 3; a
  subsequent `--resume` continued from step 4 with warm state.

Old checkpoints from before the MQA/KV-sharing shape change are rejected by
the shape check; retrain.

---

## 8. Data and tokenizer

### 8.1 Tokenizer

Handwritten byte-level BPE:

- training on a configurable prefix (`tok.train_bytes`, 8 MiB by default),
  pretokenized on whitespace/punctuation, with `min_pair_freq`;
- heap-based merge selection for encoding;
- byte fallback, so every input is representable;
- NFC-lite composition for Latin base + combining marks and CamelCase
  splitting in letter runs (16k recipe: 4.19 vs 3.64 bytes/token, 1.15× fewer
  tokens, 0.91× encode speed, 126 KB artifact);
- special tokens for system/user/assistant/tool-call/tool-result/end, preserved
  atomically, enabling JSON tool calls;
- decode of arbitrary bytes is not guaranteed valid UTF-8; the display
  sanitizer is a roadmap item.

### 8.2 Data preparation

`uenr prepare` streams the raw corpus, applies the tokenizer, writes
`data/train.bin`/`val.bin` in a compact `UDAT` format (header + `uint16`
tokens), builds SFT JSONL and preference pairs, and supports hash-based
deduplication (`prepare.dedup`, `dedup_slots`). The training data is `mmap`ed,
so the 106 MB token file is reclaimable page cache rather than anonymous
memory; batches are random-access over the map.

The shipped corpus in this workspace is TinyStories. The default
`prepare.max_bytes = 209715200` (200 MiB) streams that slice line by line; the
1.8 GB source is never loaded whole. Tokenizer training has its own separate
8 MiB sample (`tokenizer.train_bytes = 8388608`).
Dedup measured 0 duplicates in a 200 MB sample and 17.13 vs 16.85 s prepare
(0.98×).

---

## 9. Inference and serving

### 9.1 Single-sequence decode

With `inference.kv_cache=true` (default), prompt tokens are prefilled into the
compact cache, then each generated token performs one incremental step:
project Q/K/V, append K/V at the current position, attend over the valid rows,
run the MoE block, emit logits. The cache stores only donor slots and uses ring
buffers for local layers.

Measured ~1.0–1.1k tok/s for 128-token decode, versus ~65 tok/s for the
full-context reference. A test compares cached steps, batched spans and
multi-sequence decode against full recompute; FP32 is bit-identical and
FP16/INT8 match within tolerance.

### 9.2 Sampling and constraints

- greedy or temperature sampling;
- top-k, top-p (nucleus), min-p;
- repetition penalty over the last N tokens, no-repeat-ngram;
- optional JSON grammar for tool calls: candidate tokens are decoded and
  accepted only if braces/brackets stay balanced and strings well-formed;
  invalid candidates are masked and resampled; generation can end only after
  the value closes;
- stop on `<eos>`/`<end>`.

Measured: repetition penalty + no-repeat-3 cut repeated word trigrams from
2.68% to 0.00% at 1.00× decode cost.

### 9.3 Batched inference

`uenr infer --batch prompts.txt` reads up to `batch_max` prompts, builds one
compact cache per sequence, and decodes in lockstep with batched projections
and grouped MoE tokens. B=8 measured 1,718 tok/s vs 538 sequential (3.19×);
B=4 gives 1.17× because top-1 experts diverge across sequences.

### 9.4 Continuous batching and prefix cache

`inference.continuous=true` streams queued prompts into freed slots instead of
running fixed waves. The LCP prefix cache finds the longest common prefix
across the batch (typ. system prompt), prefills it once, and copies the prefix
rows into each compact cache (`uenr_kv_cache_copy_prefix`). With 16 prompts
sharing a 120-token system prefix, end-to-end time was 0.77 s vs 2.56 s
(**3.32×** a lockstep baseline) with identical outputs. Without a long shared
prefix the gain is 1.51×; the prefix cache matters most for chat/system-prompt
serving.

### 9.5 Speculative decoding

Two mechanisms:

- **Prompt lookup** (`inference.speculative`): propose continuations found in
  the prompt, verify with a batched `uenr_model_forward_span`, accept the
  longest matching prefix. Output is exactly greedy.
- **Draft model** (`inference.draft_model` + `draft_config`): a tiny 357,728-
  parameter 2-layer dense model proposes `draft_k` tokens; the main model
  verifies them in one span pass. Measured ~70% acceptance and token-identical
  output, but 0.47–0.85× decode on the 8.45M model because verification
  overhead exceeds the savings.

The draft is therefore off in the default config and enabled in
`config/scale.toml`, where the larger main model's verification GEMs amortize
the draft cost. This directly answers the “enable it when I scale parameter
size” requirement.

### 9.6 Weight quantization

INT8 and INT4 weight-only inference are implemented (`inference.weight_int8`,
`weight_int4`) with per-row scales and AVX1/SSE4.1 GEMV kernels. INT8 measured
1.15× decode with 4.7e-4 max logit error. INT4 gives 7.3× smaller quantized
copies and 0.92× decode with 1.05e-2 logit error on the test model. They are
memory modes, not speed modes, on this CPU.

### 9.7 Early exit

`inference.early_exit` skips the final shared layer for tokens whose input
changes very little (`early_exit_delta`). At delta 0.5 it skipped 62% of
tokens and gave 1.12× decode at +11.6% perplexity; delta 1.0 skipped all and
gave 1.29×. It is off by default because the quality cost is real.

---

## 10. Fast math, GEMM and threading

### 10.1 Transcendentals

All hot-path math is hand-written and AVX1-safe (see `docs/FORMULAS.md` for
derivations):

- `exp`: range reduction to `r ∈ [-ln2/2, ln2/2]`, degree-6 polynomial,
  `2^k` built from two 128-bit integer halves;
- `log`: exponent/mantissa split, atanh series to degree 11;
- `sin`/`cos`: quadrant reduction plus degree-9/8 polynomials, used to build
  RoPE tables once at init;
- `sigmoid` = reciprocal + Newton, `SiLU`, `tanh` via sigmoid, GELU tanh
  approximation;
- `rcp`/`rsqrt`: `vrcpps`/`vrsqrtps` plus one Newton step;
- `sqrt` uses `vsqrtps` where the input can be exactly zero (AdamW); the
  `x·rsqrt(x)` shortcut is used only for strictly positive norm denominators.

The AdamW NaN bug (`v·rsqrt(v)` with `v=0`) was found by the checkpoint NaN
audit and is now prevented.

### 10.2 GEMM

`uenr_gemm` computes `C = alpha·op(A)·op(B) + beta·C` for NN/NT/TN/TT. The
microkernel is a 4×8 AVX1 tile with four independent accumulator chains,
packed transposed-B panels, and row-split threading. A GEMV path handles
batch-1 decode. A 6×16 tile was implemented and measured slower (17–19 vs
19–35 GFLOP/s) and is compiled only under `-DUENR_6X16`; the 8×8 path is the
default. INT8 dot products use `maddubs`/`madd`/`sign_epi8` with 32-bit
accumulation where applicable.

### 10.3 Threading

Two threads are used only for GEMM (projections, expert MLPs, logits). Norm,
RoPE, attention scores, softmax and optimizers are single-threaded and
vectorized. The main thread is pinned to `run.pin_cpu0`, worker 1 to
`run.pin_cpu1`. A second oversubscribed pool (`run.smt_threads`) is optional
for memory-bound work but defaults to 1 because SMT contention hurts more than
it helps.

---

## 11. Configuration guide

Everything is in `config/uenr.toml`. The most important fields for a new user:

### Model

| Key | Shipped | Meaning |
|---|---|---|
| `vocab_size` | 2048 | must match the tokenizer |
| `hidden` | 192 | residual width |
| `layers` | 4 | |
| `heads` | 6 | query heads |
| `kv_heads` | 1 | MQA |
| `head_dim` | 32 | |
| `ffn` | 277 | shared/dense expert width |
| `kv_shared_layers` | 2 | number of borrowers at the end |
| `attn_pattern` | SGSG | sliding/global per layer |
| `sliding_window` | 128 | local ring size |
| `attn_softcap` / `final_softcap` | 50 / 30 | logit caps; 0 disables |
| `kv_adapter` | false | donor scale/shift adapters |
| `v_norm` | false | optional V RMSNorm |
| `max_seq` | 512 | training/eval cap; inference cache capacity |
| `flash_attn` | false | online-softmax path |

### Training

| Key | Shipped | Meaning |
|---|---|---|
| `batch_size`, `grad_accum`, `seq_len` | 8, 2, 256 | micro = 4 sequences |
| `lr`, `min_lr`, `warmup_steps` | 1e-3, 5e-5, 20 | cosine schedule |
| `adafactor` | true | factored optimizer state |
| `optim8` | false | INT8 first moment |
| `block_checkpoint` | false | full block checkpointing |
| `fp16_activations` | false | FP16 attention probabilities |
| `recompute_attn` | false | recompute only probabilities |
| `recompute_swiglu` | false | recompute FFN caches |
| `seq_ramp_steps` | 0 | sequence-length curriculum |
| `kv_qat_bits` | 0 | 4/2-bit KV QAT |
| `kv_lowrank` | 0 | low-rank KV QAT |
| `weight_qat_bits` | 0 | 4/8-bit weight QAT |
| `qlora` | false | INT4 frozen base for LoRA |
| `muon` | false | normalized-momentum experiment |

### Run / checkpoints

| Key | Shipped | Meaning |
|---|---|---|
| `checkpoint_every` | 500 | periodic atomic checkpoint; newest 3 kept |
| `checkpoint_dir` | checkpoints | step files |
| `resume` | false | also `--resume` |

### Inference

| Key | Shipped | Meaning |
|---|---|---|
| `kv_cache` | true | incremental decode |
| `kv_evict` | true | sinks 4 + newest 256 |
| `kv_evict_window` | 256 | recent rows |
| `kv_sparse_k` | 32 | random keys when eviction is off |
| `kv_int8`, `kv_fp16`, `kv_q4`, `kv_q2` | false | KV precision |
| `prefix_cache` | true | LCP prefix reuse |
| `continuous` | true | continuous batching |
| `batch_max` | 8 | prompts per batch |
| `draft_model` | "" | empty = off; scale profile enables it |
| `draft_config`, `draft_k` | draft.toml, 4 | draft model settings |
| `speculative`, `spec_k`, `spec_ngram` | false, 4, 3 | prompt lookup |
| `json_constrained` | true | tool-call JSON grammar |
| `repetition_penalty`, `no_repeat_ngram` | 1.1, 3 | constraints |

Full field-by-field documentation is in `docs/CONFIG.md`; alternate profiles:

- `config/scale.toml` — 64,823,328 total / 9,183,264 active, 8 layers, C=384,
  32 experts, block checkpointing on, draft model enabled.
- `config/draft.toml` — 357,728-parameter, 2-layer, C=96 dense draft.
- `config/tokenizer16k.toml` — 16k-vocab recipe (retrain tokenizer and model).

---

## 12. Verified measurements

All values below were produced on the target laptop; the protocol is peak RSS
from `/usr/bin/time -v`, held-out val loss/perplexity, and token throughput
from the built-in logs. Negative results are retained.

### 12.1 Quality and stability

These are earlier development measurements with their own checkpoints and eval
windows; they are not directly comparable with the v0.1.0 release-validation
numbers in the headline table.

| Experiment | Result |
|---|---|
| 100-step pretrain, AdamW | val loss 4.394, ppl 81.03 |
| 100-step pretrain, Adafactor | val loss 4.215, ppl **67.7** |
| 300-step stability run | loss 7.694→3.706, val loss 3.610, ppl 36.98, no NaN |
| Four-norm vs two-norm | ppl 129.5→81.0 at 100 steps (1.60× better) |
| Softcaps | ppl 132.53→132.09; 0.97× train / 0.95× eval |
| MQA KV-head reduction | KV/token halved; exact param bookkeeping via ffn 277 + LayerScale |
| Cross-layer sharing | params 8,474,560→8,449,920 at that stage; training 1.04×; donor grads FD-checked |
| QLoRA 50-step A/B | EMA 3.8067 vs 3.8380 (plain LoRA) |

### 12.2 Memory and speed

Earlier development measurements (different code states/configs); the v0.1.0
validation RSS is in the headline table.

| Item | Measurement |
|---|---|
| Default training RSS | 217.9 MB (Adafactor) / 349.7 MB (AdamW) |
| Block checkpointing, seq 512 | 382→248 MB, 0.81× speed |
| FP16 activation storage + block | 236 MB, 0.96× block speed, identical loss |
| LoRA/QLoRA low-RAM | 225→194 MB, unchanged speed |
| Full block+fp16 vs default | 1.62× less RSS, 0.78× speed |
| Inference RSS at 64k | 150.8 MB (was 985.8 MB) |
| Decode 128 tokens | ~1.0–1.1k tok/s |
| Batched B=8 | 3.19× sequential |
| Serving stack | 3.32× lockstep with shared 120-token prefix |
| Eviction decode | 1.00× @512, 1.41× @2048, 1.96× @4096 |
| Sparse decode (evict off) | 1.38× @2048, 3.19× @8192 |
| GEMM | 15–25 GFLOP/s; 6×16 tile slower and disabled |
| INT8 KV | 0.95–1.04× decode, 4× smaller |
| Q4 KV | 0.77–0.83× decode, 6.4× smaller, argmax 64/64 after QAT |
| Ternary KV | 0.14–0.32× decode, 10.7× smaller, argmax 63/64 |

### 12.3 What the KV numbers mean

At 1M tokens:

- Original GQA design: 2.048 GB.
- Shipped MQA + sharing: 256 MB FP32 (8× less).
- FP16: 128 MB; INT8: 64 MB; Q4: 40 MB (32×/51× less).
- Default eviction: 388 rows ≈ 0.1 MB (≈2,600× less than the no-eviction
  FP32 cache), and long-context decode is 1.4–2× faster.

This is the engineering result the user singled out: a million-token context
served from a cache smaller than half a megabyte, without retraining the model
to a different attention pattern.

---

## 13. Build, test and run

### 13.1 Build

```bash
cd /home/mike/UENR/uenr
make -j2                 # warning-free with gcc -O3 -march=native -mavx -mf16c
make test                # all test binaries
make check               # build + tests + benchmarks
```

Compiler flags intentionally do not use `-ffast-math`. Normal objects build
for `-march=ivybridge`; `*_avx2.c` objects build with `-march=x86-64-v3
-mavx2 -mfma` and are reached only through the runtime dispatch layer.

### 13.2 End-to-end

```bash
./uenr tok-train
./uenr prepare
./uenr train --steps 100
./uenr train --resume                 # after an interrupt/shutdown
./uenr eval
./uenr infer --prompt "Once upon a time" --tokens 128 --greedy
./uenr infer --chat
./uenr infer --batch prompts.txt --tokens 20 --greedy
./uenr sft   --steps N
./uenr lora  --steps N
./uenr dpo|ppo|grpo --steps N
./uenr bench --size 256
./uenr --config config/scale.toml config
./uenr --config config/scale.toml train
```

Checkpoints are written to `checkpoints/` (gitignored). The tokenizer artifacts
are in `artifacts/`, prepared data in `data/`.

### 13.3 Interactive launcher

`./uenr-menu` is a single C program with 23 numbered actions: environment
check, build, tests, full check, benchmarks, tokenizer, prepare, pretrain, SFT,
LoRA, reward model, DPO, PPO, GRPO, chat, batch inference, eval, KV calculator,
config, docs, status, clean, exit. Pretrain and SFT now ask “Resume newest
checkpoint?”. It returns to the menu after every action and survives missing
files or failed children. Piped self-test:

```bash
printf '18\n\n20\n\n22\nn\n\n23\n' | ./uenr-menu
```

### 13.4 Tests

- `test_config` — parser, defaults, shape validation.
- `test_gemm` — GEMM vs reference for all transpose modes.
- `test_math` — transcendental accuracy.
- `test_ops` — window masks, softcaps, V-norm, partial RoPE.
- `test_gradcheck` — finite differences: dense, MoE, shared-KV,
  donor-accumulation, donor-adapter, recompute-attn, block-checkpoint,
  block+donor-adapter, block+fp16-probs.
- `test_kv_cache` — FP32/INT8/FP16/Q4/rotate/Q2/shared/span/multi, INT4 weights.
- `test_infer_constraints` — JSON grammar and penalties.
- `test_optim8` — INT8 Adam vs FP32.
- `test_sampled_softmax`, `test_tokenizer` — sampled CE and NFC/camel casing.

`make check` must stay green before every commit.

---

### 13.5 Cloud training on GitHub Actions

A workflow at `.github/workflows/train.yml` runs training on GitHub's
`ubuntu-latest` runner (4 logical vCPUs over 2 physical cores, 2 SMT
threads/core, 16 GB RAM). Observed hosts are AMD EPYC 9V74/9V45 with
AVX2/FMA; `thread_siblings_list` is `0-1` and `2-3`. The diagnostic job
prints the topology on every run. Dispatch is manual via workflow_dispatch
(mode, per-phase step counts, reward source, dataset bootstrap).

Config differences for the cloud runner, all annotated in
`config/uenr.toml`: `run.threads = 4` (laptop default 2), `training.batch_size
= 16`, `training.grad_accum = 2` (micro-batch 8 vs the laptop's 4), and
`prepare.max_bytes = 2_000_000_000` to cover the full 1.92 GB TinyStories
file. The pool respects the configured pins 0 and 1, then auto-pins the
remaining workers to CPUs 2 and 3, covering both physical cores. A 2-thread
cloud run must pin 0 and 2, because 0/1 are SMT siblings. See
`docs/ARCH_OPTIMIZATION.md` for measured SIMD and threading results.

Per-run outputs (`checkpoints/model.bin`, `checkpoints/train.csv`) are
published as an artifact and a per-run release named
`cloud-training-<run>-<attempt>`. Cross-run resume from prior artifacts
is not yet wired; see `docs/CLOUD_TRAINING.md`.

## 14. Operational runbook

### 14.1 Resume after an unexpected shutdown

1. `cd /home/mike/UENR/uenr`
2. `./uenr train --resume` (or `--config <profile> train --resume`).
3. The trainer prints the chosen file and step, restores optimizer state, and
   continues from the next step.

If the process was killed during a step, the last completed step is in
`paths.model_out`; if it was killed after a periodic checkpoint, the newest
`step_*.bin` is used. The newest three step files are kept automatically.

### 14.2 Reduce training RAM further

Order of application (measured at seq 512):

1. `adafactor = true` (default): 349.7→217.9 MB at seq 256.
2. `block_checkpoint = true`: 382→248 MB at seq 512, 0.81× speed.
3. `fp16_activations = true`: 248→236 MB, 0.96× of block speed.
4. `recompute_swiglu = true` if the FFN caches dominate in a profile.
5. Reduce `seq_len`, `batch_size` or `grad_accum`; reduce `num_experts` if the
   checkpoint size, not activations, is the constraint.

LoRA/QLoRA already avoid base optimizer state. Storing the base itself in
packed INT4 is the next ~27 MB, listed in future work.

### 14.3 Serve long contexts cheaply

Keep `kv_evict=true` (default), set `kv_evict_window` to the number of recent
tokens the task needs, and leave `kv_sparse_k` inert unless eviction is
disabled. Use `kv_fp16=true` if bytes matter more than the last few percent of
decode speed. Use `prefix_cache=true` and `continuous=true` for chat serving.
For unbounded context, eviction is what makes the cache constant-size; do not
disable it unless you have measured the memory budget.

### 14.4 Scale up

Use `config/scale.toml` as the starting point. It sets C=384, L=8, 32 experts,
seq 512, block checkpointing and enables the draft model. Check the resolved
parameter counts with `./uenr --config config/scale.toml config` before
training. For a new vocabulary, retrain with `config/tokenizer16k.toml`, then
update `vocab_size` and the tokenizer paths consistently.

---

## 15. Limitations and honest caveats

- **Corpus size.** The shipped data is a TinyStories slice; “production” here
  means the engineering is complete, not that the model is a general assistant.
  The schedule/optimizers/architecture are designed to scale, but quality at
  8.45M parameters is inherently limited.
- **QLoRA storage.** The frozen base is quantize-dequantized in place; it is
  not yet stored as packed INT4. RAM savings come from eliminating base
  gradients/optimizer state. A packed-base forward is future work.
- **FP16 activations.** Only attention probabilities are stored as FP16.
  Extending FP16 to Q/K/V/SwiGLU caches would save a few more MB at this size
  but needs conversion kernels that may cost more than they save; measure
  before adding.
- **Block checkpointing speed.** 0.81× at seq 512 is the price of 1.54× less
  activation RAM. It is off by default and enabled in the scale and low-RAM
  profiles.
- **Quantized KV speed.** Q4/ternary are memory modes; they are slower than
  FP32 on this CPU. The default is FP32 with eviction.
- **Draft speculation.** 70% acceptance but slower on the 8.45M model; enabled
  only in the scale profile.
- **Tokenizer.** NFC-lite (a small Latin base+mark table), not full NFC/NFKC;
  byte output can be invalid UTF-8. Decoder sanitization is a roadmap item.
- **RL.** PPO/GRPO use sequence-level ratios, a frozen-reference k1 KL penalty
  and a configurable heuristic/model reward; there is no learned value head,
  entropy bonus or GAE.
- **Threading.** GEMM, attention, softmax+CE and the optimizer use the pool;
  other stages are vectorized
  single-thread work.
- **No GPU path.** Everything targets the measured AVX1/F16C CPU.

---

## 16. Future work, in priority order

1. **Packed INT4 base for QLoRA** with a batched quantized training forward
   (reuse the INT4 GEMV kernels with tile dequantization) and a quantized-base
   checkpoint format. Expected ~27 MB further savings.
2. **Larger-corpus training** with the existing curriculum/YaRN machinery and
   the scale profile; measure whether 64.8M total / 9.18M active is the right
   next point.
3. **Draft-model retraining for the scale profile** so draft speculation has a
   better acceptance/speed trade at the larger verification size.
4. **Full UTF-8 sanitizer** and stricter NFC for user-visible output.
5. **More aggressive KV sharing with adapters** at long context; current
   aggressive-sharing long-ppl results are mixed.
6. **True Muon/Newton-Schulz** if a numerically stable cheap variant is found;
   the lite version is a documented negative result.
7. **Inference weight INT4 with QAT** to make the memory mode accuracy-safe on
   larger checkpoints.

---

## 17. Glossary

- **Active parameters** — parameters touched per token (attention + top-1
  expert + shared expert), as opposed to total stored parameters.
- **Adafactor** — Adam variant with factored row/column second moments, much
  smaller optimizer state.
- **Block checkpointing** — store only one layer's activations and recompute a
  layer's forward during backward.
- **Donor / borrower** — active layer that owns K/V cache rows / shared layer
  that reads them.
- **GQA / MQA** — grouped-query / multi-query attention; UENR ships MQA.
- **LCP prefix cache** — longest-common-prefix prefill reuse across prompts.
- **MFU** — model FLOPs utilization against the measured hardware roof.
- **p-RoPE** — partial rotary position embedding on a subset of dimensions.
- **SwiGLU** — gated SiLU feed-forward block used by the experts.
- **UMDL** — the checkpoint container format.
- **UENR** — the project/model name.

---

## 18. One-file takeaway

UENR is not a wrapper around a framework; it is a complete, measured
co-design of a transformer trainer and inference engine for a 2012 two-core
laptop CPU. Its distinctive results are a 1.28M-active-parameter MoE at 8.45M
total, a 218 MB training footprint, a million-token context served from ~0.1 MB
of KV after eviction, ~1k tok/s cached decode, and a crash-safe training loop
that resumes with optimizer state. Every mechanism described here is
implemented in this tree, tested, and exposed through `config/uenr.toml`; if a
number seems surprising, the command to reproduce it is in §13.
