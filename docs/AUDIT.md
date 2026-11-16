# UENR Audit — Repos, CPU, and the Plan

> Purpose of this document: record (1) what each cloned repo is, (2) exactly what we
> steal from each, (3) what this laptop's CPU actually *is* — measured, not assumed —
> (4) the performance model and CPU-specific formulas we will optimize against, and
> (5) the structure of the project we are about to build.
>
> Everything here is derived for **this machine**: Intel i5-3210M (Ivy Bridge),
> 2 cores / 4 threads, no AVX2, no FMA. Generic “peak FLOPs” tables do not apply.

---

## 1. The two repositories

### 1.1 `karpathy/llm.c`

**What it is.** A from-scratch C/CUDA implementation of GPT-2/GPT-3 training and
inference. The CPU path (`train_gpt2.c`) is a self-contained ~1000-line reference
trainer that mirrors the CUDA version kernel-for-kernel. It exists to prove that a
transformer trainer needs no framework: just a tokenized binary file, a parameter
blob, and a loop.

**Files we care about** (all under `repos/llm.c/`):

| File | What it teaches us |
|---|---|
| `train_gpt2.c` | Full forward+backward training loop in plain C: parameter layout, activation buffers, gradient buffers, loss, AdamW, LR schedule, checkpointing. Our single best structural reference. |
| `llmc/layernorm.cuh` / encoder | LayerNorm forward caches `mean` and `rstd` per token; backward reuses them instead of recomputing. |
| `llmc/matmul.cuh` | GEMM forward and backward (`dW += xᵀ·dy`, `dx = dy·Wᵀ`) with explicit transpose flags. |
| `llmc/attention.cuh` | Causal attention in one kernel, padding-safe, with the 1/√d scaling folded in. |
| `llmc/gelu.cuh` | GELU forward/backward, tanh approximation. |
| `llmc/fused_classifier.cuh` | Fused softmax + cross-entropy: computes only `logit − max`, skips the full softmax, and fuses the `dlogits` computation for the backward pass. |
| `llmc/adamw.cuh` | AdamW with bias correction done in the step-counter form. |
| `llmc/dataloader.h` | Memory-mapped uint16 token stream; 1024-byte header + `(int32 tokens, int32 file_size)` records. |
| `llmc/schedulers.h` | Linear warmup + cosine decay, written as a pure function of step. |
| `llmc/mfu.h` | Model-FLOPs-utilization bookkeeping — we adapt this into our own throughput meter. |
| `llmc/tokenizer.h` | Minimal GPT-2 BPE loader (vocab + merges from files). |
| `dev/cpu/matmul_forward.c` | Naive kernel vs. `ngc92` register-blocked kernel with `LOOP_UNROLL=8`. **Caveat below.** |
| `Makefile` | Builds with `clang -Ofast -march=native -fopenmp`. |

**The one warning.** `ngc92`’s blocking assumes the compiler can emit FMA (it relies
on `-Ofast` contraction). Ivy Bridge has **no FMA**. If we copy that kernel verbatim
we get *two* uops per multiply-add on the same port and lose ~half the throughput.
We keep the *blocking idea* and re-derive the microkernel for mul+add on ports 0/1
(§3.4).

### 1.2 `michaelowusuntim6/minimind`

**What it is.** A compact PyTorch/HuggingFace training stack for a ~64M-parameter
Qwen3-aligned LLM, designed to be trainable on one consumer GPU. It ships the model
definition plus the full finetuning family (SFT, LoRA, DPO, PPO, GRPO, distillation).

**The model we are actually rebuilding** (from `model/model_minimind.py`,
`MiniMindConfig`):

| Hyperparameter | Value |
|---|---|
| vocab_size | 6400 |
| hidden_size (C) | 768 |
| num_hidden_layers (L) | 8 |
| attention heads (H) | 8 |
| KV heads (GQA) | 4 |
| head_dim | 96 |
| intermediate_size | `ceil(768·π/64)·64` = **2432** |
| hidden_act | SiLU (SwiGLU FFN) |
| RMSNorm eps | 1e-6 |
| RoPE theta | 1e6 (optional YaRN ×16) |
| tie_word_embeddings | True |
| dropout | 0.0 |
| MoE | off by default (4 experts, top-1, aux 5e-4 if enabled) |
| q_norm / k_norm | per-head RMSNorm on Q and K |

**Architectural differences from GPT-2 that we must implement** (llm.c does *not*
have these):

1. **RMSNorm**, not LayerNorm — no mean subtraction, no bias.
2. **RoPE**, not learned position embeddings — rotary applied to Q and K per head.
3. **GQA** (8 query heads, 4 KV heads) — KV cache and attention are not square.
4. **SwiGLU** FFN — `down(silu(gate(x)) * up(x))`, three matrices, width 2432.
5. **Per-head QK-norm** — RMSNorm over `head_dim=96` before RoPE.
6. **Tied embeddings** — `lm_head` weight *is* the token embedding matrix.

**Pretraining recipe** (`trainer/train_pretrain.py`) to mirror as defaults:
bf16 autocast, AdamW `lr=5e-4`, `batch=32`, `seq=340`, `grad_accum=8`,
grad-clip 1.0, cosine schedule, 2 epochs. Data is JSONL `{"text": ...}` with
BOS/EOS added and labels padded to `-100`.

**Files we borrowed knowledge from:** `model/model_minimind.py` (config + forward
math), `trainer/train_pretrain.py` (recipe), `trainer/trainer_utils.py` (LR/clip
helpers), `dataset/lm_dataset.py` (packing, BOS/EOS, ignore index),
`model/tokenizer_config.json` (special-token ids: BOS=1, EOS=2).

**Not yet fetched:** the 451 KB `tokenizer.json` BPE file. Decision deferred (§5.4):
we will handwrite a byte-level BPE rather than ship that artifact, but if we later
want exact parity we can pull it with the same resumable-curl recipe.

### 1.3 What we take from each — one table

| Concern | Source | What we adopt |
|---|---|---|
| Training loop skeleton | llm.c | `train_gpt2.c` buffer layout + AdamW + schedule |
| Backward-pass bookkeeping | llm.c | Cache `rstd`, reuse in backward; explicit `dW` accumulation |
| Fused loss | llm.c | softmax+CE fused, `dlogits = softmax − onehot` |
| Data pipeline | llm.c | mmap’d uint16 stream |
| Model math | minimind | RMSNorm, RoPE, GQA, SwiGLU, QK-norm, tied embeddings |
| Hyperparameters | minimind | C=768, L=8, H=8/KV=4, hd=96, FFN=2432, V=6400 |
| Training recipe | minimind | lr 5e-4, bs 32, seq 340, accum 8, clip 1.0, cosine, 2 ep |
| Tokenizer | minimind | byte-level BPE vocab 6400, BOS/EOS 1/2 |

---

## 2. The machine (measured)

### 2.1 Identity

| Property | Value |
|---|---|
| CPU | Intel Core i5-3210M @ 2.50 GHz (Ivy Bridge, 22 nm) |
| Topology | 1 socket, 2 cores, 2 SMT threads/core = **4 logical CPUs (0–3)** |
| SMT siblings | (0,1) and (2,3) — thread pairs share one physical core |
| Microarchitecture | `-march=native` resolves to **`ivybridge`** |
| RAM | 16 GiB installed, **14 GiB usable**, 29 GiB swap |
| Disk free (`/home/mike/UENR`) | ~42 GiB |
| Virtualization | bare metal (no hypervisor) |

### 2.2 Instruction set — the hard boundaries

| Feature | Present? | Consequence |
|---|---|---|
| SSE / SSE2 / SSE3 / SSSE3 / SSE4.1 / SSE4.2 | ✅ | 128-bit baseline |
| **AVX (256-bit)** | ✅ | 8-wide FP32, *but see split rule* |
| **FMA / FMA4** | ❌ | **Multiply and add are separate uops** |
| **AVX2** | ❌ | No 256-bit integer; no `vpgather`; no 256-bit shifts |
| **BMI1/BMI2** | ❌ | No `pext`/`pdep`/`mulx`; no fast `tzcnt` |
| AVX-512 | ❌ | — |
| F16C | ✅ | `vcvtph2ps` / `vcvtps2ph` — cheap fp16 ↔ fp32 |
| AES-NI / PCLMULQDQ | ✅ | Available but unused by us |
| POPCNT / RDRAND | ✅ | — |
| `-mavx256-split-unaligned-load/store` | **enabled by default** | Unaligned 256-bit loads/stores are split into 2×128-bit. **Alignment is not optional.** |

### 2.3 Clocks and thermals

- Rated 2.5 GHz base / 3.1 GHz turbo.
- Governor `schedutil`; **measured sustained ~2.2 GHz under full AVX load**.
- Package temperature under load 62–66 °C; no throttling cliff observed, but the
  clock droop is real and is why our arithmetic below uses **2.2 GHz**, not 2.5/3.1.
- Idle floor 1.2 GHz.

### 2.4 Cache hierarchy

| Level | Size | Assoc. | Latency (approx cycles) | Scope |
|---|---|---|---|---|
| L1d | 32 KiB | 8-way | ~4 | per core |
| L1i | 32 KiB | 8-way | — | per core |
| L2 | 256 KiB | 8-way | ~12 | per core, **private** |
| L3 | 3 MiB | 12-way | ~30–40 | **shared**, inclusive |
| Line size | 64 B | — | — | — |

Key facts that shape the design:

- L2 is **per-core** (512 KiB total across 2 cores), so two threads do **not** share
  L2 bandwidth. L3 is shared and small (3 MiB) — a 124M-param model does not fit.
- 64 B lines mean a cache-line’s worth of fp32 is **16 floats**. Every inner-loop
  tile should be a multiple of 16 floats along its contiguous axis.
- L1 is 32 KiB = 8 192 fp32. A 64×64 fp32 tile is 16 KiB, which already half-fills L1
  before you count the other operand.

---

## 3. Measured ceilings and the performance model

All numbers below were produced by `bench/cpubench.c` and `bench/threadbench.c`
under `taskset` pinning. They are **achieved**, not theoretical.

### 3.1 Compute

| Kernel | 1 thread | 2 threads |
|---|---|---|
| Scalar FP32 | 3.6 – 4.2 GFLOP/s | — |
| SSE 128-bit | 14.9 – 16.9 GFLOP/s | — |
| **AVX 256-bit** | **33.9 GFLOP/s** | **67.8 GFLOP/s (×2.02)** |

**Why 33.9 and not higher.** Arithmetic peak = cores × (AVX lanes) ×
(FP uops/cycle) × clock. Ivy Bridge has **2 FP execution ports** but only **one is
256-bit-capable for FP multiply** (port 0); a 256-bit add can also issue on port 1
in some steppings, but the *balanced* sustained rate for independent mul/add streams
is effectively **one 256-bit FP op per cycle** for the multiply-heavy case.

```
peak_avx_fp32 ≈ 1 core × 8 lanes × 2 ops/cycle × 2.2e9
             ≈ 35.2 GFLOP/s theoretical
measured      = 33.9 GFLOP/s  (96% of that — the model is honest)
```

So **33.9 GFLOP/s/core is our roof**, and **67.8 GFLOP/s** with both cores.
Scaling is essentially linear (×2.02) *for compute-bound work* because SMT siblings
do not help a saturating FP loop — we pin to physical cores 0 and 2.

### 3.2 Memory

| Level | Read | Write | Copy | Triad (a·b+c) |
|---|---|---|---|---|
| DRAM, 1 thread | ~12.5 | ~7.8 | ~14.2 | ~10.5 GB/s |
| DRAM, 2 threads | — | — | — | **~13.4 GB/s (×1.30)** |
| L2, 1 thread | — | — | — | 17.3 GB/s |
| L2, 2 threads | — | — | — | **35.2 GB/s** |
| L3 (2 MiB working set) | — | — | — | ~12.5 GB/s |
| L1 | — | — | — | ~28 GB/s |

**Critical asymmetry:** compute scales ×2.0 with the second core, but DRAM only
scales **×1.30**. Two threads on a streaming kernel contend for the same memory
controller. This single fact drives the entire parallelization policy.

### 3.3 The arithmetic-intensity crossover

Roofline crossover for this machine:

```
AI* = peak_FLOPs / DRAM_bandwidth
    = 67.8e9 / 13.4e9
    ≈ 5.06 FLOP/byte
```

Interpretation for our model:

| Workload | AI (FLOP/byte) | Verdict |
|---|---|---|
| Elementwise (GELU/SiLU, residual, AdamW) | ~0.1 – 0.5 | **Hard memory-bound** — single thread, vectorize, fuse |
| RMSNorm | ~2 | Memory-bound |
| GEMV (batch=1 decode) | ~1 | Memory-bound |
| Attention (decode, short ctx) | ~2–4 | Memory-bound |
| **GEMM (training, large batch)** | **50 – 500** | **Compute-bound — this is where cores matter** |
| Attention (prefill, seq 340) | ~20–40 | Marginally compute-bound |

**Policy:** use **2 threads only for GEMM and prefill attention**. Everything else
runs single-threaded — a second thread makes it *slower* (contention + sync) while
adding no bandwidth.

### 3.4 Tile budgets derived from this cache, not from textbooks

These are computed from *our* L1/L2/L3 sizes, with fp32 = 4 B, line = 64 B:

| Budget | Derivation | Our number |
|---|---|---|
| Microkernel register tile | ≤ 16 vector registers × 8 lanes; keep 4–6 regs for pointers/accumulators | **4×8 or 6×8 output tile** |
| L1-resident block | ≤ 50 % of 32 KiB for operands | **A-block + B-block ≤ 16 KiB** → e.g. 32×32 fp32 each side |
| L2-resident block | ≤ 50 % of 256 KiB | **≤ 128 KiB per core** → e.g. 96×96 fp32 |
| L3 slice per thread | 3 MiB shared ÷ 2 threads, minus slack | **≤ 1 MiB/thread** |
| Alignment | 64 B line, split-unaligned penalty | **32-byte align all buffers; 64-byte align panel starts** |

Concretely for the microkernel: with no FMA, each `c += a*b` is **one `vmulps` +
one `vaddps`** = 2 uops. To hide the ~4-cycle FP latency we need ≥ 4 independent
accumulator chains in flight. A **6×8 tile** gives 6 independent 8-wide accumulator
registers — enough to keep both FP ports busy without spilling. That is why we pick
6×8 over the FMA-era 8×8: with mul/add split we want *more, smaller* independent
chains, not fewer, wider ones.

### 3.5 Measured GEMM vs. the roof (headroom we intend to close)

Naive register-tiled GEMM on this machine:

| Size | GFLOP/s | % of 33.9 roof |
|---|---|---|
| 64³ | 7.0 | 21 % |
| 256³ | 6.3 | 19 % |
| 768³ | 3.4 | 10 % |

That is a **10–20× gap** to the compute roof. Closing it is the core engineering
goal: packed panels, 6×8 AVX microkernel, L1/L2 blocking at the sizes above, and
2-thread tiling over output rows (not K). At 50 % of roof we would hit ~17 GFLOP/s
per core / ~34 total — a 5–10× training speedup over naive.

---

## 4. Why the generic formulas are wrong here — and our replacements

Every transcendental in a transformer is normally written for a machine with FMA
and fast divide. We have neither. Each item below is a **deliberate substitution**.

### 4.1 exp (for softmax, SiLU, sigmoid)

- **Generic:** `expf()` from libm, or a range-reduced polynomial assuming FMA.
- **Ours:** range-reduce `x = k·ln2 + r`, `|r| ≤ ln2/2`, build `2^k` by integer
  bit-shift into the exponent field, and evaluate a **degree-5 Horner polynomial in
  r** using only `vmulps`/`vaddps` (no FMA needed). ~8 ops vs. ~20 for libm.
- Constants fitted for fp32 over `|r| ≤ 0.347`: `1 + r(1 + r(0.5 + r(0.1667 +
  r(0.04167 + r·0.00833))))`.

### 4.2 sigmoid / SiLU

- **Generic:** `1/(1+exp(−x))` with a divide.
- **Ours:** use the polynomial exp above, then replace the divide with
  `rcpps` + one Newton–Raphson step (`y = y·(2 − d·y)`), which is ~4× cheaper than
  `vdivps` and accurate to < 1 ulp for our tolerance.

### 4.3 tanh (GELU tanh approximation, if we keep it)

- **Ours:** `tanh(x) = 2·sigmoid(2x) − 1`, reusing the fast sigmoid. Avoids a
  separate rational approximation.

### 4.4 RMSNorm reciprocal-sqrt

- **Generic:** `1/sqrt(mean(x²)+eps)` via `sqrt` + divide.
- **Ours:** `_mm256_rsqrt_ps` (1 uop, ~11-bit accurate) + **one Newton–Raphson
  step** `y = y·(1.5 − 0.5·x·y²)` → full fp32 accuracy. ~6 ops instead of ~30.
- Sum of squares uses 4 partial accumulators to hide add latency.

### 4.5 Softmax + cross-entropy

- **Generic:** full `softmax` into a buffer, then `log`, then `−y·log(p)`.
- **Ours (from llm.c’s fused classifier):** never materialize probabilities for the
  loss. Compute `max`, then `sum(exp(x−max))`, then `loss = logsumexp − x[target]`.
  The backward `dlogits = softmax − onehot` is produced in the same pass. For V=6400
  this removes a 6400-wide buffer round-trip per token.

### 4.6 FMA-free dot products and Horner evaluation

Anything the textbooks write as `a*b + c` becomes **two uops**. We therefore:

- Prefer **Horner form** for all polynomials (one multiply + one add per term — the
  same 2 uops, but perfectly pipelined with no dependency stall).
- Keep **4–6 independent accumulator chains** in every inner loop so the 2 uops
  from chain *i* overlap with the 2 from chain *j*.
- **Never** let the compiler “help” with `-ffast-math` contraction into FMA (it
  can’t, there is none) — but we do compile with `-O3 -march=native` and inspect
  `objdump` for unexpected `vdivps`/`vsqrtps` in hot loops.

### 4.7 fp16 storage, fp32 compute

Weights and activations are stored **fp16** (halves DRAM traffic, which is our
bottleneck) and converted with `vcvtph2ps` (F16C, present) into fp32 registers for
arithmetic. This is not generic advice — it is specifically enabled by F16C and
specifically motivated by the **×1.30 DRAM scaling** wall.

### 4.8 Fused kernels (cut memory traffic, not FLOPs)

Because we are memory-bound on most non-GEMM work, the wins come from **fusion**:

1. **RMSNorm + RoPE** in one pass over Q/K.
2. **SiLU(gate)·up** in one pass, feeding `down` directly (no 2432-wide temp round-trip).
3. **Softmax + CE** as above.
4. **AdamW** fused over the parameter array: `m`, `v`, bias-correction, and `p`
   update in one streaming loop; no separate tensor passes.

---

## 5. The project

### 5.1 Principles

1. **Zero external dependencies.** No BLAS, no libm beyond what we implement, no
   Python at runtime. Every dependency is a `.c`/`.h` in the tree.
2. **Rust benefits, expressed as C.** Arena allocators with explicit lifetimes;
   `Result`-style error codes instead of `assert`; strict aliasing/UB discipline;
   zero-cost SIMD wrappers (`static inline` + `__attribute__((always_inline))`).
3. **Mojo benefits, expressed as C.** Explicit vector types, explicit tiling, and
   layout/autotune knobs as compile-time macros — but the output is plain C.
4. **Tuned for i5-3210M, not for a generic x86.** All constants in §3 live in one
   header (`uenr/config/cpu_ivybridge.h`) so they can be re-measured and updated.

### 5.2 Directory structure

```
uenr/
├── Makefile                     # single entry: `make`, `make bench`, `make test`
├── README.md                    # quickstart
├── config/
│   ├── cpu_ivybridge.h          # §3 constants: clock, roofs, tile budgets, align
│   ├── model_minimind.h         # V=6400 C=768 L=8 H=8 KV=4 hd=96 FFN=2432
│   └── train_defaults.h         # lr 5e-4, bs 32, seq 340, accum 8, clip 1.0
├── include/uenr/
│   ├── uenr.h                   # umbrella include
│   ├── result.h                 # Result-style error enum + helpers
│   ├── arena.h                  # bump arena allocator (Rust-Vec-like lifetimes)
│   ├── simd.h                   # AVX/SSE intrinsics wrappers, always_inline
│   ├── math.h                   # exp/sigmoid/tanh/rsqrt (§4)
│   ├── thread.h                 # 2-thread pool pinned to cpu0/cpu2
│   └── io.h                     # mmap, checked read/write
├── src/
│   ├── core/
│   │   ├── arena.c
│   │   └── thread.c
│   ├── ops/
│   │   ├── matmul.c             # packed GEMM, 6×8 AVX microkernel (§3.4)
│   │   ├── rmsnorm.c            # rsqrt + NR, fused with RoPE
│   │   ├── rope.c
│   │   ├── silu.c               # SiLU·up fusion
│   │   ├── softmax_ce.c         # fused softmax + cross-entropy
│   │   ├── attention.c          # GQA causal attention
│   │   └── adamw.c              # fused streaming update
│   ├── model/
│   │   ├── transformer.c        # forward/backward, tied embeddings
│   │   └── params.c             # param layout, init, checkpoint
│   ├── data/
│   │   ├── tokenizer.c          # byte-level BPE, vocab 6400
│   │   └── dataloader.c         # mmap uint16 stream, BOS/EOS, packing
│   ├── train/
│   │   ├── train.c              # main loop, cosine schedule, grad clip
│   │   └── schedule.c
│   └── main.c
├── tests/
│   ├── test_math.c              # exp/sigmoid/rsqrt vs. reference
│   ├── test_matmul.c            # GEMM vs. naive, bit-tolerance
│   ├── test_ops.c               # each op vs. scalar reference
│   └── test_gradcheck.c         # numerical gradient check on tiny model
├── bench/
│   ├── cpubench.c               # existing: compute/memory roofs
│   ├── threadbench.c            # existing: 1t vs 2t scaling
│   └── gemmbench.c              # new: our microkernel vs. roof
├── tools/
│   └── tokens.c                 # inspect tokenized corpus
└── data/
    └── tinystories_train.txt    # 1.8 GB, one story per line
```

### 5.3 Build and run

```bash
make                 # builds uenr (release, -O3 -march=native)
make bench           # builds cpubench, threadbench, gemmbench
./bench/cpubench     # reproduce §3 ceilings on your machine
./uenr --config config/model_minimind.h --data data/tinystories_train.txt
```

Compiler flags (fixed in the Makefile):

```
-O3 -march=native -mavx -mf16c -fno-math-errno -fno-trapping-math
-fstrict-aliasing -std=c11 -Wall -Wextra
```

No `-Ofast` (we want reproducible fp32 and we do not want the compiler to assume
FMA/`-ffinite-math-only`). No `-fopenmp` — we hand-roll a 2-thread pool pinned to
physical cores 0 and 2.

### 5.4 Tokenizer decision

We handwrite a byte-level BPE with vocab 6400 and BOS=1/EOS=2 to match minimind’s
special tokens. If exact parity with minimind’s `tokenizer.json` is later required,
we fetch it with the same resumable-curl recipe used for the repos and add a
`tools/bpe_import.c`. We do not block on it.

### 5.5 Milestones

1. **M1 — Math + SIMD foundation.** `math.h`, `simd.h`, tests green vs. reference.
2. **M2 — GEMM.** 6×8 AVX microkernel; `gemmbench` reaches ≥ 40 % of roof.
3. **M3 — Model forward.** RMSNorm, RoPE, GQA, SwiGLU, tied head; match a PyTorch
   reference on random weights (forward only, fp32).
4. **M4 — Backward + gradcheck.** Numerical gradient check on a 1-layer tiny model.
5. **M5 — Training.** End-to-end on TinyStories; loss decreases; throughput logged
   in tokens/s and FLOP/s (mfu.h-style).

---

## 6. One-paragraph summary

`llm.c` gives us a framework-free C training skeleton (fused softmax+CE, cached
backward statistics, AdamW, mmap dataloader); `minimind` gives us a 64M-parameter
RMSNorm/RoPE/GQA/SwiGLU model and its pretraining recipe. This laptop is a bare-metal
Ivy Bridge i5-3210M: 2 cores / 4 threads, AVX but **no FMA and no AVX2**, ~33.9
GFLOP/s per core, ~13.4 GB/s DRAM at two threads. The arithmetic-intensity crossover
is ~5 FLOP/byte, so GEMM is the only place two cores pay off; everything else is
memory-bound and must be fused and fp16-packed. We replace every FMA-assuming formula
with mul/add-friendly Horner forms, `rsqrt`+Newton, and FMA-free microkernels, and we
build the whole thing as a dependency-free C project whose tuning constants live in
one header so they can be re-measured on this exact machine.
