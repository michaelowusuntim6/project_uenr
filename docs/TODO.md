
> **Status note:** the authoritative, item-by-item execution record with
> measured xN and commit hashes is `docs/CHECK_LIST.md` (items 1–58). Several boxes
> below predate the optimization/DOCUMENT program; see `STATUS.md` for the
> final measured state.
>
# UENR TODO — Detailed Build Checklist (living)

> Tick `[x]` when complete. Add new items as discovered.
> Rule: **no hardcoded hyperparameters outside `config/uenr.toml`**.
> Target: i5-3210M, 2C/4T, AVX1 + F16C, no FMA/AVX2, measured 33.9 GFLOP/s/core
> and 13.4 GB/s DRAM at 2 threads. Model: 8.47M total / 1.30M active params.

---

## 0. Foundation

- [x] `docs/AUDIT.md` — repos, measured CPU, ceilings, plan.
- [x] `config/uenr.toml` — single source of truth; parser in `src/core/config.c`.
- [x] `result.h` — Result-style errors + `UENR_TRY`.
- [x] `arena.h/.c` — 64B bump allocator.
- [x] `simd.h` — AVX1-safe wrappers, unaligned ops, hsum/hmax, rcp/rsqrt,
      2^k via split 128-bit ints, F16C convert.
- [x] `math.h/.c` — exp, log, sin, cos, sigmoid, SiLU, tanh, GELU, rcp, rsqrt,
      scalar + vector; no libm transcendentals; no FMA assumptions.
- [x] `rng.h/.c` — xorshift128+, seed, uniform, normal, shuffle.
- [x] `io.h/.c` — read/write, mmap, atomic write, recursive mkdir, sizes.
- [x] `thread.h/.c` — persistent pool, condvars, CPU pinning, range jobs.
- [x] Header dependency tracking (`-MMD -MP`) — fixed a real stale-layout bug.
- [x] fp16 storage of the largest activation (attention probabilities) with
      F16C conversions; weights/master copies remain fp32 (QAT is opt-in).
- [x] block checkpointing, QLoRA low-RAM base state, crash-safe resume.

## 1. Custom formulas (docs/FORMULAS.md)

- [x] exp: range reduction + degree-6 Horner + `2^k` bit construction, clamped.
- [x] log: exponent/mantissa split + atanh series.
- [x] sin/cos: quadrant reduction + Taylor, RoPE tables.
- [x] sigmoid/SiLU/tanh/GELU: reuse exp/rcp.
- [x] rcp/rsqrt: hardware estimate + one Newton step; safe sqrt for Adam.
- [x] RMSNorm forward/backward with cached rstd, fused affine.
- [x] QK-norm via the same row function over `T·H` / `T·G` rows.
- [x] RoPE forward + inverse transpose, per-sequence positions.
- [x] GQA causal attention forward/backward, softmax derivative.
- [x] SwiGLU forward/backward, recomputed gate/up, 5-buffer scratch.
- [x] fused softmax+CE, `dlogits = softmax − onehot`, ignore mask.
- [x] AdamW with bias correction, decoupled decay, no-decay norms.
- [x] global L2 clip, loss EMA, plateau early stop.
- [x] custom epoch formula (`tokens_per_param · active_params`).
- [x] adaptive cosine LR with clamped loss-based multiplier.
- [x] MoE top-k router gradient + Switch auxiliary-loss gradient.
- [x] DPO pairwise log-prob derivative.
- [x] sequence-level PPO clipped ratio; GRPO group-normalized advantage.
- [ ] token-level PPO ratio + learned value head.

## 2. Operators

- [x] `gemm.c`: NN/NT/TN/TT, 4×8 AVX microkernel, packed B panels, threaded
      over rows, GEMV batch-1 path.
- [x] `norm.c`, `rope.c`, `attn.c`, `swiglu.c`, `softmax_ce.c`, `optim.c`.
- [x] gradient-checked dense + MoE configurations.

## 3. Model

- [x] contiguous 64B-aligned parameter block + tensor descriptors.
- [x] init: N(0,0.02), norms=1, residual output scaling.
- [x] embedding → 4 layers → final norm → tied logits.
- [x] full backward incl. tied-head accumulation and embedding scatter.
- [x] MoE grouping/dispatch/scatter and router/shared gradients.
- [x] `UA` activation layout, reusable buffers, no per-step allocation.
- [x] atomic `UMDL` checkpoints with optional Adam state + resume.
- [x] LoRA adapters on configured q/k/v/o targets, frozen base, merge.

## 4. Tokenizer

- [x] byte-level BPE trainer with generation-stamped pair hash table.
- [x] deterministic merge tie-break; configurable sample and vocab.
- [x] O(n log n) heap encoder with special-token barriers.
- [x] decoder with recursive merge expansion.
- [x] sanitizer: CRLF, controls, UTF-8 validation, lowercase.
- [x] special ids from config (system/user/assistant/tool_call/tool_result/end).
- [x] chat inference + SFT templates; assistant-only labels.
- [x] atomic tool-call markup preserved in content.
- [x] vocab/merges text dumps.
- [ ] full NFC/NFKC normalization.
- [ ] regex-style pretokenizer.
- [ ] terminal display sanitizer for invalid UTF-8 output.

## 5. Data

- [x] `UDAT` uint16 train/val binaries with header.
- [x] line-wise BOS/EOS packing, configurable `max_bytes`, val split.
- [x] SFT JSONL generation from TinyStories prompts/completions.
- [x] preference pair generation.
- [x] handwritten JSON parser for SFT messages and pairs.
- [ ] smarter packed sequence boundaries (no padding in SFT batches).

## 6. Training

- [x] pretrain loop: micro-batches, accumulation, clip, AdamW, logging.
- [x] SFT with assistant-only masked loss.
- [x] LoRA finetune + merge.
- [x] DPO with frozen reference.
- [x] PPO sequence-level clipped policy gradient.
- [x] GRPO group-relative advantage.
- [x] CSV metrics, tokens/s, FLOP/s, MFU.
- [x] periodic + final checkpoints; resume.
- [x] custom step budget and early stop.
- [ ] token-level PPO / value head.
- [ ] schema-aware tool-call reward.

## 7. Inference

- [x] checkpoint/tokenizer load, shape validation.
- [x] full-context generation, greedy/temperature/top-k/top-p.
- [x] chat template and special-token rendering.
- [x] interactive chat (menu calls it in-process).
- [ ] GQA KV cache decode (top roadmap item).

## 8. CLI and launcher

- [x] `uenr` subcommands: config, tok-train, prepare, train, sft, lora, dpo,
      ppo, grpo, infer, bench, selftest.
- [x] `uenr-menu`: numbered 1..17, 17=Exit; every action returns to menu after
      Enter; toolchain check/install offer; clean confirmation.
- [x] menu calls inference in-process so piped stdin works.
- [x] every menu option exercised end-to-end with a scripted session.

## 9. Benchmarks and tests

- [x] `bench/cpubench.c`, `threadbench.c`, `gemmbench.c`.
- [x] `test_math`, `test_config`, `test_gemm`, `test_ops`, `test_tokenizer`,
      `test_gradcheck`.
- [x] `make check` builds all, runs tests and benchmarks.
- [ ] `opbench` per-op bandwidth, arena/RNG/IO unit tests, model-loss regression.

## 10. Docs and repo hygiene

- [x] `README.md`, `.gitignore`, `docs/` complete set (12 documents).
- [x] AUDIT and FORMULAS/ARCHITECTURE/CONFIG/TOKENIZER/TRAINING/MOE/
      INFERENCE/BUILD/LAUNCHER/ROADMAP.
- [x] `git init` + commit (local only; no push).
- [x] final clean build + tests + tokenizer/prepare + pretrain + inference demo.

---

## Progress log / measurements

- CPU audit: 33.9 GFLOP/s/core, 67.8 two-core; DRAM 10.5→13.4 GB/s at 2T;
  AI crossover ≈ 5 FLOP/byte.
- Naive GEMM 256³ ≈ 6.3 GFLOP/s → UENR GEMM ≈ 15–24 GFLOP/s.
- Pretrain default: ~870 tok/s, ~8.8 GFLOP/s, ~13% MFU; loss 7.67 → 7.52 in
  20 warmup steps, then 7.52 → 6.61 (EMA 7.04) over a 100-step run.
- Tokenizer: 1.5 MB → 353,769 tokens (1,776 merges).
- Prepared dataset: 16 MiB → 4,055,303 tokens (train 3,852,538 / val 202,765),
  2,999 SFT stories and preference pairs.
- Bugs found and fixed:
  - missing header dependency tracking caused stale config layouts;
  - SwiGLU backward scratch was 5 buffers but used 6;
  - AdamW `v·rsqrt(v)` produced NaN for zero-gradient tensors (now `vsqrtps`);
  - tokenizer worker arguments were freed while threads still used them;
  - menu inference via `system()` lost piped stdin (now in-process).
- Still open: token-level PPO/value head, packed INT4 training base for QLoRA,
  full NFC/NFKC + display sanitizer, optional opbench. See `ROADMAP.md`.
