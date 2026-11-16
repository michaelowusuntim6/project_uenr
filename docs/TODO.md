# UENR TODO — Detailed Build Checklist

> Living document. Tick `[x]` as items complete. Add new items as discovered.
> Rules: **no hardcoded hyperparameters anywhere except `config/uenr.toml`**;
> everything written in C; every "dependency" is a handwritten file in this tree.
> Target machine: i5-3210M (Ivy Bridge), 2C/4T, no FMA/AVX2, measured ~33.9
> GFLOP/s/core and ~13.4 GB/s DRAM at two threads. See `AUDIT.md`.
>
> Model: **UENR-TinyMoE** — ~8.5M total parameters, ~1.3M active parameters per
> token, sparse top-1 MoE, GQA + QK-norm + RoPE + SwiGLU + RMSNorm + tied head.

---

## 0. Foundation & conventions

- [x] `AUDIT.md` — repo audit, measured CPU ceilings, project plan.
- [ ] `TODO.md` — this checklist (keep updated forever).
- [x] `config/uenr.toml` — single source of truth for every setting.
- [x] `include/uenr/result.h` — Result-style error enum + `UENR_TRY`.
- [x] `include/uenr/config.h` + `src/core/config.c` — handwritten TOML-subset parser.
- [ ] Extend config: `[prepare]`, `[tokenizer].train_bytes`, `[schedule]` epoch formula,
      `[bench]`, `[align]`, validation of every derived shape.
- [ ] `include/uenr/arena.h` + `src/core/arena.c` — extend: zeroed alloc, scratch
      watermark (save/restore), stats, 64B alignment guarantee.
- [ ] `include/uenr/simd.h` — extend: unaligned loads/stores, FP16 load/store (F16C),
      horizontal sum/max, vector add/sub/mul/rcp/rsqrt, AVX1-safe integer split.
- [ ] `include/uenr/math.h` + `src/core/math.c` — full scalar + vector formula set:
      exp, exp2, log, sin, cos, sigmoid, silu, tanh, gelu, rsqrt, rcp, pow-free
      helpers. Clamped range reduction; no libm in hot paths.
- [ ] `include/uenr/rng.h` + `src/core/rng.c` — xorshift128+, uniform, normal
      (Box–Muller via our own log/sin), seeded from config.
- [ ] `include/uenr/io.h` + `src/core/io.c` — file read/write, mmap, atomic
      checkpoint save (tmp + rename), little-endian helpers, resumable-friendly.
- [ ] `include/uenr/thread.h` + `src/core/thread.c` — persistent 2-worker pool
      pinned to physical CPUs from config, barrier, `pool_run_range`.
- [ ] `include/uenr/uenr.h` — umbrella header includes everything public.

## 1. Custom formulas — derived for Ivy Bridge (no FMA, no AVX2)

All formulas live in `include/uenr/formulas.h` + implementations under `src/ops/`,
with derivations in `docs/FORMULAS.md`.

- [ ] `exp` — range reduction `x = k·ln2 + r`, degree-5 Horner (mul/add only),
      `2^k` by exponent-field construction that is **AVX1-safe** (split 128-bit ints).
- [ ] `log` — bit-level exponent/mantissa split + degree-7 Horner; needed by CE.
- [ ] `sin`/`cos` — quadrant reduction + odd/even polynomials (RoPE tables).
- [ ] `sigmoid`, `silu` — reuse exp; divide via `rcpps` + one Newton–Raphson step.
- [ ] `tanh` — `2·sigmoid(2x) − 1`.
- [ ] `gelu` — tanh-free rational approximation, mul/add only.
- [ ] `rsqrt` — `_mm256_rsqrt_ps` + one Newton step (already scalar+vector).
- [ ] `rcp` — `_mm256_rcp_ps` + one Newton step.
- [ ] Vector versions of every function above (`uenr_v_*`).
- [ ] Fused dot product with 4 accumulator chains; horizontal sum.
- [ ] Fused **softmax + cross-entropy**: `loss = logsumexp − logit[target]`;
      backward `dlogits = softmax − onehot`; no full-probability buffer for loss.
- [ ] **RMSNorm** forward caches `rstd` per token; backward reuses it; fused affine;
      supports per-head views (QK-norm) and masked/ignore tokens.
- [ ] **RoPE** — custom sin/cos table per position, per-head pair rotation, fused
      with QK-norm; inverse rotation in backward.
- [ ] **QK-norm** — RMSNorm over `head_dim` for Q and K, before RoPE.
- [ ] **SwiGLU** — fused `silu(gate)·up` then down projection; backward recomputes
      gate/up from the saved normed input (no wide temp storage).
- [ ] **GQA causal attention** — online softmax (Flash-style) forward, saves probs
      for backward; KV head sharing `group = heads / kv_heads`; causal mask.
- [ ] **GEMM** — 4×8 AVX microkernel, 4 independent chains, packed B panels for
      transposed-B, blocked for L1/L2; GEMV fast path for decode.
- [ ] **AdamW** — fused single-pass update, decoupled weight decay, bias correction
      in step form, optional grad accumulation.
- [ ] **Global gradient clipping** — L2 norm fused over param arrays before update.
- [ ] **Router** — top-1 softmax gating, optional jitter, Switch-style load-balance
      auxiliary loss with derived gradient.
- [ ] **Custom epoch / LR schedule** — warmup + cosine + loss-plateau early stop +
      token-budget epoch formula (`tokens_per_param`), documented.
- [ ] **Loss EMA** for plateau detection and early stop.
- [ ] **DPO log-prob** formula (sequence logprob with ignore mask).
- [ ] **PPO clipped-ratio** and **GRPO group-relative advantage** formulas.

## 2. Core operators (`src/ops/`)

- [ ] `gemm.c` — NN/NT/TN/TT dispatch, packed panels, threaded over M.
- [ ] `gemv.c` (inside gemm.c) — batch-1 decode path.
- [ ] `norm.c` — RMSNorm / QK-norm forward+backward.
- [ ] `rope.c` — table build + forward/backward.
- [ ] `attn.c` — GQA causal attention forward+backward.
- [ ] `swiglu.c` — SwiGLU forward+backward.
- [ ] `softmax_ce.c` — fused loss + dlogits.
- [ ] `optim.c` — SGD/AdamW, clip, zero-grad, param norm.
- [ ] `elementwise.c` — residual add, copy, scale, masked ops.

## 3. Model: tiny sparse MoE (~8.5M total, ~1.3M active)

- [ ] `include/uenr/model.h` + `src/model/model.c` — `UenrModel` parameter layout,
      per-layer views, contiguous 64B-aligned arena, param count helper.
- [ ] `src/model/init.c` — deterministic init (normal, residual scaling, norm=1).
- [ ] Forward: embedding → per layer [RMSNorm → QKV → QK-norm → RoPE → GQA attn →
      out-proj → residual → RMSNorm → router/shared/routed SwiGLU → residual] →
      final RMSNorm → tied logits.
- [ ] Backward for every stage; gradient buffers; tied-embedding accumulation.
- [ ] MoE dispatch: group tokens by top-1 expert, gather → SwiGLU → scatter with
      gate weight; gate + router softmax backward; shared expert backward.
- [ ] Activation/checkpoint memory arena (`UenrActs`) sized from config.
- [ ] Numerical gradient check on a 1-layer tiny config (`tests/test_gradcheck.c`).
- [ ] Parameter/optimizer checkpoint save + resume (atomic, versioned).
- [ ] Activation memory report helper.

## 4. Tokenizer (production-grade, tool-call-aware)

- [ ] `include/uenr/tokenizer.h` + `src/data/tokenizer.c`.
- [ ] Sanitization: CRLF→LF, strip C0 controls except `\n\t`, validate UTF-8,
      optionally lowercase; lossless byte fallback.
- [ ] Byte-level BPE **trainer** in C (pair counting with generation-stamped hash
      table, deterministic tie-break, configurable sample + merge count).
- [ ] BPE **encoder** with linked list + min-heap (O(n log n)), special-token
      barriers, byte fallback.
- [ ] BPE **decoder** (recursive expansion), vocab/merges text dumps.
- [ ] Special tokens reserved: system/user/assistant/tool_call/tool_result/end,
      BOS/EOS/PAD/UNK — all ids from config.
- [ ] Chat template renderer + tool-call markup preserved intact.
- [ ] JSONL reader for pretrain text and SFT messages (handwritten JSON subset).
- [ ] `tools/tokens.c` CLI: train, encode, decode, inspect.

## 5. Data pipeline (`src/data/`)

- [ ] `dataset.c` — stream text, sanitize, encode, write `train.bin`/`val.bin`
      (uint16, magic+header), configurable `max_bytes` and val split.
- [ ] Packed sampler: random seq windows, inputs + next-token targets.
- [ ] SFT dataset: chat-template tokenization, assistant-only labels, ignore index.
- [ ] Preference pairs generator for DPO (synthetic from TinyStories).
- [ ] `tools/prepare.c` CLI: prepare binary, SFT jsonl, pairs.

## 6. Training (`src/train/`)

- [ ] `schedule.c` — custom warmup/cosine/adaptive LR, epoch/token budget formula,
      plateau EMA + early stop.
- [ ] `train.c` — pretrain loop: micro-batches, grad accumulation, clip, AdamW,
      logging (loss, tokens/s, FLOP/s, MFU), checkpoint at interval, resume.
- [ ] SFT loop — assistant-only masked loss, chat template.
- [ ] LoRA — low-rank adapters on config targets (q/k/v/o), frozen base, SFT variant.
- [ ] DPO — policy vs frozen reference, beta from config, pairwise loss.
- [ ] PPO — rollout sampling, value head, clipped surrogate, KL/entropy.
- [ ] GRPO — group sampling, normalized advantages, KL penalty.
- [ ] Metrics line + CSV log in `checkpoints/`.

## 7. Inference (`src/infer/`)

- [ ] `inference.c` — load checkpoint, GQA KV cache, single-token decode.
- [ ] Sampling: greedy, temperature, top-k, top-p (our RNG + our softmax).
- [ ] Chat template + tool-call output rendering.
- [ ] `uenr infer` CLI with `--prompt`, `--tokens`, interactive chat mode.

## 8. CLI & interactive launcher

- [ ] `src/main.c` — subcommands: `config`, `tok-train`, `tok-encode`, `prepare`,
      `train`, `sft`, `lora`, `dpo`, `ppo`, `grpo`, `infer`, `bench`, `selftest`.
- [ ] `src/menu.c` — **single numbered interactive launcher 1..N, N=Exit**; every
      action returns to menu after press-Enter; resilient to missing tools.
- [ ] Menu options: 1 env check, 2 build, 3 tests, 4 benchmarks, 5 tokenizer,
      6 prepare data, 7 pretrain, 8 SFT, 9 LoRA, 10 DPO, 11 PPO, 12 GRPO,
      13 inference, 14 config viewer, 15 docs viewer, 16 clean, N exit.
- [ ] Self-test every menu path end-to-end (piped input + short config).
- [ ] `--selftest` mode runs a fast smoke of every subsystem non-interactively.

## 9. Benchmarks & verification

- [ ] `bench/gemmbench.c` — our GEMM vs 33.9 GFLOP/s roof, multiple shapes.
- [ ] `bench/opbench.c` — per-op bandwidth vs measured roofs.
- [ ] `tests/test_math.c` — extended: exp/log/sin/cos/sigmoid/silu/tanh/gelu.
- [ ] `tests/test_config.c` — all sections + derived-shape validation.
- [ ] `tests/test_arena.c`, `test_rng.c`, `test_io.c`.
- [ ] `tests/test_gemm.c` — vs naive reference, NN/NT/TN.
- [ ] `tests/test_tokenizer.c` — round-trip, specials, tool-call template.
- [ ] `tests/test_ops.c` — RMSNorm/RoPE/attn/SwiGLU forward vs scalar reference.
- [ ] `tests/test_model.c` — forward shapes + deterministic loss decrease on 20 steps.
- [ ] `tests/test_gradcheck.c` — finite-difference gradient check (tiny model).
- [ ] `tests/test_schedule.c` — LR curve, plateau stop, epoch formula.
- [ ] `make check` = build + every test + benches smoke.

## 10. Docs, repo hygiene, commit

- [ ] Move/extend audit into `docs/AUDIT.md`; add `docs/FORMULAS.md`,
      `docs/ARCHITECTURE.md`, `docs/CONFIG.md`, `docs/TOKENIZER.md`,
      `docs/TRAINING.md`, `docs/MOE.md`, `docs/INFERENCE.md`, `docs/BUILD.md`,
      `docs/LAUNCHER.md`, `docs/ROADMAP.md`.
- [ ] Root `README.md` — overview, quickstart, structure, measured numbers.
- [ ] `.gitignore` — build/, *.o, binaries, checkpoints/, artifacts/, data/*.bin,
      huge raw data, logs.
- [ ] `git init` (if absent), stage source/docs/config, commit. Do **not** push.
- [ ] Final clean build + `make check` + short end-to-end pretrain + inference demo.

---

## Progress log

- [x] (audit) Repos cloned/analyzed; CPU benchmarked; AUDIT.md written.
- [x] (scaffold) arena/math/simd/config + TOML parser + tests.
- [x] (todo) Re-scoped for the real project (tiny MoE, tokenizer, RL family, menu).
- [ ] (now) Foundation hardening → tokenizer → data → model → train → infer → menu → docs.
