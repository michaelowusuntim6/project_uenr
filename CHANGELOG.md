# Changelog

All notable changes to UENR are documented here.
The format follows a human-readable Keep-a-Changelog style; versions use
Semantic Versioning once the public API stabilizes.

## [Unreleased]

- **KV eviction is off by default:** inference keeps the full context. INT8 KV
  is 64 B/token in the global layer (64 MB / 61 MiB at 1M, plus the 128-row
  local ring). `inference.kv_evict = true` restores the constant-size
  StreamingLLM cache (388 rows, ~24 KiB INT8).
- **TOML-driven datasets:** `[datasets.*]` now carries the full fetch recipe
  (`release_tag`, `release_asset`, `url`, `derive_from`, `convert_from`,
  `convert_to`, `limit`, `bootstrap`). `uenr datasets` prints it and
  `scripts/bootstrap_datasets.sh` consumes it, so local and browser-triggered
  workflow runs use the TOML without any workflow edit. Formats and examples:
  `docs/DATASETS.md`.
- **Linear training pipeline:** the supported sequence is now
  `pretrain → SFT → DPO → GRPO → eval`, each phase loading the previous
  checkpoint. `scripts/train_pipeline.sh` runs it locally with `--short`,
  `--steps` and per-phase overrides; workflow `mode=pipeline` runs the same
  order; the launcher's option 8 is the straight-line entry point.
- **Retired weak/duplicated methods:** LoRA/QLoRA (`uenr lora`), PPO
  (`uenr ppo`) and the learned reward model (`uenr train-rm`) are no longer
  exposed by the CLI, menu or workflow. Low-level implementations and tests
  remain as legacy research; GRPO uses the rule-based `[rl]` reward.
- **INT8 KV is the default:** 64 B/token at 1M (4× smaller than FP32),
  ~1.04× FP32 decode at 128 tokens, max logit error 4.5e-3. Batched and
  continuous inference support INT8 caches. FP16/Q4/Q2 remain opt-in.
- **SFT document-boundary packing:** `training.sft_pack` (default true)
  first-fit packs conversations into each row; attention takes a per-position
  `doc_start` mask so documents never attend across boundaries. Measured
  733 vs 643 tok/s (1.14x) at seq 128.
- **R1 post-training refinements:** `rl.reward_verify` adds an exact-answer
  arithmetic verifier, groups with reward sd < 1e-4 are skipped, and
  `grpo.ref_refresh` (400) periodically copies the policy into the reference.
- **Newton-Schulz Muon:** `uenr_muon_orthogonalize` implements the quintic
  iteration (tall/wide transpose-tested) but costs 75x an Adafactor step on
  this CPU, so `muon_ns_iters` defaults to 0 and the true update is opt-in.
- **SwiGLU activation clamp:** `training.swiglu_clamp = 10.0` clamps the gate
  to `[0, c]` and the up branch to `[-c, c]` with zero derivative at the
  clamp, shared across scalar, AVX2 and quantized paths.
- **ADAPT study:** moved to `docs/ADAPT.md`; `docs/CHECK_LIST03.md` records
  the applied, experiment-only and rejected items.
- **GitHub orchestration:** `./uenr-menu` option 22 opens a 9-choice
  Cloud / GitHub submenu (auth, token, run status, pipeline/smoke/bench
  dispatch, checkpoint cleanup). `scripts/dispatch.sh` modes are now
  `pipeline|smoke|bench|pretrain|sft|dpo|grpo|eval`.
- **GitHub orchestration (earlier iteration):** the Cloud / GitHub submenu
  added authentication, run status, dispatch and release cleanup. `scripts/dispatch.sh` and
  `scripts/cleanup_releases.sh` expose the same actions from a shell.
- **Token setup:** `github_token.txt` (gitignored, mode 0600) or
  `.github_token`/`config/github_token.txt`; falls back to `GH_TOKEN` and
  `gh auth`. `scripts/github_auth.sh setup` validates and stores the token;
  `github_token.txt.example` is the template.
- **`install.py`:** stdlib-only global dependency installer with
  `--check/--yes/--global/--force/--dry-run`, distro package-manager detection,
  skipped-present tools and a gh release-binary fallback.
- **Release cleanup:** checkpoint tags can be deleted while keeping the
  dataset caches; v0.1.0 is never touched.
- **Cloud config tuned:** batch_size 16→64, grad_accum 2→1, LR 1.41e-3
  (sqrt scaling), pretrain default 1600 steps (26.2M tokens at 16,384
  tokens/step); SFT 1000, DPO 600, GRPO 400.
  `batch_size` and `lr` are workflow_dispatch inputs. The earlier 5000-step
  figure assumed 2,048 tokens/step and would have been 82M tokens.
- **AVX2 saturation of non-GEMM kernels:** attention AXPY/max, SwiGLU
  forward/gradients, final softcap (also fixed to run after the logits
  projection), AdamW/L2 pieces. AVX2 4t pretrain 6957→7463 tok/s (+7.3%).
- **AVX-512 opt-in dispatch:** 4x16 zmm GEMM and 16-wide attention; 384-cube
  4t 229 vs 195 GFLOP/s, 100-step pretrain 7887 vs 7463 tok/s. `auto` prefers
  AVX-512 when AVX512F/VL/BW are present; AVX1 remains the laptop path.
- **`config/uenr.laptop.toml`:** one-flag revert to batch 16 / accum 2 /
  two threads / pins 0,2.
- **Runtime AVX2+FMA kernels.** The Ivy Bridge AVX1 path remains the default
  build target and is unchanged at runtime on the laptop. A second set of
  `*_avx2.c` objects (GEMM/GEMV, attention, RMSNorm, SwiGLU, softmax-CE,
  AdamW, RoPE, INT8/Q4 KV unpack) is selected with `[run] simd =
  auto|avx1|avx2` when the CPU reports AVX2 and FMA. Both paths are linked;
  `make check-isa` checks the AVX1 objects for AVX2/FMA and the AVX2 objects
  for actual FMA. The exp/log polynomials stay mul+add in both paths.
- A `mode=bench` workflow dispatch measures GEMM AVX1 vs AVX2 at 384/1024,
  four 100-step pretrain runs, and INT8/Q4 KV decode at 512-token context,
  and writes the tables to the run summary. Runner topology (`lscpu -e`,
  SMT sibling lists) is reported on every diagnostic run.
- Cloud training: GitHub Actions workflow with stable per-phase checkpoint
  releases on `master` (manual dispatch, 4 vCPU/16 GB runner,
  `-march=x86-64-v3 -mavx2 -mf16c`). Laptop defaults preserved as inline
  config comments. See `docs/CLOUD_TRAINING.md`.
- Cloud training uses all four vCPUs (`run.threads = 4`); the pool respects
  configured pins and auto-pins additional workers to CPUs 2 and 3. The
  laptop profile remains two threads.
- Per-phase, multi-dataset cloud pipeline: `[datasets.*]` config tables,
  `prepare --phase <name>`, C-only dataset and checkpoint conversion, stable
  dataset/checkpoint release caches, all four runner vCPUs, and workflow modes
  for pretrain/SFT/LoRA/DPO/GRPO/PPO/reward/eval/full-core/full-rl/smoke.
  The full smoke pipeline (13 jobs, all eight phases) passed twice on
  GitHub Actions on 2026-11-18; the second run reused the dataset assets and
  resumed pretrain from step 40 to 60.

## [0.1.0] — 2026-11-17

First public release. UENR is a dependency-free, CPU-first tiny
mixture-of-experts language model written from scratch in C for an Intel
i5-3210M (Ivy Bridge). It does not use PyTorch, BLAS, CUDA, Python at runtime,
or any third-party model runtime.

### Added

- **Model and kernels**: 8,451,456-parameter sparse MoE transformer with
  1,275,264 active parameters per token; 24 top-1 routed experts plus a shared
  expert; RMSNorm, QK-norm, V-norm, split-half p-RoPE, causal MQA (1 KV head),
  cross-layer KV sharing, hybrid sliding/global attention, four-norm residual
  blocks, logit softcaps, SwiGLU and a tied output head.
- **Handwritten AVX1 + F16C implementation** for Ivy Bridge: 8×8 GEMM
  microkernel, packed panels, GEMV decode path, fused softmax+CE, RMSNorm,
  RoPE, SwiGLU, attention forward/backward and optimizer kernels; FMA-free
  formulas for exp/log/sin/cos/sigmoid/SiLU/tanh/GELU.
- **Training pipeline**: byte-level BPE tokenizer, streaming data preparation,
  pretraining, SFT with correct autoregressive next-token labels, LoRA/QLoRA,
  DPO, PPO and GRPO with a trainable pairwise reward model.
- **Optimizers and reliability**: AdamW, Adafactor, Muon and blockwise
  INT8-Adam; gradient clipping; optimizer state in checkpoints; crash-safe
  resume that discovers periodic checkpoints by header step even without a
  final model file.
- **Inference**: KV-cached decode, greedy/temperature/top-k/top-p/min-p
  sampling, repetition and no-repeat-n-gram constraints, constrained tool-call
  JSON, speculative decoding, batched and continuous serving, prefix cache and
  StreamingLLM-style KV eviction.
- **Memory modes**: FP16/INT8/Q4/ternary KV caches, INT8/INT4 weight inference,
  attention-probability checkpointing, full block checkpointing, FP16
  activation storage and frozen-base QLoRA.
- **Engineering and docs**: strict dependency-free TOML-subset configuration,
  Result-style error propagation, aligned arena allocation, xorshift RNG, atomic
  I/O, a numbered interactive launcher, 16 test executables, an end-to-end
  smoke script, sanitizer support and a complete documentation set.

### Changed

- SFT labels are shifted to next-token prediction; truncation preserves the
  supervised assistant span.
- DPO/PPO/GRPO gradients, PPO clipping and the reference-policy KL follow the
  documented objectives, with finite-difference and directional tests.
- Optimizer initialization is idempotent across checkpoint restore; a mismatched
  optimizer kind fails instead of silently dropping state.
- Configuration parsing is strict: unknown sections/keys, malformed values and
  out-of-range settings are errors with file/line/section/key context.
- The default build targets Ivy Bridge (`-march=ivybridge`) and `make check-isa`
  rejects AVX2/FMA instructions in the binaries.

### Fixed

- DPO preference-gradient direction, PPO/GRPO policy-gradient signs and
  clipping regions, and the previously unused reference KL.
- Adafactor/Muon state overwrite and leaks on resume.
- SFT preparation leak; malformed preference-JSONL frees and buffer over-read.
- Continuous/batched inference cleanup, prefix-cache initialisation and
  tokenizer leaks.
- Silent GEMM and attention-backward allocation failures.
- NFC-lite `e/E` diaeresis mappings.
- Shared-KV configurations without a same-type donor (KV buffer overflow).
- GNU nested function in inference that blocked Clang.

### Verification

- GCC 15.2 and Clang 21.1.8 builds are warning-free.
- All 16 test executables pass under GCC, Clang, and ASan+UBSan.
- `make check` passes build, tests, benchmarks and the ISA assertion.
- `make smoke` exercises the full pipeline in a temporary directory, including
  resume without a final checkpoint, SFT, eval, all inference modes, DPO, PPO,
  GRPO and reward-model training.

### Notes

- UENR is an experimental 0.x research/engineering project, not a
  production-ready or state-of-the-art model.
- Reported throughput, perplexity and memory values depend on the run and are
  documented with their measurement context in `docs/BENCHMARKS.md` and
  `docs/RELEASE.md`.
- Licensed under the Apache License, Version 2.0 (`Apache-2.0`).
