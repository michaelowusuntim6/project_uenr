# Changelog

All notable changes to UENR are documented here.
The format follows a human-readable Keep-a-Changelog style; versions use
Semantic Versioning once the public API stabilizes.

## [Unreleased]

- Cloud training: GitHub Actions workflow with per-run checkpoint
  releases on `master` (manual dispatch, 4 vCPU/16 GB runner,
  `-march=native`). Laptop defaults preserved as inline config
  comments. See `docs/CLOUD_TRAINING.md`.
- Cloud training uses all four vCPUs (`run.threads = 4`); the pool respects
  configured pins and auto-pins additional workers to CPUs 2 and 3. The
  laptop profile remains two threads.

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
