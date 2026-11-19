# UENR v0.1.0 — Release Documentation

Release date: **2026-11-17**  
Release commit: the commit this document ships in (tagged `v0.1.0`)  
License: **Apache-2.0** (root [`LICENSE`](../LICENSE))  
Repository: <https://github.com/michaelowusuntim6/project_uenr>

UENR is an experimental, dependency-free tiny sparse mixture-of-experts
language model written from scratch in C for a 2012 laptop CPU. This document
describes the shipped implementation; deeper material lives in the linked
documents and is not duplicated here.

## 1. Project overview

- Pure C11 implementation: model, kernels, tokenizer, data pipeline, training,
  alignment, inference and serving.
- No PyTorch, TensorFlow, BLAS, CUDA, ONNX Runtime, llama.cpp, Python runtime or
  third-party numerical library. `-lm` and `-pthread` are the only link
  dependencies.
- Target machine: Intel i5-3210M (Ivy Bridge), 2 cores / 4 threads, AVX1 +
  F16C, **no AVX2/FMA**, 16 GB DDR3, Ubuntu.
- The shipped model is intentionally small so it can be trained and run on the
  target laptop. It is **not** production-ready, state-of-the-art, or a
  general-purpose assistant.

Detailed documentation: [`docs/DETAILED.md`](DETAILED.md), architecture in
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md), formulas in
[`docs/FORMULAS.md`](FORMULAS.md).

## 2. Model architecture

| Property | Shipped value |
|---|---|
| Transformer layers | 4 |
| Hidden width | 192 |
| Query heads / KV heads | 6 / 1 (MQA) |
| Head dimension | 32 |
| MoE experts | 24, top-1 routed, plus one shared expert |
| Shared expert width | 277 |
| Attention | Hybrid sliding-window/global, cross-layer KV sharing, QK-norm, split-half p-RoPE |
| FFN | SwiGLU |
| Residuals | Four-norm blocks with zero-init post norms |
| Output | Tied embedding/lm-head, logit softcaps |
| Tokenizer | Byte-level BPE, 2,048-vocab shipped, atomic tool-call tokens |
| Context | `max_seq` 512 by default; sliding window 128; KV eviction/sinks |

The model is a research/engineering demonstration of CPU-first LM
implementation, not a reproduction of a published result.

## 3. Parameter counts

Verified by `./uenr config` on the shipped `config/uenr.toml`:

- **8,451,456 total parameters** (32.2 MiB in FP32).
- **1,275,264 active parameters per token** (top-1 routing + shared expert).
- 68 tensors, 24 routed experts + shared expert, MQA with one KV head.

Historical development stages used slightly different counts (for example
8,474,560 / 1,298,368 before the KV-head/sharing changes); those numbers are
historical and are not the shipped model. See
[`docs/CHECK_LIST.md`](CHECK_LIST.md) for the item-by-item history.

## 4. Hardware requirements

- x86-64 CPU with **AVX1 and F16C**. AVX2/FMA are not required.
- Linux (POSIX `mmap`, `pthread`, `sched_setaffinity`).
- Tested on: Intel Core i5-3210M, 2C/4T, 16 GB DDR3, Ubuntu, kernel 7.0.0-43.
- Memory: training RSS is roughly 200–220 MiB with the shipped Adafactor
  configuration for the model itself; the corpus is mmap’d (the shipped 106 MiB
  token file). 16 GB is recommended for the full 1.8 GB source corpus and
  long-context experiments.
- No GPU is used. There is no CUDA path.

## 5. Compiler and build requirements

- C11 compiler: tested with **gcc 15.2.0** and **clang 21.1.8**.
- POSIX `make`.
- `objdump` is optional, used by `make check-isa`.
- No CMake/autotools, Python, BLAS or third-party runtime is required to build
  or run.
- Default target profile: `-march=ivybridge -mavx -mf16c` (AVX2/FMA disabled).
  `make native` builds for the machine doing the build; do not ship that binary
  to the target laptop.

## 6. Installation / build

```bash
git clone https://github.com/michaelowusuntim6/project_uenr.git
cd project_uenr
make -j2
```

This produces `./uenr` and `./uenr-menu`. No `make install` step is needed.

Verification:

```bash
make test          # 16 test executables
make check         # build + tests + benchmarks + AVX2/FMA ISA check
make check-isa     # assert no AVX2/FMA instructions
make smoke         # end-to-end pipeline in a temporary directory
```

Clang uses the same target profile:

```bash
make clean
make -j2 CC=clang
```

## 7. Quickstart

The repository does not include the training corpus. Place a text corpus at
`data/tinystories_train.txt` (one story per line); the shipped config streams
up to 200 MiB (`prepare.max_bytes = 209715200`). The tokenizer separately
samples 8 MiB (`tokenizer.train_bytes = 8388608`).

```bash
./uenr tok-train
./uenr prepare
./uenr train --steps 100
./uenr train --resume          # after an interrupt
./uenr infer --prompt "Once upon a time"
```

The interactive launcher runs the same commands:

```bash
./uenr-menu
```

`make smoke` requires **no external corpus** and exercises the whole pipeline
on generated synthetic text in a temporary directory.

## 8. Training workflow

- **Pretraining** (`uenr train`): mmap’d token stream, `grad_accum`
  micro-batches, fused softmax+CE, clipped global norm, AdamW/Adafactor/Muon/
  INT8-Adam, periodic optimizer-state checkpoints, SIGINT/SIGTERM save.
- **SFT** (`uenr sft`): assistant-only, next-token targets; user/system/tool
  spans masked with a configurable ignore index; tail truncation preserves the
  supervised span.
- **LoRA / QLoRA** (`uenr lora`): frozen base, low-rank adapters merged into the
  base at save; QLoRA quantize-dequantizes the frozen base to INT4 without base
  gradients or optimizer state.
- **Reward model** (`uenr train-rm`): linear pairwise head over final hidden
  states.
- **DPO** (`uenr dpo`): logistic preference objective against a frozen
  reference; gradients derived and finite-difference tested.
- **PPO / GRPO** (`uenr ppo`/`uenr grpo`): sequence-level policy gradients with
  the clipped PPO surrogate and a k1 reference-KL penalty. There is no learned
  value head, entropy bonus or GAE.
- **Resume**: `--resume` discovers periodic checkpoints by header step even if
  the final model file is absent, restores optimizer state, and rejects a
  checkpoint written by a different optimizer configuration.

Full details: [`docs/TRAINING.md`](TRAINING.md), configuration:
[`docs/CONFIG.md`](CONFIG.md).

## 9. Inference workflow

```bash
./uenr infer --prompt "Once upon a time" --tokens 128 --greedy
./uenr infer --chat
./uenr infer --batch prompts.txt --tokens 32 --greedy
./uenr eval
```

Implemented: KV-cached decode, greedy/temperature/top-k/top-p/min-p sampling,
repetition/no-repeat-n-gram penalties, constrained tool-call JSON, speculative
decoding, batched and continuous serving, prefix caching and KV eviction.
Quantized KV (FP16/INT8/Q4/ternary) and INT8/INT4 weight inference are optional;
the default is FP32 KV with eviction.

See [`docs/INFERENCE.md`](INFERENCE.md).

## 10. Testing instructions

The release was validated on the target laptop on 2026-11-17 with the
commands below. No test is claimed as passed unless it was executed.

```bash
make clean
make -j2
make test
make bench
make check-isa
make check
make smoke
```

Results of the release validation:

- 16 test executables: **all passed** under GCC 15.2.
- Clang 21.1.8 clean build: **all 16 tests passed**, no warnings, ISA clean.
- ASan + UBSan build: **all 16 tests passed**; full smoke pipeline passed with
  leak detection enabled; no sanitizer errors.
- `make check`: build, tests, benchmarks and ISA assertion passed.
- `make smoke`: passed (temporary directory; no real checkpoints or data
  modified).
- No compiler warnings under either compiler.
- Gradient checks (dense, MoE, shared-KV/donor and block-checkpoint cases)
  passed as part of `test_gradcheck`.

The test matrix is described in [`docs/BUILD.md`](BUILD.md).

## 11. Benchmark methodology and results

All measurements below are from the release-validation session on the target
laptop: Intel i5-3210M, 2C/4T, 2.50 GHz base, 16 GB DDR3, Ubuntu kernel
7.0.0-43, gcc 15.2.0, `-O3 -march=ivybridge -mavx -mf16c`, two physical
threads pinned to CPUs 0 and 2, shipped `config/uenr.toml`.

| Measurement | Context | Result |
|---|---|---|
| Pretrain throughput | Fresh random model, 20-step `bench/measure.sh train 20`, final step | **1179 tok/s**, 11.80 GFLOP/s, 17.4% of the measured 2-core AVX roof |
| Training loss | Fresh model, same 100-step fresh run | Step 100 train loss **4.2821** (EMA 5.0224) |
| Validation loss / perplexity | Same run, `eval --batches 8 --seq 512` (16,384 tokens) | **val loss 4.1572, ppl 63.89** |
| GEMM 256³ | `uenr bench --size 256`, two threads | **20.2 GFLOP/s** this run (20–22 GFLOP/s across runs) |
| Decode, short | 20 greedy tokens including prefill/setup | 587 tok/s |
| Decode, longer | 128-token request (107 tokens emitted, EOS) | 1010 tok/s |
| Peak training RSS | `/usr/bin/time -v`, one pretrain step, Adafactor | **215,748 KiB** |
| Model size | `./uenr config` | 8,451,456 total / 1,275,264 active params, 32.2 MiB FP32 |

Run-to-run variation is roughly ±2% for throughput. Historical measurements in
`docs/CHECK_LIST.md` and `docs/CHECK_LIST02.md` used different code states and
model shapes and are **not** current release numbers. Full methodology and
rejected optimization experiments: [`docs/BENCHMARKS.md`](BENCHMARKS.md).

### KV-cache sizing (calculated, not a 1M-token validation)

The shipped compact cache uses 256 B/token in FP32 for two active KV slots.
At 1M tokens the arithmetic projections are:

| Format | 1M-token KV |
|---|---|
| FP32 | 256 MB |
| FP16 | 128 MB |
| INT8 | 64 MB |
| Packed Q4 | 40 MB |
| Optional eviction (sinks=4, newest 256, sliding ring) | ~388 rows ≈ 24 KiB INT8 |

Current revision note: eviction is now **off by default**; the full context
is kept (INT8 ≈64 MB (61 MiB) at 1M). The v0.1.0 measurements below were taken with
eviction enabled.

These are layout calculations at a 1M context length. **End-to-end model
quality at 1M tokens was not validated.** Tested long-context work includes
sliding-window/ring-buffer decode, eviction, compact caches and a 64k-context
inference memory measurement (150.8 MB RSS) documented in
[`docs/STATUS.md`](STATUS.md). Do not interpret the cache arithmetic as a
validated 1M-context capability claim.

## 12. Known limitations

- Small experimental model; generation quality is limited and repetitive on
  smoke-scale checkpoints. Not production-ready or state-of-the-art.
- AVX1+F16C only; no runtime ISA dispatch and no GPU path.
- No full NFC/NFKC normalization (NFC-lite only); generated byte output can be
  invalid UTF-8.
- PPO/GRPO use sequence-level ratios and no learned value head or GAE.
- Q4/ternary KV are memory modes and slower than FP32.
- Speculative decoding and FlashAttention can be slower at short contexts on
  this CPU; they are optional and off by default.
- Items 37–39 and 45 (large-corpus scaling, longer runs, 20–50M models,
  distillation) are intentionally deferred.
- 1M-token end-to-end quality/throughput is not validated; only cache sizing
  and shorter long-context paths are.

## 13. Roadmap

Near-term work is tracked in [`docs/ROADMAP.md`](ROADMAP.md) and
[`docs/TODO.md`](TODO.md). High-level directions: larger-corpus/longer
training, optional token-level PPO with a value head, fuller Unicode handling,
display sanitization, larger model profiles, and further long-context
validation. No date commitments are made.

## 14. License

UENR is distributed under the **Apache License, Version 2.0**
(SPDX-License-Identifier: `Apache-2.0`). The complete text is the root
[`LICENSE`](../LICENSE) file.

Provenance: the project is original C code. `karpathy/llm.c` (MIT) and the
Apache-2.0 MiniMind project informed architecture/training design only; no
third-party source is copied, linked or vendored, and no separate NOTICE is
required.
