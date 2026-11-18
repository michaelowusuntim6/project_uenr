# UENR — a dependency-free tiny MoE LLM for a 2012 laptop CPU

UENR trains and runs a **tiny sparse mixture-of-experts transformer from
scratch in plain C** on an Intel i5-3210M (Ivy Bridge, 2 cores / 4 threads,
AVX1, **no FMA, no AVX2**, ~2.2 GHz sustained). No PyTorch, no BLAS, no Python
at runtime, no third-party libraries. Every formula, kernel, optimizer,
tokenizer, trainer, and sampler is handwritten in this tree.

```
8,451,456 total parameters  ·  1,275,264 active parameters per token
24 experts, top-1 routing, shared expert, MQA (1 KV head), cross-layer KV
sharing, hybrid sliding/global attention, p-RoPE, V-norm, QKV-norm, four-norm
residuals, logit softcaps, SwiGLU, tied embeddings, byte-level BPE with atomic
tool-call tokens
```

Measured on the target laptop: **33.9 GFLOP/s per core** and 67.8 two-core AVX
roof, 13.4 GB/s DRAM at two threads. The tuned GEMM reaches **~15–24 GFLOP/s**;
pretraining runs at **~1.15–1.18k tokens/s** with the default Adafactor
(an earlier optimizer A/B measured 0.92× AdamW's throughput in exchange for
lower memory and better ppl), and 128-token KV-cache decode at **~1.0k
tokens/s**. The combined serving stack (continuous batching + prefix cache +
eviction) reached **3.32×** a lockstep baseline with a shared long system
prompt in an earlier serving measurement. In the v0.1.0 release-validation
run, a fresh 100-step Adafactor pretrain reached **val loss 4.157 / ppl 63.89**
(8 batches × 512-token eval windows, 16,384 tokens) and one-step peak training
RSS was **215,748 KiB**. Throughput, perplexity and RSS numbers are
context-dependent; see `docs/BENCHMARKS.md` and `docs/RELEASE.md`.

Long context is where the architecture pays off: the shipped 4-layer model has
only two active KV slots (one sliding-window ring, one global), so KV storage is
**256 B/token** FP32 (256 MB at 1M tokens; 128 MB FP16, 64 MB INT8, 40 MB
packed 4-bit) versus 2,048 B/token in the original GQA design — **8× less** in
FP32 and **32×/51× less** with INT8/Q4 KV. With the default StreamingLLM-style
eviction (sinks=4 + newest 256), the 1M-context cache is only **388 rows
(~0.1 MB FP32)** at a 1.4–2× decode gain on long contexts. At `max_seq=65536`,
end-to-end inference RSS is **150.8 MB** (vs 985.8 MB with the naive full-slot
cache and full-size inference scratch).

## What changed in the productionization pass

- **Block checkpointing** (`training.block_checkpoint`): one layer's saved
  activations, each layer forward re-run in backward. At seq 512 peak training
  RSS 382→248 MB (1.54× less) for 0.81× speed.
- **FP16 activation storage** (`training.fp16_activations`): attention
  probabilities stored as IEEE half. With block checkpointing: 236 MB
  (1.62× less than default), 0.96× of block speed, identical loss.
- **QLoRA low-RAM**: frozen base gets no gradient buffer and no optimizer
  state; LoRA/QLoRA RSS 225→194 MB at unchanged speed.
- **Crash-safe resume**: v2 checkpoints carry Adafactor/AdamW/INT8-Adam/Muon
  state; periodic `step_*.bin` files (newest 3 kept), SIGINT/SIGTERM save the
  current step, and `--resume` discovers periodic checkpoints even when the
  final model file is absent, ranking files by their header step. Optimizer
  initialization after a load is idempotent, and an optimizer-config mismatch
  fails instead of silently resuming with cold moments.
- **Strict configuration**: the handwritten TOML subset rejects unknown
  sections/keys, malformed or overflowing values, bad strings/arrays and
  out-of-range settings with the file, line, section and key. No-op options
  were removed rather than silently ignored.
- **Scale profile**: `config/scale.toml` (64.8M total / 9.18M active) turns on
  block checkpointing and draft-model speculation (`config/draft.toml`).
- **`docs/DETAILED.md`**: one long, ordered, deduplicated document for a new
  collaborator.

## Quickstart

Place the training corpus at `data/tinystories_train.txt` (one story per
line). The shipped config reads a 200 MiB slice for `prepare`
(`prepare.max_bytes = 209715200`), so the 1.8 GB file stays out of git and
out of memory. Tokenizer training separately samples 8 MiB
(`tokenizer.train_bytes = 8388608`).

```bash
make -j2
./uenr tok-train
./uenr prepare
./uenr train --steps 100
./uenr train --resume          # after a shutdown/interrupt
./uenr infer --prompt "Once upon a time"
```

Or use the numbered launcher (everything returns to the menu after Enter):

```bash
./uenr-menu
```

Full pipeline (all are real, tested commands):

```bash
./uenr config                     # resolved config + parameter counts
./uenr tok-train                  # train byte-level BPE on the corpus
./uenr prepare                    # tokenize data, build SFT + preference files
./uenr train --steps N            # pretrain
./uenr sft   --steps N            # assistant-only supervised finetune
./uenr lora  --steps N            # LoRA (frozen base, merged at the end)
./uenr dpo   --steps N            # direct preference optimization
./uenr ppo   --steps N            # clipped policy gradient
./uenr grpo  --steps N            # group-relative policy gradient
./uenr infer --prompt "..."       # generation (greedy/top-k/top-p)
./uenr infer --chat               # interactive chat
./uenr bench --size 256           # GEMM throughput vs measured roof
make test                         # full test suite
make check                        # build + tests + benches + ISA verification
make smoke                        # end-to-end pipeline in a temp directory
```

> The consolidated `master` config carries the GitHub-runner overrides
> documented below. The laptop defaults (200 MiB prepare slice, batch 8,
> pins 0/2) are preserved as inline `# laptop default:` comments; reverting
> those lines restores the original laptop profile.

## Cloud training (GitHub Actions)

A cloud training workflow lives at `.github/workflows/train.yml` on
`master` with a `mode` input selecting pretrain, SFT, LoRA, DPO, GRPO, PPO,
reward-model, eval, `full-core`, `full-rl` or `smoke`. Trigger it manually from the Actions tab
(https://github.com/michaelowusuntim6/project_uenr/actions/workflows/train.yml),
select the branch and step count, and it runs on a 4-vCPU / 16 GB
`ubuntu-latest` runner with `-march=x86-64-v3 -mavx2`. The training pool uses all
four worker threads (CPUs 0–3); the laptop default remains two.

The local laptop build is unchanged: `/home/mike/UENR/uenr` is the
primary development tree, and `/home/mike/UENR/uenr_github` is the
cloud working copy. See `docs/CLOUD_TRAINING.md` for details and
current limitations.

## What is implemented

- **CPU-tuned math**: FMA-free exp/log/sin/cos/sigmoid/SiLU/tanh/GELU,
  rcp/rsqrt + Newton, AVX1-safe exponent construction, F16C helpers. The
  default build targets Ivy Bridge (`-march=ivybridge`); `make check-isa`
  rejects any AVX2/FMA instruction in the binaries.
- **GEMM**: 4×8 AVX microkernel with four accumulator chains, packed B panels,
  NN/NT/TN/TT, GEMV decode path, two-thread pool pinned to physical cores.
- **Transformer**: RMSNorm, per-head QK-norm, V-norm, split-half p-RoPE,
  causal MQA with cross-layer KV sharing, hybrid sliding/global windows,
  four-norm residual blocks with zero-init post norms, attention/final logit
  softcaps, SwiGLU, tied head, full forward **and** backward.
- **KV/memory**: compact donor-slot cache, ring buffers for local layers,
  FP32/FP16/INT8/Q4/ternary KV, eviction + sinks, INT8 weights, blockwise INT8
  Adam moments, attention-prob recompute, full block gradient checkpointing,
  FP16 attention-probability storage, and a no-base-state QLoRA path.
- **Sparse MoE**: top-1 router, shared expert, load-balance auxiliary loss,
  grouped expert dispatch, expert/router gradients.
- **Tokenizer**: handwritten byte-level BPE trainer + heap encoder + decoder,
  sanitizer, chat template, atomic tool-call markup.
- **Training**: pretrain, SFT with autoregressive next-token labels (assistant
  content and its `<end>` only), LoRA/QLoRA (with merge), DPO, PPO, GRPO;
  custom epoch formula, adaptive cosine LR, plateau early stop, grad
  accumulation, clipping, AdamW/Adafactor/Muon/INT8-Adam, atomic checkpoints
  with optimizer state, SIGINT-safe save, and header-ranked resume.
- **Inference**: greedy/temperature/top-k/top-p, min-p, repetition and
  no-repeat-ngram penalties, constrained tool-call JSON, speculative decoding
  (prompt lookup + batched span verification), batched prompt-file inference,
  interactive chat.
- **Engineering**: strict single-source TOML config, Result-style errors,
  arena allocation, xorshift RNG, atomic IO, header dependency tracking,
  sanitizer-clean tests, end-to-end smoke script, and full docs.

## Configuration

Everything lives in **`config/uenr.toml`** — model shape, MoE, tokenizer ids,
training, schedule, LoRA, alignment, rewards, inference, paths. There are no
hardcoded hyperparameters in the C sources. `docs/CONFIG.md` documents every
field and the exact supported syntax; unknown keys are errors, not silent
defaults.

## Repository layout

```
config/uenr.toml     single source of truth
include/uenr/        public headers
src/core/            arena, math, config, rng, io, thread
src/ops/             gemm, norm, rope, attention, swiglu, softmax_ce, optim
src/model/           parameter layout, forward/backward, checkpoints
src/data/            tokenizer, dataset
src/train/           schedule, pretrain/SFT/LoRA, DPO/PPO/GRPO
src/infer/           generation and chat
src/main.c           CLI
src/menu.c           numbered interactive launcher
tests/ bench/ docs/
```

The corpus, tokenizer artifact, prepared binaries, and checkpoints are generated
and git-ignored.

## Documentation

`docs/README.md` is the index. Highlights:

- [docs/RELEASE.md](docs/RELEASE.md) — v0.1.0 release documentation, validation results and limitations.
- [CHANGELOG.md](CHANGELOG.md) — release history; v0.1.0 is the first public release.
- [docs/DETAILED.md](docs/DETAILED.md) — the complete guide (start here).
- [docs/CHECK_LIST.md](docs/CHECK_LIST.md) — items 1–58 with measured xN and commits.
- [docs/CHECK_LIST02.md](docs/CHECK_LIST02.md) — second optimization plan + addendum.
- [docs/TODO.md](docs/TODO.md) — living build checklist.
- [docs/AUDIT.md](docs/AUDIT.md) — measured hardware audit and the plan.
- [docs/FORMULAS.md](docs/FORMULAS.md) — every formula and derivation.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — layouts and data flow.
- [docs/TOKENIZER.md](docs/TOKENIZER.md) — BPE and tool-call template.
- [docs/TRAINING.md](docs/TRAINING.md) — all training modes.
- [docs/MOE.md](docs/MOE.md) — routing and gradients.
- [docs/BUILD.md](docs/BUILD.md) — build/test/benchmark details.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — measurement protocol and results.
- [docs/STATUS.md](docs/STATUS.md) — verified commands and handoff state.
- [docs/ROADMAP.md](docs/ROADMAP.md) — what is done and what is next.

## Honest limitations

- Generation uses the compact MQA + cross-layer-shared KV cache by default;
  `kv_cache=false` keeps the full-recompute reference path. Long context is
  handled by sliding windows + eviction (constant-size cache), not by storing
  the full 1M-row cache unless eviction is disabled.
- Weights/master copies are fp32. Attention probabilities can be stored as
  FP16 (`training.fp16_activations`), QLoRA quantize-dequantizes the frozen
  base to INT4, and QAT/INT8/INT4 inference modes exist; a packed INT4 training
  base is future work.
- PPO uses a sequence-level clipped ratio and the group mean as baseline (no
  learned value head, entropy bonus or GAE). The reference-policy KL is the
  single-sample k1 estimator with a genuine gradient; the reward is a
  configurable length/format/repetition heuristic, not a human preference
  model.
- The tokenizer sanitizer does UTF-8 validation, line normalization and a small
  NFC-lite Latin composition table, not full NFC/NFKC; generated byte output
  may be invalid UTF-8.
- The laptop default trains on a 200 MiB TinyStories slice, not the full
  1.8 GB, by design (`prepare.max_bytes = 209715200`); the tokenizer trains on
  a separate 8 MiB sample (`tokenizer.train_bytes`). The consolidated master
  config overrides this for the cloud runner (full 1.92 GB file).

## License / provenance

UENR is distributed under the **Apache License, Version 2.0**
(SPDX-License-Identifier: `Apache-2.0`). The complete license text is in the
root [`LICENSE`](LICENSE) file.

```
SPDX-License-Identifier: Apache-2.0
```

The project is original C code. Its architecture and training ideas were
informed by `karpathy/llm.c` (MIT-licensed; framework-free C trainer
structure) and the Apache-2.0-licensed MiniMind project (RMSNorm/RoPE/GQA/
SwiGLU model family), audited in `docs/AUDIT.md`. No source code from either
repository is copied, linked, or vendored; the implementations in this tree
were written for UENR. Both reference licenses are permissive and compatible
with Apache-2.0. Because no third-party source or required NOTICE is included,
no separate NOTICE file is distributed.
