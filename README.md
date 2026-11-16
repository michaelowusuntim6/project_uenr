# UENR — a dependency-free tiny MoE LLM for a 2012 laptop CPU

UENR trains and runs a **tiny sparse mixture-of-experts transformer from
scratch in plain C** on an Intel i5-3210M (Ivy Bridge, 2 cores / 4 threads,
AVX1, **no FMA, no AVX2**, ~2.2 GHz sustained). No PyTorch, no BLAS, no Python
at runtime, no third-party libraries. Every formula, kernel, optimizer,
tokenizer, trainer, and sampler is handwritten in this tree.

```
8,474,560 total parameters  ·  1,298,368 active parameters per token
24 experts, top-1 routing, shared expert, GQA, QK-norm, RoPE, SwiGLU,
RMSNorm, tied embeddings, byte-level BPE with atomic tool-call tokens
```

Measured on the target laptop: **33.9 GFLOP/s per core** and 67.8 two-core AVX
roof, 13.4 GB/s DRAM at two threads. UENR's GEMM reaches **~15–24 GFLOP/s**
(vs ~6 GFLOP/s for a naive kernel), and end-to-end pretraining runs at
**~870 tokens/s, ~8.8 GFLOP/s, ~13% of the measured roof**. A 100-step
pretraining run moved the loss from 7.52 to **6.61** (EMA 7.04) — early, but
the curve is monotone and the pipeline is end-to-end real.

## Quickstart

Place the training corpus at `data/tinystories_train.txt` (one story per
line). The shipped config reads a 16 MiB slice for `prepare`, so the 1.8 GB
file stays out of git and out of memory.

```bash
make -j2
./uenr tok-train
./uenr prepare
./uenr train --steps 100
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
```

## What is implemented

- **CPU-tuned math**: FMA-free exp/log/sin/cos/sigmoid/SiLU/tanh/GELU,
  rcp/rsqrt + Newton, AVX1-safe exponent construction, F16C helpers.
- **GEMM**: 4×8 AVX microkernel with four accumulator chains, packed B panels,
  NN/NT/TN/TT, GEMV decode path, two-thread pool pinned to physical cores.
- **Transformer**: RMSNorm, per-head QK-norm, RoPE, causal GQA attention,
  SwiGLU, tied head, full forward **and** backward.
- **Sparse MoE**: top-1 router, shared expert, load-balance auxiliary loss,
  grouped expert dispatch, expert/router gradients.
- **Tokenizer**: handwritten byte-level BPE trainer + heap encoder + decoder,
  sanitizer, chat template, atomic tool-call markup.
- **Training**: pretrain, SFT, LoRA (with merge), DPO, PPO, GRPO; custom epoch
  formula, adaptive cosine LR, plateau early stop, grad accumulation, clipping,
  AdamW, atomic checkpoints + resume.
- **Inference**: greedy/temperature/top-k/top-p, interactive chat.
- **Engineering**: single-source TOML config, Result-style errors, arena
  allocation, xorshift RNG, atomic IO, header dependency tracking, full docs.

## Configuration

Everything lives in **`config/uenr.toml`** — model shape, MoE, tokenizer ids,
training, schedule, LoRA, alignment, rewards, inference, paths. There are no
hardcoded hyperparameters in the C sources. `docs/CONFIG.md` documents every
field.

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

- [docs/AUDIT.md](docs/AUDIT.md) — measured hardware audit and the plan.
- [docs/FORMULAS.md](docs/FORMULAS.md) — every formula and derivation.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — layouts and data flow.
- [docs/TOKENIZER.md](docs/TOKENIZER.md) — BPE and tool-call template.
- [docs/TRAINING.md](docs/TRAINING.md) — all training modes.
- [docs/MOE.md](docs/MOE.md) — routing and gradients.
- [docs/BUILD.md](docs/BUILD.md) — build/test/benchmark details.
- [docs/ROADMAP.md](docs/ROADMAP.md) — what is next (KV cache, fp16 storage).

## Honest limitations

- Generation recomputes the full context per token; a GQA KV cache is the top
  roadmap item. Decode is ~65 tok/s at short context.
- Weights are fp32; F16C conversion helpers exist but fp16 storage is not wired
  into training yet.
- PPO uses a sequence-level clipped ratio and the group mean as baseline (no
  learned value head); the reward is a configurable length/format/repetition
  heuristic, not a human preference model.
- The tokenizer sanitizer does UTF-8 validation and line normalization, not
  full NFC; generated byte output may be invalid UTF-8.
- The shipped run trains on a 16 MiB TinyStories sample, not the full 1.8 GB,
  by design (`prepare.max_bytes`) so that end-to-end runs finish quickly on
  this CPU.

## License / provenance

The project is original C code. Its architecture and training ideas were
informed by `karpathy/llm.c` (framework-free C trainer structure) and
`michaelowusuntim6/minimind` (RMSNorm/RoPE/GQA/SwiGLU model family), audited in
`docs/AUDIT.md`. No code from those repositories is linked or vendored.
