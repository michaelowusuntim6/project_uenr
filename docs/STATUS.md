# Project Status / Handoff

> **Release:** v0.1.0, first public release (2026-11-17). See
> [`RELEASE.md`](RELEASE.md) and [`../CHANGELOG.md`](../CHANGELOG.md).

> Read this first when resuming. Owning plan: `docs/CHECK_LIST.md` (items 1–58 with
> measured xN and commits). Architecture: `docs/ARCHITECTURE.md` and
> `docs/DOCUMENT.md`. Build/ops: `docs/BUILD.md`.

## State

UENR is a complete dependency-free C system: byte-level BPE, streaming data
preparation, and a strict linear training pipeline
(**pretrain -> SFT -> DPO -> GRPO -> eval**) where each phase loads the
previous checkpoint. LoRA/QLoRA, PPO and the learned reward model were
retired from the CLI/menu/workflow after the DeepSeek adaptation study
(`docs/ADAPT.md`); their code and tests remain as legacy research. The system
also ships KV-cached and batched inference, constrained decoding, evaluation,
a 23-action launcher, tests, and documentation. It builds warning-free on the
target i5-3210M (AVX1+F16C, no FMA/AVX2). Runtime AVX2+FMA copies of the hot
kernels are selected on newer CPUs; see `docs/ARCH_OPTIMIZATION.md`.

Shipped model: **8,451,456 total / 1,275,264 active** parameters, 24 experts
top-1 plus a 277-wide shared expert, MQA (1 KV head), two-layer cross-layer KV
sharing (SGSG plan), sliding window 128 + global attention, four-norm residual
blocks with zero-init post norms, p-RoPE tables, optional V-norm/softcaps.

## Verified commands

```bash
cd /home/mike/UENR/uenr
make -j2                    # warning-free
make test                   # all test binaries pass
./uenr tok-train            # byte-level BPE (pretokenized), ~4 s on 8 MiB
./uenr prepare              # streams 200 MiB -> UDAT (tokenizer: 8 MiB sample)
./uenr train --steps 100    # pretrain (val ppl 81.0 at 100 steps)
./uenr eval                 # val loss/ppl, tokenizer + generation metrics
./uenr infer --prompt "Once upon a time" --tokens 128 --greedy
./uenr infer --batch prompts.txt --tokens 20 --greedy
./uenr sft|dpo|grpo --steps N
scripts/train_pipeline.sh --short   # 4/4/2/2 linear pipeline
./uenr bench --size 256
./uenr-menu                 # interactive; last option exits
```

Data/tokenizer artifacts and checkpoints are local (gitignored). Old checkpoints
from before item 49 (the KV-head/width change) are rejected by the shape check;
retrain.

## Headline measurements (target laptop)

| Metric | Value |
|---|---|
| GEMM 256³ | ~21 GFLOP/s (1.32× the first blocked kernel) |
| Pretrain throughput | ~1.15k tok/s default Adafactor (1.23k AdamW), vs 878 baseline |
| 100-step pretrain (v0.1.0 validation) | val loss **4.1572**, ppl **63.89** with Adafactor (8×512 eval, 16,384 tokens) |
| Training RSS (default) | **215,748 KiB** one step with Adafactor (v0.1.0 validation); 349.7 MB AdamW measured separately |
| Serving stack | 16 prompts, shared 120-token prefix: 0.77 s vs 2.56 s (**3.32×**) |
| 300-step stability run | loss 7.694→3.706, val loss 3.610, ppl 36.98, no NaN |
| Decode, 128 tokens | ~1.0–1.1k tok/s, exact KV parity |
| Batched inference, B=8 | 1718.5 tok/s vs 538.5 sequential (3.19×) |
| KV at S=512 | 160 KiB FP32 (640 rows) |
| KV at 1M (no eviction) | 256 MB FP32 / 128 MB FP16 / 64 MB INT8 / 40 MB Q4 |
| KV at 1M (default eviction) | ~388 rows ≈ 0.1 MB FP32 |
| Default KV precision | INT8 (64 B/token at 1M, ~1.04× FP32 decode, 4.5e-3 max logit error) |
| SFT packing | 733 vs 643 tok/s (1.14×) at seq 128 |
| GRPO verifier | exact-answer arithmetic rule reward, zero-variance skip, reference refresh 400 |
| True Muon | opt-in only: 75× the Adafactor step cost |
| Inference RSS at 64k ctx | 150.8 MB (was 985.8 MB) |
| Attention-prob checkpointing | S=1024 RSS 610→463 MB (1.32× less, 0.86× speed) |
| Full block checkpointing | seq512 RSS 382→248 MB (1.54× less), 0.81× speed |
| FP16 activation storage | + block at seq512: 248→236 MB (1.62× less vs default), 0.96× of block speed, identical loss |
| QLoRA / LoRA low-RAM | 225→194 MB (1.16× less), speed unchanged, no base grad/optimizer state |
| Resume | newest step file vs model_out by header step; optimizer state restored; SIGINT saves after current step |
| Sparse decode | 1.38× @2048, 3.19× @8192 (eviction off) |
| INT8 Adam | optimizer state 64→40 MB (1.6×), RSS 349.6→259.4 MB (1.35×), ppl 81.27 |
| Tokenizer | 3.64 bytes/token, round-trip exact; 55.2M train tokens from 200 MB |

Rows explicitly labelled as earlier development measurements (optimizer A/Bs,
300-step run, batched/serving) use their own checkpoints and eval windows; the
two v0.1.0 validation rows above are the current release numbers. Do not
combine them.

## Tests

`make test` covers math/config/GEMM/ops, gradcheck (dense, MoE, shared-KV,
donor accumulation, recompute-attn, softcaps), tokenizer, sampled softmax, KV
cache (FP32/INT8/FP16, sharing, sliding windows, span/multi parity), INT8 Adam,
and JSON constraints. `make check` builds everything including benchmarks.

## Known limitations / deferred items

- Items 7/26 (FP16 weight storage, two-row GEMV) were implemented, measured
  slower, and reverted; see `docs/CHECK_LIST.md`.
- Item 22 speculative decoding is implemented and exact, but prompt-lookup
  acceptance on the smoke model was 0/37 (0.38×), so it defaults off.
- Item 30 is complete: `training.block_checkpoint` keeps one layer's saved
  activations and re-runs each layer forward during backward (gradcheck cases
  `block-checkpoint`, `block+donor-adapter`, `block+fp16-probs`).
- Item 32 QLoRA is implemented as INT4 quantize-dequantize of the frozen base
  plus LoRA; base gradients and optimizer state are not allocated. Storing the
  base itself in packed INT4 would save another ~27 MB and is future work.
- `training.fp16_activations` stores attention probabilities as FP16 (largest
  saved activation); other activation buffers remain FP32.
- Items 37–39/45 (large-corpus/scale/distillation) are intentionally deferred.
- The sanitizer implements NFC-lite (a small Latin base+mark composition table),
  not full NFC/NFKC; byte output can be invalid UTF-8 (display sanitizer is
  roadmap).
- Configuration is strict: unknown sections/keys, malformed values and
  out-of-range settings are errors; no-op hardware/PPO keys were removed.
- Resume discovers periodic checkpoints by header step even when the final
  model is absent; optimizer initialization is idempotent and a mismatch in
  optimizer kind fails instead of silently dropping state.
- DPO/PPO/GRPO fail closed when the policy or reference checkpoint is missing;
  PPO/GRPO use the k1 reference KL with a correct gradient.

## Correctness and release pass (2026-11-17)

The mission fixes were applied one commit at a time, each with focused
regression tests; the full suite, Clang build, ASan/UBSan build and the
end-to-end smoke script pass.

| Fix | Commit |
|---|---|
| SFT next-token label alignment; tail truncation keeps supervision | `12f1975`, `d6da88e` |
| DPO gradient direction | `d555074` |
| PPO/GRPO gradient sign, clipping, reference k1 KL, temperature | `49e660c` |
| Idempotent optimizer init and mismatch rejection | `d26777f` |
| Resume discovers periodic checkpoints without a final model | `0035b24` |
| DPO/RL reference and reward-model fail-closed loading | `d88cfc9` |
| Malformed preference JSONL (uninitialised frees, size bound, NUL) | `232c625`, `1c42a97` |
| Continuous/batched inference ownership and cleanup | `a56315e`, `7648d6c` |
| GEMM allocation-failure fallback | `31b4cb6` |
| Attention-backward failure propagation + persistent scratch | `0101548` |
| NFC-lite E/e diaeresis mappings + table tests | `2c7910c` |
| SFT preparation buffer leak | `cb08d08` |
| Strict config parser, ranges, dead-option removal, profiles | `7faa5f5`, `d7fac0d` |
| Same-type donor validation for shared KV layers | `fbd0c86` |
| Preparation-size consistency (200 MiB / 8 MiB) | `e41d444` |
| GNU nested function removed (Clang) | `bd9c358`, `431c90f` |
| Ivy Bridge build profile + `check-isa` | `cc5f0b7` |
| Smoke/sanitizer infrastructure and benchmark record | `b660951`, `07adc54` |

`make check` runs the GCC build, all tests, benchmarks and the ISA check;
`make smoke` runs the temporary-directory end-to-end pipeline.

## License

**Licensing decision completed (2026-11-17): UENR is distributed under the
Apache License, Version 2.0 (SPDX: `Apache-2.0`).** The canonical full text is
in the repository-root `LICENSE` file. Provenance review found no copied or
vendored third-party source: `karpathy/llm.c` (MIT) and the Apache-2.0
MiniMind project informed the architecture and training design only, as
recorded in README and `docs/AUDIT.md`. No separate NOTICE is required.

## Files to update when continuing

- `docs/CHECK_LIST.md` — living plan, measured xN, commit per item.
- `docs/ARCHITECTURE.md` — model and KV architecture.
- `docs/STATUS.md` — this file.
- `docs/TODO.md` / `docs/ROADMAP.md` — future work.
