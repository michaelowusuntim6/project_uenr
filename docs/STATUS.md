# Project Status / Handoff

> Read this first when resuming. Owning plan: `CHECK_LIST.md` (items 1–57 with
> measured xN and commits). Architecture: `docs/ARCHITECTURE.md` and
> `docs/DOCUMENT.md`. Build/ops: `docs/BUILD.md`.

## State

UENR is a complete dependency-free C system: byte-level BPE, streaming data
preparation, pretraining, SFT, LoRA, DPO/PPO/GRPO with a trainable reward
model, KV-cached and batched inference, constrained decoding, evaluation, a
numbered launcher, tests, and documentation. It builds warning-free on the
target i5-3210M (AVX1+F16C, no FMA/AVX2).

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
./uenr prepare              # streaming 200 MB corpus -> UDAT
./uenr train --steps 100    # pretrain (val ppl 81.0 at 100 steps)
./uenr eval                 # val loss/ppl, tokenizer + generation metrics
./uenr infer --prompt "Once upon a time" --tokens 128 --greedy
./uenr infer --batch prompts.txt --tokens 20 --greedy
./uenr sft|lora|dpo|ppo|grpo --steps N
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
| Pretrain throughput | ~1.25–1.35k tok/s (vs 878 baseline) |
| 100-step pretrain | val loss 4.395, ppl 81.0 |
| 300-step stability run | loss 7.694→3.706, val loss 3.610, ppl 36.98, no NaN |
| Decode, 128 tokens | ~1.0–1.1k tok/s, exact KV parity |
| Batched inference, B=8 | 1718.5 tok/s vs 538.5 sequential (3.19×) |
| KV at S=512 | 160 KiB FP32 (640 rows) |
| KV at 1M tokens | 256 MB FP32 / 128 MB FP16 / 64 MB INT8 (original: 2.048 GB) |
| Inference RSS at 64k ctx | 150.8 MB (was 985.8 MB) |
| Attention-prob checkpointing | S=1024 RSS 610→463 MB (1.32× less, 0.86× speed) |
| INT8 Adam | optimizer state 64→40 MB (1.6×), RSS 349.6→259.4 MB (1.35×), ppl 81.27 |
| Tokenizer | 3.64 bytes/token, round-trip exact; 55.2M train tokens from 200 MB |

## Tests

`make test` covers math/config/GEMM/ops, gradcheck (dense, MoE, shared-KV,
donor accumulation, recompute-attn, softcaps), tokenizer, sampled softmax, KV
cache (FP32/INT8/FP16, sharing, sliding windows, span/multi parity), INT8 Adam,
and JSON constraints. `make check` builds everything including benchmarks.

## Known limitations / deferred items

- Items 7/26 (FP16 weight storage, two-row GEMV) were implemented, measured
  slower, and reverted; see `CHECK_LIST.md`.
- Item 22 speculative decoding is implemented and exact, but prompt-lookup
  acceptance on the smoke model was 0/37 (0.38×), so it defaults off.
- Item 30 is partial: attention probabilities are recomputed in backward, but a
  full single-layer block-checkpoint mode (q/k/v/norm buffers) is future work.
- Item 32 QLoRA is not implemented: it needs a batched INT8 training forward +
  quantized-base checkpoint format; the ceiling on this 8.5M model is ~24 MB.
- Items 37–39/45 (large-corpus/scale/distillation) are intentionally deferred.
- No NFC/NFKC composition; byte output can be invalid UTF-8 (display sanitizer
  is roadmap).

## Files to update when continuing

- `CHECK_LIST.md` — living plan, measured xN, commit per item.
- `docs/ARCHITECTURE.md` — model and KV architecture.
- `docs/STATUS.md` — this file.
- `TODO.md` / `docs/ROADMAP.md` — future work.
