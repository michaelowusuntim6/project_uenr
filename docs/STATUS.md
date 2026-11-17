# Project Status / Handoff

> Read this first when resuming. Owning plan: `docs/CHECK_LIST.md` (items 1–58 with
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
./uenr prepare              # streams 200 MiB -> UDAT (tokenizer: 8 MiB sample)
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
| Pretrain throughput | ~1.15k tok/s default Adafactor (1.23k AdamW), vs 878 baseline |
| 100-step pretrain | val loss 4.215, ppl 67.7 with Adafactor |
| Training RSS (default) | **217.9 MB** (Adafactor), 349.7 MB AdamW |
| Serving stack | 16 prompts, shared 120-token prefix: 0.77 s vs 2.56 s (**3.32×**) |
| 300-step stability run | loss 7.694→3.706, val loss 3.610, ppl 36.98, no NaN |
| Decode, 128 tokens | ~1.0–1.1k tok/s, exact KV parity |
| Batched inference, B=8 | 1718.5 tok/s vs 538.5 sequential (3.19×) |
| KV at S=512 | 160 KiB FP32 (640 rows) |
| KV at 1M (no eviction) | 256 MB FP32 / 128 MB FP16 / 64 MB INT8 / 40 MB Q4 |
| KV at 1M (default eviction) | ~388 rows ≈ 0.1 MB FP32 |
| Inference RSS at 64k ctx | 150.8 MB (was 985.8 MB) |
| Attention-prob checkpointing | S=1024 RSS 610→463 MB (1.32× less, 0.86× speed) |
| Full block checkpointing | seq512 RSS 382→248 MB (1.54× less), 0.81× speed |
| FP16 activation storage | + block at seq512: 248→236 MB (1.62× less vs default), 0.96× of block speed, identical loss |
| QLoRA / LoRA low-RAM | 225→194 MB (1.16× less), speed unchanged, no base grad/optimizer state |
| Resume | newest step file vs model_out by header step; optimizer state restored; SIGINT saves after current step |
| Sparse decode | 1.38× @2048, 3.19× @8192 (eviction off) |
| INT8 Adam | optimizer state 64→40 MB (1.6×), RSS 349.6→259.4 MB (1.35×), ppl 81.27 |
| Tokenizer | 3.64 bytes/token, round-trip exact; 55.2M train tokens from 200 MB |

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
- No NFC/NFKC composition; byte output can be invalid UTF-8 (display sanitizer
  is roadmap).

## Files to update when continuing

- `docs/CHECK_LIST.md` — living plan, measured xN, commit per item.
- `docs/ARCHITECTURE.md` — model and KV architecture.
- `docs/STATUS.md` — this file.
- `docs/TODO.md` / `docs/ROADMAP.md` — future work.
