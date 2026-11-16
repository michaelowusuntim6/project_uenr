# Project Status / Handoff

> Read this first when resuming after an interruption.
> Then read `TODO.md`, `docs/ROADMAP.md`, and the relevant section of
> `docs/FORMULAS.md`/`docs/ARCHITECTURE.md`.
> Update this file and `TODO.md` after any substantial change.

## State (commit `43b3803`, local `master`)

The project is complete as an end-to-end, dependency-free C system:

- Clean build from scratch: `make clean && make -j2` (no warnings).
- `make check`: build + 6 test binaries + benchmark builds, all pass.
- Tokenizer trained: 1.5 MB sample → 353,769 tokens, 1,776 merges.
  Artifacts: `artifacts/tokenizer.bin`, `vocab.txt`, `merges.txt`.
- Data prepared from a 16 MiB corpus slice:
  4,055,303 tokens (train 3,852,538 / val 202,765), 2,999 SFT stories and
  2,999 preference pairs (`data/train.bin`, `val.bin`, `sft.jsonl`,
  `pairs.jsonl`).
- Pretrain verified: 100 steps, loss 7.52 → 6.61 (EMA 7.04), ~870 tok/s,
  ~8.8 GFLOP/s, ~13% MFU. Checkpoint: `checkpoints/model.bin` (step 100).
- SFT, LoRA, DPO, PPO, GRPO each verified for at least one optimizer step and
  saved to `checkpoints/{sft,lora,dpo,ppo,grpo}.bin`.
- Inference verified: greedy and sampled generation, interactive chat.
- Interactive menu: all 17 options exercised, including clean + rebuild.
- No NaN/Inf in checkpoints (audited after the AdamW sqrt fix).

## Exact commands used

```bash
cd /home/mike/UENR/uenr
make -j2
./uenr tok-train
./uenr prepare
./uenr train --steps 100
./uenr sft --steps 1
./uenr lora --steps 1
./uenr dpo --steps 1
./uenr ppo --steps 1
./uenr grpo --steps 1
./uenr infer --prompt "Once upon a time" --tokens 40
make check
./uenr-menu        # interactive, option 17 exits
```

Menu self-test sequence used (all options, with 1-step trainers):

```
1, 2, 3, 4, 5, 6, 7(1), 8(1), 9(1), 10(1), 11(1), 12(1),
13("hello"), 14, 15, 16(clean), 2(rebuild), 17
```

## Measurements

| Metric | Value |
|---|---|
| CPU | i5-3210M, 2C/4T, AVX1 + F16C, no FMA/AVX2 |
| AVX roof | 33.9 GFLOP/s/core, 67.8 two-core |
| DRAM triad | 10.5 GB/s (1T), 13.4 GB/s (2T) |
| Naive GEMM 256³ | ~6.3 GFLOP/s |
| UENR GEMM | ~15–24 GFLOP/s |
| Pretrain | ~870 tok/s, ~8.8 GFLOP/s, ~13% MFU |
| Inference | ~50–65 tok/s (full-context recompute) |

## Known limitations / next work

1. **GQA KV cache** for decode (biggest win; currently recomputes the context).
2. fp16 storage + fp32 master weights (F16C helpers exist; storage is fp32).
3. Token-level PPO with a value head (currently sequence-level ratio, group-mean
   baseline).
4. Full NFC/NFKC and a display sanitizer for invalid UTF-8 byte output.
5. `opbench` and unit tests for arena/RNG/IO/model shapes.
6. Smarter SFT packing (no padding waste).

## Files to read/update when continuing

- `TODO.md` — tick completed items, add discoveries.
- `docs/ROADMAP.md` — reprioritize.
- `docs/STATUS.md` — this file; update measurements and verified commands.
- `README.md` — keep the measured numbers and limitations honest.
- `docs/FORMULAS.md` — add a derivation before changing any math.
- `src/model/model.c` + `tests/test_gradcheck.c` — any model/backward change
  must keep the finite-difference check green.
- `config/uenr.toml` — all new knobs go here first, then the parser and docs.

## Safety notes

- `run.resume` loads `checkpoints/model.bin` with Adam state; without it, the
  params are warm-started but the optimizer and step counter reset.
- Checkpoints are written atomically (`.tmp` + rename). A crash cannot leave a
  torn file.
- `.gitignore` ignores `artifacts/`, `checkpoints/`, `data/*.bin`,
  `data/*.jsonl`, and the large raw corpus. A clone needs the corpus placed at
  `data/tinystories_train.txt` before `uenr tok-train`.
- The `core` ignore pattern must stay root-anchored (`/core`); an unanchored
  `core` once ignored `src/core/` and `include/uenr/` (via the `uenr` binary
  pattern). Verify with `git check-ignore -v include/uenr/uenr.h`.
