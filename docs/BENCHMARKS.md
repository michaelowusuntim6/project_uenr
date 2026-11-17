# Benchmarks and Measurement Methodology

All numbers here were measured on the target laptop, not extrapolated.

## Machine and toolchain

- Intel Core i5-3210M (Ivy Bridge), 2 cores / 4 threads, 2.50 GHz base.
- AVX1 + F16C, **no AVX2, no FMA**; 16 GB DDR3; Ubuntu, kernel 7.0.0.
- gcc 15.2.0 and clang 21.1.8, target profile `-march=ivybridge`.
- Flags: `-O3 -march=ivybridge -mavx -mf16c -std=c11 -Wall -Wextra`
  `-fno-math-errno -fno-trapping-math -fstrict-aliasing -pthread`.
- `make check-isa` confirms neither binary contains AVX2/FMA mnemonics.

## Protocol

- Fresh random model for training comparisons (no warm checkpoint).
- Shipped `config/uenr.toml` (8,451,456 total / 1,275,264 active params,
  Adafactor, 2 physical threads pinned to CPUs 0 and 2).
- 20 optimizer steps via `bench/measure.sh train 20`; tokens/s is the final
  step's measured value. Short runs have run-to-run variation of roughly
  ±2%; single-run values are reported honestly rather than as medians.
- GEMM: `./uenr bench --size 256` (NN, two threads) and
  `./build/bench/gemmbench` (NN/NT/TN at 64–384).
- Decode: `bench/measure.sh infer 20` with the local checkpoint.
- Peak RSS: `/usr/bin/time -v` around one pretrain step.
- No benchmark modifies source files; experiments that did not win were
  reverted before committing.

## Current results (this revision)

| Metric | Result |
|---|---|
| Pretrain, 20 steps | **1145–1179 tok/s**, 11.5–11.8 GFLOP/s, 16.9–17.4% of the measured 2-core roof |
| Final loss after 20 fresh steps | 5.7953 (identical across the A/B runs below) |
| GEMM 256³, `uenr bench` | **20.2–22.0 GFLOP/s** across runs |
| GEMM microbench NN/NT/TN 256³ | 18.6 / 19.8 / 15.5 GFLOP/s |
| GEMM microbench NN/NT/TN 384³ | 18.7 / 20.1 / 12.8 GFLOP/s |
| Decode, 20 greedy tokens | 586–587 tok/s (local checkpoint, includes prefill/setup) |
| Decode, 128-token request | 1010 tok/s (107 tokens emitted, local checkpoint) |
| Peak RSS, pretrain 1 step | **215,612–215,748 KiB** (Adafactor default) |
| 100-step fresh pretrain | train loss 4.2821; val loss **4.1572**, ppl **63.89** (8×512, 16,384 tokens) |
| Model size | 8,451,456 total / 1,275,264 active params, 32.2 MiB fp32 |

These figures are from the v0.1.0 release-validation session; repeated runs
vary by about ±2%. The 100-step row is a separate fresh-model run from the
20-step throughput rows, and its validation perplexity depends on the eval
window (8×512 here). Historical item-by-item A/B measurements and their
commits are recorded in `docs/CHECK_LIST.md` and `docs/CHECK_LIST02.md`; do
not mix numbers across different code states, checkpoints or eval windows.

## Experiments in this pass

### GEMM N-block 96 → 128

`UENR_GEMM_NB` controls packed-panel width.  With 128:

| Metric | NB=96 | NB=128 |
|---|---:|---:|
| Train, 20 steps | 1175 tok/s | 1152 tok/s |
| NN 256³ microbench | 18.64 GFLOP/s | 19.23 GFLOP/s |
| NT 256³ microbench | 19.79 GFLOP/s | 20.65 GFLOP/s |
| TN 256³ microbench | 15.45 GFLOP/s | 15.95 GFLOP/s |

The larger panel helps two microbench shapes but is 0.98× end-to-end, so it
was reverted.  The shipped value remains 96.

### Attention-backward scratch reuse

The CPU backward path previously malloc'd/freed `nthreads · S` floats in
every layer.  It now uses a persistent `UenrActs.attn_dprob` buffer, removing
allocation from the hot loop and making allocation failure a one-time
`acts_init` concern.  The buffer is 64·S floats (≤256 KiB at S=1024).  The
per-layer allocation was too small to show a stable end-to-end percentage,
so no speedup is claimed; the change is retained for correctness/ownership
(the backward path can no longer silently lose gradients) and to avoid
allocator traffic.

### Other candidate changes

- ARNOR/AVX2/FMA paths are intentionally unavailable on this CPU; the build
  profile forbids them.
- Speculative decoding, FlashAttention at S≤256, two-row GEMV, 6×16 GEMM and
  FP16 weights were measured slower in the earlier checklist runs and remain
  off by default (see `docs/CHECK_LIST.md`).

## Reproducing

```bash
make -j2
bench/measure.sh train 20
bench/measure.sh infer 20
bench/measure.sh gemm 256
bench/measure.sh mem
./build/bench/gemmbench
make smoke          # correctness pipeline, temporary directory
```

The absolute values depend on sustained clocks and thermal state; compare
runs made on the same machine in the same session.
