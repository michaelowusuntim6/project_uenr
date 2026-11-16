# Build, Test, Benchmark

## Requirements

- Linux (POSIX `mmap`, `pthread`, `sched_setaffinity`, `readlink`).
- `gcc` or `clang` with AVX/F16C support (any modern gcc; tested with
  gcc 15.2 on Ubuntu 26.04).
- `make`.
- No BLAS, no Python at runtime, no third-party libraries. `-lm` and
  `-pthread` are the only link flags.

CPU: the code is tuned for Intel Ivy Bridge (i5-3210M), but `-march=native`
plus the scalar fallbacks make it work on any x86-64 with AVX. If the CPU has
AVX2/FMA the compiler may still not use FMA contraction because we never pass
`-ffast-math`; the formulas remain valid.

## Targets

```bash
make            # build ./uenr and ./uenr-menu
make -j2        # parallel build
make test       # build and run every tests/test_*.c
make bench      # build bench/gemmbench, bench/cpubench, bench/threadbench
make check      # all + tests + bench build
make lto        # ./uenr-lto with -flto (measured +1%)
bench/pgo.sh    # ./uenr-pgo with profile-guided optimization
make clean
```

Compiler flags (from the Makefile):

```
-O3 -march=native -mavx -mf16c -std=c11 -Wall -Wextra -Wno-unused-parameter
-fno-math-errno -fno-trapping-math -fstrict-aliasing -pthread -MMD -MP
```

- `-MMD -MP` generates header dependencies. This is important: without it,
  changing a struct in `config.h` leaves stale `.o` files compiled against the
  old layout (this actually caused a `tensor sum 0` failure during
  development). Clean rebuilds are reproducible.
- No `-ffast-math`, no `-Ofast`: we want reproducible fp32 and no assumptions
  about FMA or finite math.
- No OpenMP: threading is a hand-written two-worker pool pinned to physical
  cores.

## Tests

| Test | What it checks |
|---|---|
| `test_config` | TOML parse, derived shapes, arrays. |
| `test_math` | exp/sigmoid/silu/rsqrt against reference values. |
| `test_gemm` | NN/NT/TN/TT and GEMV vs naive, many shapes. |
| `test_ops` | RMSNorm fwd/bwd vs reference + finite difference, RoPE round-trip, GQA attention vs scalar reference, fused CE loss/gradient, schedule curve. |
| `test_tokenizer` | BPE training, round-trip, special ids, tool-call atomics, SFT masking. |
| `test_gradcheck` | Finite-difference gradient check for every parameter family on dense and single-expert models. |

Expected: each prints `all passed`; `make test` stops on the first failure.

## Benchmarks

```bash
./uenr bench --size 256        # NN GEMM, 2 threads
./build/bench/gemmbench        # NN/NT/TN for 64/128/256/384
./bench/cpubench               # CPU roofs (compute, DRAM, L2)
./bench/threadbench            # 1 vs 2 thread scaling
```

Measured on the target i5-3210M, 2.2 GHz sustained:

| Kernel | Result |
|---|---|
| AVX FP32 compute roof | 33.9 GFLOP/s per core, 67.8 two-core |
| DRAM triad, 1 / 2 threads | ~10.5 / 13.4 GB/s |
| Naive GEMM 256³ | ~6.3 GFLOP/s |
| UENR GEMM 256³ | ~15–24 GFLOP/s (shape/cache dependent) |
| End-to-end pretrain | ~880 tok/s, ~9 GFLOP/s, ~13% MFU |

## Troubleshooting

**`cannot load config`** — run from the project root, or use
`--config /path/to/uenr.toml`. The menu changes to its own directory
automatically.

**`prepare: cannot load data/train.bin`** — run `./uenr tok-train` then
`./uenr prepare`.

**Weird results after pulling changes** — `make clean && make -j2`. Header
dependency files (`.d`) should prevent this, but a fully clean build is the
canonical state.

**NaN checkpoints** — the AdamW `v·rsqrt(v)` zero-gradient bug is fixed. To
audit any checkpoint: read the file, skip the 80-byte header, and scan the
fp32 values for NaN/Inf (see the development notes in `docs/ROADMAP.md`).
