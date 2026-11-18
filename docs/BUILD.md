# Build, Test, Benchmark

## Requirements

- Linux (POSIX `mmap`, `pthread`, `sched_setaffinity`, `readlink`).
- `gcc` or `clang` with AVX/F16C support (tested with gcc 15.2 and clang 21
  on Ubuntu 26.04).
- `make`; `objdump` for the optional ISA assertion.
- No BLAS, no Python at runtime, no third-party libraries. `-lm` and
  `-pthread` are the only link flags.

CPU: the default build targets Intel Ivy Bridge (`-march=ivybridge`, i.e.
AVX1 + F16C with AVX2 and FMA disabled). It runs on any x86-64 CPU that has
AVX and F16C. There is no runtime ISA dispatch; the minimum supported ISA is
therefore AVX1 + F16C. `make check-isa` disassembles both binaries and fails
if any AVX2/FMA mnemonic appears. If you build on a newer machine for that
same machine, use `make native` (which sets `MARCH=native`); do not ship that
binary to the Ivy Bridge laptop.

## Targets

```bash
make            # build ./uenr and ./uenr-menu for Ivy Bridge
make -j2        # parallel build
make native     # build for the machine doing the build (MARCH=native)
make CC=clang   # build with Clang using the same target profile
make test       # build and run every tests/test_*.c
make bench      # build bench/gemmbench, bench/cpubench, bench/threadbench
make check      # all + tests + benches + ISA check
make check-isa  # disassemble and reject AVX2/FMA instructions
make smoke      # end-to-end pipeline in a temp directory
make lto        # ./uenr-lto with -flto (measured +1%)
bench/pgo.sh    # ./uenr-pgo with profile-guided optimization
bench/governor.sh performance   # needs sudo; restores turbo if governor-limited
make clean
```

Compiler flags (from the Makefile; `MARCH` defaults to `ivybridge`):

```
-O3 -march=$(MARCH) -mavx -mf16c -std=c11 -Wall -Wextra -Wno-unused-parameter
-fno-math-errno -fno-trapping-math -fstrict-aliasing -pthread -MMD -MP
```

`-march=ivybridge` enables AVX1 and F16C and explicitly leaves AVX2/FMA
disabled; `-mavx -mf16c` make the requirement explicit. Override with
`make MARCH=native` only for local builds.

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
| `test_config` | Strict TOML parsing, invalid key/value rejection, derived shapes, arrays, donor-type validation. |
| `test_math` | exp/sigmoid/silu/rsqrt against reference values. |
| `test_gemm` | NN/NT/TN/TT and GEMV vs naive, many shapes, allocation-free fallback path. |
| `test_ops` | RMSNorm fwd/bwd vs reference + finite difference, RoPE round-trip, GQA attention vs scalar reference, fused CE loss/gradient, supplied scratch, schedule curve. |
| `test_tokenizer` | BPE training, round-trip, special ids, tool-call atomics, NFC-lite table, SFT masking. |
| `test_sft` | Next-token label alignment, assistant/user/tool masking, truncation, padding, configurable ignore index. |
| `test_pairs` | Malformed/oversized preference JSONL safely skipped. |
| `test_align` | DPO finite-difference gradient, PPO clipping regions/signs, KL direction, GRPO. |
| `test_align_ckpt` | DPO/RL reference fallback and fail-closed policy. |
| `test_optim` | Adam/Adafactor/Muon/INT8-Adam save/restore, idempotent init, step and kind checks. |
| `test_resume` | Periodic checkpoint discovery without a final model, header-step ranking, corrupt-file rejection. |
| `test_thread` | Worker count/pinning resolution when more threads are requested than explicit CPU ids. |
| `test_gradcheck` | Finite-difference gradient check for every parameter family on dense and single-expert models. |

Expected: each prints `all passed`; `make test` stops on the first failure.

A sanitizer build is reproducible with:

```bash
make -j2 BUILD=build-asan OUT=uenr-asan MENU_OUT=uenr-menu-asan \
  EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1"
make -j2 BUILD=build-asan EXTRA_CFLAGS="..." test
ASAN_OPTIONS=detect_leaks=1 scripts/smoke.sh ./uenr-asan
```

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
