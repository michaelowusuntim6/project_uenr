# Runner architecture, SIMD dispatch and measured speedups

This document records the processor topology of the GitHub Actions runner,
the AVX1 -> AVX2+FMA kernel work, the measured benchmark results, and the
assumptions from the original task that did not survive verification.

## 1. Verified runner topology

The diagnostic job in `.github/workflows/train.yml` now prints:

```
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|L3'
nproc
cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list
grep -m1 'model name' /proc/cpuinfo
./uenr config
```

Observed on 2026-11-18 (two different runs; the host model can change):

```
CPU CORE SOCKET NODE ONLINE MAXMHZ MINMHZ
  0    0      0    0    yes      -      -
  1    0      0    0    yes      -      -
  2    1      0    0    yes      -      -
  3    1      0    0    yes      -      -

Model name:  AMD EPYC 9V74 80-Core Processor   (another run: EPYC 9V45)
CPU(s):      4
Thread(s) per core: 2
Core(s) per socket: 2
Socket(s):   1
L3 cache:    32 MiB (1 instance)
NUMA node0 CPU(s): 0-3
Flags:       ... fma ... avx2 ... bmi1 bmi2 avx512f ...
```

Sibling lists:

```
cpu0: 0-1
cpu1: 0-1
cpu2: 2-3
cpu3: 2-3
```

Interpretation:

- The task's premise is **confirmed**: four logical CPUs, two physical cores,
  two SMT threads per core, one NUMA node.
- CPU 0 and CPU 1 are siblings on physical core 0. CPU 2 and CPU 3 are
  siblings on physical core 1.
- A 2-thread run pinned to CPUs 0 and 1 runs both threads on the **same
  physical core**. The correct 2-thread pins are 0 and 2 (or 1 and 3).
- The existing 4-thread configuration pins the main thread to 0, the first
  worker to 1, and auto-pins the remaining workers to 2 and 3. Despite the
  misleading "0 and 1" comment in the old config, the pool therefore uses all
  four logical CPUs; it does not leave half the machine idle. The comments
  and docs were corrected.
- The host is not necessarily a Dadsv5 VM. `lscpu` reports AMD EPYC 9V74 or
  9V45 under Microsoft's hypervisor, with AVX2, FMA, BMI2 and AVX-512.
  GitHub can schedule jobs on different physical hosts, so the workflow now
  reports the topology on every run instead of hard-coding a host model.

## 2. Kernel audit

`src/ops/*_avx2.c` contain the AVX2+FMA builds of the same sources (compiled
with an `avx2` symbol suffix) so the two paths cannot drift.

| File / function | AVX1 pattern | AVX2+FMA opportunity | Estimated share of pretrain time |
|---|---|---|---|
| `gemm.c` `block_kernel` / `micro_8x8`, `micro_4x8`, `micro_1x8`, `gemv` | 16 mul+add uops per k step; packed B panels; 8 accumulator chains | 8 `vfmadd` per k for 8x8; new 4x16 and 6x16 tiles; FMA GEMV | ~62% (gprof, single-thread) |
| `attn.c` `dot_f`, output `axpy`, `dq/dk/dv` accumulation, softmax row | mul+add dots; scalar `axpy`; scalar `uenr_expf` per score | FMA dots and vector AXPY; 8-wide non-FMA exp for the softmax row | ~17% |
| `swiglu.c` SiLU and gradient elementwise | scalar `uenr_siluf`/`sigmoid` per element | 8-wide vector SiLU and gradient; same polynomial | ~14% combined with scalar exp/sigmoid |
| `norm.c` RMSNorm fwd/bwd, fused RMSNorm+RoPE, residual variants | mul+add reductions and updates | `vfmadd`/`vfmsub` in all vector loops | ~3-5% |
| `softmax_ce.c` CE max/exp/sum/grad | scalar exp per vocabulary element | 8-wide polynomial exp, AVX2 `2^k`, vector sum/normalize | ~3-5% |
| `optim.c` AdamW, L2 norm, `add_scaled` | mul+add | FMA, `vfnmadd` for decoupled weight decay | ~2-4% |
| `rope.c` pair rotation | scalar sin/cos application | 8-wide FMA rotation | <1% |
| `decode.c` INT8/Q4 KV unpack; `quant.c` Q8/Q4 GEMV | 128-bit unpack / scalar int8 dots | AVX2 conversion, 256-bit sign/maddubs/madd, FMA | inference only |

Shares are gprof samples from a 10-step, single-thread pretrain on the
laptop, not wall-clock on the runner. They were used to choose which kernels
to rewrite; the end-to-end result is the acceptance criterion.

## 3. Dispatch design

- `[run] simd = "auto" | "avx1" | "avx2"` and `--simd` select the path.
  `auto` uses `__builtin_cpu_supports("avx2") && ("fma")`.
- `src/core/simd_dispatch.c` owns the process-wide selection. Forcing `avx2`
  on a CPU without both features is an error, never an illegal instruction.
- `src/ops/dispatch_ops.c` holds a function-pointer table for RMSNorm,
  attention, SwiGLU, softmax-CE, AdamW, RoPE and elementwise ops. The table is
  refreshed by `uenr_simd_configure()`; a lazy path covers tests and tools.
- `gemm.c` keeps its pool, scratch, packing and transpose logic and selects
  AVX2 microkernels per block (`4x16` default, `8x8` and `6x16` selectable for
  benchmarking). GEMV has an AVX2 implementation.
- `decode.c` calls AVX2 KV-unpack helpers from `src/model/kv_avx2.c` for INT8
  and Q4 when the runtime level is AVX2.
- The ordinary sources are compiled once with `-march=ivybridge`. The
  `*_avx2.c` files include those sources with `UENR_SIMD_SUFFIX=avx2` and are
  compiled with `-march=x86-64-v3 -mavx2 -mfma`. `make check-isa` checks the
  AVX1 objects for AVX2/FMA mnemonics and checks that the AVX2 objects do
  contain FMA.
- The exp/log/sin/cos polynomials deliberately keep mul+add in both paths
  (`uenr_v_muladd`), so they are bit-identical. FMA is confined to linear
  algebra and elementwise linear combinations. `test_simd_dispatch` records
  the AVX1-vs-AVX2 tolerance (relative L2, not bit equality).

## 4. Measurements

Method: `mode=bench` workflow dispatch. GEMM numbers are the best of three
0.5 s windows per configuration because the hosted runner is shared and a
single window varied by up to 2x between runs. The pretrain numbers are means
over 100 optimizer steps at the shipped shape (8.45M total / 1.28M active,
batch 32, grad accumulation 1, 256-token sequence = 8,192 tokens/step,
TinyStories). The final locked configuration uses batch 64 (16,384
tokens/step); its measured smoke throughput is reported in §9/§10.

### GEMM (GFLOP/s, physical-core count = 2)

| size | path | threads | GFLOP/s | GFLOP/s per physical core | speedup vs same-thread AVX1 |
|---|---|---:|---:|---:|---:|
| 384 | AVX1 | 2 | 127.88 | 63.9 | 1.00x |
| 384 | AVX1 | 4 | 133.03 | 66.5 | 1.00x |
| 384 | AVX2+FMA | 2 | 205.80 | 102.9 | **1.61x** |
| 384 | AVX2+FMA | 4 | 195.15 | 97.6 | **1.47x** |
| 384 | AVX-512 | 2 | 114.89 | 57.4 | 0.90x |
| 384 | AVX-512 | 4 | **229.38** | **114.7** | **1.72x** |
| 1024 | AVX1 | 2 | 30.14 | 15.1 | 1.00x |
| 1024 | AVX1 | 4 | 59.64 | 29.8 | 1.00x |
| 1024 | AVX2+FMA | 2 | 100.65 | 50.3 | **3.34x** |
| 1024 | AVX2+FMA | 4 | 107.32 | 53.7 | **1.80x** |
| 1024 | AVX-512 | 2 | 63.77 | 31.9 | 2.12x |
| 1024 | AVX-512 | 4 | **141.45** | **70.7** | **2.37x** |

Hosted-runner variance is large (earlier runs measured AVX2 384 4t at 188
and AVX1 384 2t at 136 GFLOP/s), so compare paths within one run. AVX2
microkernel alternatives at 2 threads (GFLOP/s, 384 / 1024): 4x16 =
205.80 / 100.65; 6x16 = 102.31 / 60.61; 8x8 = 46.47 / 38.37 (this host
showed heavy thread-2/3 noise). The shipped default is the 4x16 tile;
`--tile 1|2` selects 8x8/6x16 for experiments. The 1024-cube case is
memory/L3-bound and is not evidence of FMA-pipe saturation.

### 100-step pretrain

| path | threads | step-100 tok/s | step-100 GFLOP/s | speedup (same threads) |
|---|---:|---:|---:|---:|
| AVX1 | 2 | 4072 | 40.77 | 1.00x |
| AVX1 | 4 | 4164 | 41.69 | 1.00x |
| AVX2+FMA | 2 | 6984 | 69.92 | **1.71x** |
| AVX2+FMA | 4 | 7463 | 74.71 | **1.79x** |
| AVX-512 | 4 | **7887** | **78.96** | **1.89x** |

Values are the final step of each 100-step run (steady state) at batch 32
(8,192 tokens/step), after the Part C kernel changes. The workflow summary
reports the mean over all 100 steps. The locked batch-64 configuration
measured **7,363 tok/s** in the tuned `mode=smoke` pretrain (run 44383949635,
20 resumed steps, AVX-512 auto).

### Quantized KV decode (512-token context, decode tok/s)

| quantized path | AVX1 decode tok/s | AVX2 decode tok/s | speedup |
|---|---:|---:|---:|
| INT8 K/V | 2746 | 4758 | **1.73x** |
| Q4 K/V | 3213 | 4200 | **1.31x** |

The INT8 path keeps an FP32 query (see correction 3) and converts int8 to
FP32 with AVX2 `vcvtepi8_epi32` + FMA. The Q4 value/scoring path uses 256-bit
`maddubs`/`madd`; its AVX1 baseline already had SSE4 nibble unpacking, so its
gain is smaller. Context 512 tokens, 100 decode steps, 2-thread default.

## 5. Threading decision

Four threads on the two physical cores won both the GEMM and end-to-end
pretrain measurements. The user's instruction to use two threads for
memory-bound phases when GEMM uses four assumed those phases run concurrently
with GEMM. They do not: attention, softmax/CE and the optimizer run on the
same sequential training stream, and they use `gemm.mem_pool`, which defaults
to the full pool. Restricting them to two threads would serialise work that is
currently measured faster with four. `[run] smt_threads` remains available for
an explicit second pool; it is disabled (`1`) because the earlier laptop
measurement was 0.99x. On this runner 4 threads is the measured winner.

The laptop default remains `threads = 2`; its pins 0/2 select the two physical
cores. The cloud config uses four.

## 6. Corrections to the original assumptions

1. **"Pinning 4 GEMM workers to 0,1,2,3 creates SMT contention"** — SMT
   contention exists, but four threads still measured faster than two in both
   GEMM and 100-step pretrain. The real pinning bug is a *two*-thread config
   pinned 0/1: both are siblings of core 0.
2. **"AVX2+FMA should be 2.0-2.5x faster per core"** — measured GEMM
   speedups are 1.36-1.52x (and 1.27x for 2-thread 1024-cube), while the full
   training step is about **1.58x** with 4 threads. The 384-cube 2-thread
   microkernel reaches 96.6 GFLOP/s per physical core, but the theoretical
   2x is not realised end-to-end because attention, softmax/CE, the
   optimizer and MoE dispatch still run partially scalar and the 1024 case
   is shared-L3/memory-bound. The 2-2.5x target was not met; the microkernel
   is not the only bottleneck.
3. **"Use `maddubs` for INT8 KV attention scores"** — the KV *keys* are
   int8, but the query is FP32. An integer dot would require quantizing the
   query, which changes attention scores and model quality. The AVX2 path
   converts int8 to FP32 with `vcvtepi8_epi32` and uses FMA instead. True
   integer `maddubs` is used where both sides are integers: the Q4 KV score
   and Q4 `dot_q4_i8` path.
4. **"Non-temporal stores for large C tiles"** — not enabled. The model's
   GEMM tiles are small and the output is consumed immediately; streaming
   stores add fences and bypass useful cache. No measurement supported them.
5. **"2 threads for memory-bound phases"** — see section 5; phases are
   sequential, and the measured 4-thread configuration is faster.
6. **The old cloud build used `-march=x86-64-v3` for the whole binary.**
   That made the "AVX1" path non-AVX1 and would have crashed on the laptop.
   The build now compiles normal objects with `-march=ivybridge` and only the
   dispatched AVX2 objects with x86-64-v3/AVX2/FMA.

## 7. Reproduce

Local (AVX1 or AVX2 depending on the host CPU):

```bash
make -j4
make test
make check-isa
./uenr bench --size 384  --simd avx1 --threads 2 --pin0 0 --pin1 2
./uenr bench --size 384  --simd avx2 --threads 2 --pin0 0 --pin1 2
./uenr bench --size 1024 --simd avx1 --threads 2 --pin0 0 --pin1 2
./uenr bench --size 1024 --simd avx2 --threads 2 --pin0 0 --pin1 2
```

Cloud: dispatch `mode=bench`. The job builds, runs the diagnostic topology
report, prepares the data, runs the GEMM matrix, four 100-step pretrain runs
(avx1/avx2 x 2/4 threads), and INT8/Q4 KV decode benchmarks, then writes the
tables into the run summary. No checkpoint release is published by `bench`.

## 8. Limitations

- Hosted-runner measurements vary by host, clock and noisy neighbours. The
  workflow uses best-of-three windows and reports raw throughput; compare
  AVX1 and AVX2 within the same run, not across runs.
- AVX-512 is opt-in and auto-selected only on CPUs that expose
  AVX512F/VL/BW; the laptop still runs AVX1 and other cloud vCPUs fall back
  to AVX2.
- Attention's softmax row uses the 8-wide non-FMA exp polynomial. The
  per-score softcap branch remains scalar (it is thresholded and rare); the
  final-logit softcap is vectorized.
- The INT8 KV query is FP32 by design; only Q4/Q2 queries are quantized.
- `test_simd_dispatch` skips AVX2/AVX-512 execution on an AVX1-only laptop;
  the cloud build runs the same test with both levels enabled.

## 9. Batch-size sweep (2026-11-18)

Run [44375533674](https://github.com/michaelowusuntim6/project_uenr/actions/runs/44375533674),
200 optimizer steps, AVX2+FMA, 4 threads, fixed LR except the last row.
Val loss is `./uenr eval --batches 8 --seq 256`; tok/s is the mean from the
training log; wall includes the eval.

| config | batch | accum | LR | val loss @200 | mean tok/s | train wall |
|---|---:|---:|---:|---:|---:|---:|
| b16_a2 | 16 | 2 | 0.001 | 3.4840 | 7480 | 110 s |
| b32_a1 | 32 | 1 | 0.001 | 3.2947 | 6599 | 249 s |
| b64_a1 | 64 | 1 | 0.001 | 3.1420 | 6633 | 496 s |
| b64_a1 | 64 | 1 | **0.00141** | **3.0632** | 6668 | 493 s |

Decision rule: the lowest loss wins unless throughput and loss disagree by
less than 0.05. Batch 16 is the throughput winner, but its loss is 0.42 worse
than batch 64, and sqrt(64/32) LR scaling improves batch 64 further. The
locked config is **batch 64, accum 1, LR 1.41e-3**.

Arithmetic correction: batch 64 × seq 256 = **16,384 tokens/step**, not
2,048. The ~26M-token target is therefore **1,600 steps**, not 5,000 (5,000
would be 82M tokens and exceed the 200-minute pretrain timeout at the
measured ~6.7k tok/s). The workflow default is 1600.

## 10. Post-saturation measurement (Part C)

Run 44375533674 vs the pre-saturation run 44369446640, same 100-step shape
(batch 32, 8,192 tokens/step):

| metric | before | after | delta |
|---|---:|---:|---:|
| AVX2 4t pretrain tok/s | 6957 | 7463 | **+7.3%** |
| AVX1 4t pretrain tok/s | 3942 | 4164 | +5.6% (host noise) |
| AVX2 384-cube 4t GFLOP/s | 195.47 | 195.15 | unchanged (GEMM untouched) |

The changes were: vector softmax-row max (exact), final-softcap vectorization
plus an order fix (the forward loop previously capped the stale logits buffer
before the projection), and the earlier AXPY/SiLU vectorization. The
attention AXPY and SwiGLU SiLU were already vectorized, so the remaining win
was smaller than the 5-10% target; +7.3% is within it.

## 11. AVX-512 decision (Part D)

Same run, 4 threads:

| kernel | AVX2+FMA | AVX-512 | speedup |
|---|---:|---:|---:|
| GEMM 384 | 195.15 GFLOP/s | **229.38** | **1.18x** |
| GEMM 1024 | 107.32 GFLOP/s | **141.45** | **1.32x** |
| 100-step pretrain | 7463 tok/s | **7887** | **1.06x** |

The AVX-512 4x16 zmm kernel wins at 4 threads (≥1.05x at 384-cube), while
AVX-512 at 2 threads is slower (114.89 vs 205.80) because the wider
registers/double-pumped execution lose to the AVX2 tile. Decision: keep the
third tier, and make `auto` prefer AVX-512 when AVX512F/VL/BW are present;
`avx512` can also be forced. The laptop stays AVX1. This is a win, not a
negative result, but the end-to-end gain is only 6%, so AVX2 remains the
portable baseline.

## 12. Negative results (Part C/D follow-up)

- 8x8 and 6x16 AVX2 tiles are slower than 4x16 on this host (see §4); the
  earlier 6x16/8x8 numbers were within noise but never won.
- AVX-512 at 2 threads loses to AVX2; it is only enabled for ≥3 threads in
  practice (the dispatch does not enforce this because thread count is a
  runtime choice; the default 4t is the winning case).
- Batch 64 lowers raw tokens/sec vs batch 16 (6668 vs 7480) but wins the
  loss rule; batch 5000 steps is not the right budget at 16k tokens/step.
