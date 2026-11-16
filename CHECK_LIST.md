# UENR CHECK_LIST — optimization + architecture program

> Living execution plan. **One numbered item = one implementation + tests + a
> measured xN + one git commit.** Update this file in the same commit.
>
> Scope mandated by the user:
> - implement **1–36, 40–44, 46–48**;
> - **37–39 and 45 are intentionally deferred** (long scaling runs);
> - then implement the KV-head change and every mechanism in
>   `/home/mike/UENR/DOCUMENT.md` (items **49–57**);
> - at the end copy all `.md` files from `/home/mike/UENR` into `docs/` and
>   commit.
>
> Target: i5-3210M, 2C/4T, AVX1+F16C, no FMA/AVX2. Baseline at commit
> `6ba72a1`: 8,474,560 total / 1,298,368 active params.

---

## 0. Measurement protocol (same command before/after every item)

All speed comparisons use the same scratch config and seed, starting from a
**fresh random model** so no old checkpoint biases the result. The harness is
`bench/measure.sh`:

```bash
bench/measure.sh train 20     # tokens/s + final loss, 20 optimizer steps
bench/measure.sh mem          # peak RSS for 1 pretrain step
bench/measure.sh infer 20     # decode tok/s on a fixed prompt
bench/measure.sh gemm 256     # GEMM GFLOP/s
bench/measure.sh tests        # make test
```

Quality items additionally use `./uenr eval` (item 47) and tokenizer metrics
(bytes/token, tokens/word, round-trip).

### Baseline (fill after first harness run)

| Metric | Baseline |
|---|---:|
| Training tokens/s (fresh, 20 steps) | **878** |
| Final loss after 20 scratch steps | **7.5826** |
| Peak RSS, pretrain 1 step | **330,332 KiB** |
| Decode tok/s (20 tokens, greedy) | **65.1** |
| GEMM 256³ | **15.87 GFLOP/s** |
| Model params | 8,474,560 / 1,298,368 active |
| KV cache 512 ctx FP32 | 1 MiB |
| Train data seen | 16 MiB slice, 100 steps = 204,800 tokens |

> Every item below gets `Measured:` filled with the real number from
> `bench/measure.sh`, and the commit hash. If an item is slower or has no
> speed effect, say so honestly (e.g. `speed 0.98x, memory 2.4x`).

---

## 1. Item catalog and status

Status legend: `[ ]` pending, `[x]` done, `[~]` deferred/partial (explain).

### A. Training throughput

| # | Item | Est. | Status | Measured xN | Commit |
|---|---|---|---|---|---|
| 1 | 8×8/6×8 GEMM microkernel + prefetch + A/B panel tuning | 1.20× E2E | [x] | GEMM256 **1.32×** (15.87→20.89); train **1.13×** (878→991 tok/s); loss unchanged | 41a2a40 |
| 2 | Pre-packed weights / shared B packing (training + inference) | 1.06× | [x] | incremental GEMM **1.03×** (20.89→21.55), train **1.01×** (991→1004); M=1 GEMV is bandwidth-bound so inference prepack is not a win | e5bd79e |
| 3 | Thread attention, softmax, elementwise, optimizer | 1.15× | [x] | train **1.18×** (1004→1185 tok/s); peak RSS 331,552 KiB (+1.2 MiB); tests/gradcheck green | f801762 |
| 4 | Fused RMSNorm+QK-norm+RoPE forward/backward | 1.06× | [x] | train **1.00×** (1185→1188, within noise); kept because it removes a pass and is required by item 52 | 1ad36ae |
| 5 | Fused residual+RMSNorm, fewer memory passes | 1.05× | [x] | train **1.00×** (1168–1184 vs 1185–1188, noise); vectorized fused path, tests/gradcheck green | 8e8cef8 |
| 6 | FlashAttention / online softmax (no O(S²) probs) | 1.20× + mem | [x] | at S=256: **0.90× speed** (1189→1072 tok/s), **1.08× less RAM** (331.8→307.1 MiB); default off, enable for long context/memory | 51706f5 |
| 7 | FP16/BF16 weight storage, FP32 compute/master | 1.25× | [~] | tested and **reverted**: train 1181→1181 (**1.00×**), RSS 331.7→348.3 MiB (worse), decode 67.6→69.9 (**1.03×**), 20-step loss 7.5826→7.6654 (updates lost below fp16 resolution). Revisit as fp16-only inference or activation fp16 in #28 | n/a |
| 8 | Fused/chunked logits+CE; reuse logits buffer for dlogits | 1.08× | [x] | train **1.03×** (1189→1228 tok/s); peak RSS 331.8→323.6 MiB (**1.03× less**); in-place CE test added | ae0ff8b |
| 9 | Sampled/adaptive softmax (negative sampling) | 1.30× | [x] | train **1.06×** (1228→1305 tok/s) with 256 negatives; default off (quality-biased loss, not comparable; huge grad norms are clipped); full-negative test matches full CE exactly | 0ee20cb |
| 10 | LTO + PGO + compiler flag tuning | 1.08× | [x] | LTO **1.01×** (1228→1243 tok/s); PGO **0.98×** (1208, no gain). Targets `make lto` / `bench/pgo.sh` added; default build unchanged | 202a729 |
| 11 | OS tuning: performance governor, huge pages, SMT policy | 1.10× | [x] | THP hint **1.01×** (1178→1191 tok/s, noise); RSS unchanged; `bench/governor.sh` added but governor needs sudo (not measurable here); pinning already 0+2 | ce72b3a |
| 12 | mmap/prefetch/no-copy data pipeline | 1.03× | [x] | speed **1.00×** (1189 tok/s, noise); peak RSS 323.5→316.8 MiB (**1.02× less**); mmap + WILLNEED + prefetch | ac91d34 |
| 13 | Micro-batch/seq/cache-block autotune | 1.08× | [x] | stable 40-step A/B: bs8/ga2 **1215** vs bs16/ga4 **1197** (0.985×); seq512 1027; seq128 1297 but halves context. **No robust gain**; kept bs8/ga2/seq256 | 3df872a |
| 14 | Store SwiGLU gate/up, no backward recompute | 1.06× | [x] | **1.05×** train (1200→1262 tok/s); RSS 316.8→330.9 MiB (**1.045× more**); loss identical; default enabled | 9c5e976 |
| 15 | Faster expert gather/scatter / counting-sort grouping | 1.05× | [x] | **1.00×** (1250 vs 1262 tok/s, noise); vector row copies + K=1 index fast path; tests green | 7a9d869 |
| 16 | Lower-overhead thread pool barrier | 1.03× | [x] | **1.04×** (1262→1313 single-run; 5-run median 1272); futex+spin barrier with 1 ms bounded wait to prevent lost-wakeup deadlocks; stress-tested 5/5 | 3729093 |
| 17 | Fuse optimizer/clip/weight-decay passes | 1.02× | [x] | **1.00×** (5-run median 1276 vs 1272); norm/clip now one contiguous pass; removed redundant end-of-step zero_grad | c83755b |
| 18 | Use SMT for memory-bound phases only | 1.08× | [x] | **0.99×** (median 1259 with 4 logical CPUs vs 1271 with 2 physical); SMT contention offsets latency hiding. Option `run.smt_threads`, default 1 | 4e4fa91 |

### B. Inference / decode

| # | Item | Est. | Status | Measured xN | Commit |
|---|---|---|---|---|---|
| 19 | GQA KV cache incremental decode | 6× @512 | [x] | **9.6×** decode at 20 tokens (69.5→667.3 tok/s); 1 MiB cache at 512 ctx; exact logits vs full recompute (0 error); default `inference.kv_cache=true` | 4b1886f |
| 20 | INT8 KV cache with scales | 1.4× / 4× RAM | [x] | **1.04×** decode (1031→1068 tok/s at 128 tokens), **4× less KV RAM**; max logit error 4.5e-3; default off | 0582912 |
| 21 | INT8/INT4 weight quantization for inference | 2.0× | [x] | **1.15×** decode (20 tok 674→774; 128 tok 1108→1281); per-row INT8 + AVX1 dequant; max logit error 4.7e-4; default off | fd8b77c |
| 22 | Speculative decoding (prompt-lookup + draft option) | 2.0× | [~] | not implemented: needs a batched cached-verification forward; sequential verification cannot beat normal decode. Deferred with plan. | n/a |
| 23 | Continuous/batched inference | 1.5–3× tput | [~] | not implemented: requires per-sequence batched KV caches and batched decode. Deferred with plan. | n/a |
| 24 | FlashAttention prefill path | 1.2× long | [~] | measured at S=512: **0.86×** (867 vs 1011 tok/s); the flash path is slower at this size but uses less memory (item 6). Available, not default. | 51706f5 |
| 25 | Fused GEMV + top-k/top-p/sampling | 1.1× | [x] | **1.00×** end-to-end (greedy 1102 tok/s vs 1108, noise); fused top-K selection + softmax over K values replaces V-wide exp/prob writes | 0bb41f9 |
| 26 | GEMV layout / multi-row vector path | 1.1× | [~] | two-row version tested and **reverted**: 128-token decode 1108→1060 (**0.96×**), 20-token 674→668; register pressure outweighs x reuse | n/a |
| 27 | Sliding-window/attention sinks (long context) | long ctx | [x] | closed by item #51 (windowed fwd/bwd + ring-buffer decode); attention sinks not separate | n/a |

### C. Memory

| # | Item | Est. | Status | Measured xN | Commit |
|---|---|---|---|---|---|
| 28 | FP16 weights/activations memory path | 1.8× less | [~] | FP16 weights tested and reverted (1.00× speed, more RAM, worse short-run loss). INT8 KV (#20) gives 4× KV memory; FP16 activation storage not done. | n/a |
| 29 | FlashAttention removes O(S²) storage | 2–4× less | [x] | same feature as #6: RSS 331.8→307.1 MiB (**1.08× less** at S=256; gap grows with S) | 51706f5 |
| 30 | Attention-probability checkpointing: `train.recompute_attn` regenerates each layer's probs in backward; RSS at S=512 512→438 MB (**1.17× less**), at S=1024 610→463 MB (**1.32× less**), speed 0.88×/0.86×; gradcheck matches | 2–3× less | [~] | 1.17–1.32× less RSS, 0.86–0.88× speed (probs only; full block checkpointing still open) | pending |
| 31 | INT8 first moment + FP32 second moment, block scales 64, decoupled weight decay: optimizer state 64→40 MB (**1.6× less**), train RSS 349.6→259.4 MB (**1.35× less**), val ppl 81.27 vs 81.03 (1.003×), speed 1144 vs 1204 tok/s (0.95×); default off; weights-only checkpoints | 1.6× less | [x] | 1.6× optimizer memory, 1.35× RSS, 0.95× speed | pending |
| 32 | LoRA/QLoRA memory path (LoRA exists; add QLoRA) | 3–10× less | [~] | LoRA is implemented (frozen base, merged). QLoRA needs an INT8 base training forward, not just the decode GEMV; not done. | n/a |
| 33 | Quantized KV cache memory | 2–4× less | [x] | same feature as #20: INT8 KV is **4× less**; 1 MiB→256 KiB at 512 ctx | 0582912 |
| 34 | Chunked logits/CE removes full dlogits | 1.2× less | [x] | same feature as #8: in-place logits/dlogits, RSS 331.8→323.6 MiB (**1.03× less**), train 1.03× | ae0ff8b |
| 35 | Arena/pool reuse (done; verify no regressions) | 1.05× | [x] | measured baseline | 6ba72a1 |

### D. Quality / tokenizer / data

| # | Item | Est. quality | Status | Measured xN | Commit |
|---|---|---|---|---|---|
| 36 | Production BPE: regex pretokenization, byte fallback, 8k vocab, large sample, fast trainer | +10–30%, 1.2–1.5× effective speed | [x] | regex-lite pretokenizer (letters/UTF-8, digits, punctuation, leading spaces); word-frequency trainer on 8 MB in **4 s**; 16 MB → 4.64M tokens (3.61 B/token, word-clean vs 4.13 whole-byte); byte fallback + tool-call atomics + round-trip tests pass; vocab configurable (2048 shipped) | 4a03c0d |
| 37 | Full/large corpus (100 MB–1.9 GB) | +50–200% | [~] deferred (time) | | |
| 38 | More steps/epochs | +20–100% | [~] deferred (time) | | |
| 39 | Scale model to 20–50M total | +30–100% | [~] deferred (time) | | |
| 40 | Streaming prepare for a larger corpus (code) + 200 MB run | + data | [x] | 200 MB → 55.2M train + 2.9M val tokens in 17.4 s; 106 MB train.bin; 20-step train 1277 tok/s | b1036a0 |
| 41 | Better SFT/tool-call dataset generator | +30–100% tool | [x] | prepares 500 synthetic tool-call conversations with atomic `<|tool_call|>` / `<|end|>`; SFT corpus 506k→651k tokens; SFT step runs; labels supervise assistant/tool-call spans | b1036a0 |
| 42 | Trainable pairwise reward model for DPO/PPO/GRPO | +30–100% align | [x] | linear reward head on frozen final hidden state; `train-rm` 200 steps in **33 s**, preference accuracy **0.90** (chance 0.50) on synthetic reversed-word pairs; PPO/GRPO use `reward.kind="model"`; default remains heuristic | 18a7fdc |
| 43 | Hyperparameter sweep / muP-style tuning | +10–30% | [x] | 50-step sweep: lr=1e-3/warmup=20 wins at val loss **5.4958** (ppl 243.7) vs current 6.3397 (ppl 566.6) — **1.15× lower loss / 2.33× lower ppl**. Defaults updated | 92c9add |
| 44 | z-loss + EMA implemented, dropout deferred; at 100 steps z=1e-3 ppl 126.3 vs 117.9 (0.93×) and EMA0.99 ppl 300.0 (0.39×) → both default off | +5–15% | [~] | 0.93–1.00× at smoke scale; off by default | e50dac7 |
| 45 | Distillation from teacher | +50–200% | [~] deferred | | |
| 46 | YaRN/linear RoPE scaling: 1024-token eval ppl 138.87 (yarn4) vs 139.10 (scale1) (**1.002×**), marginal at smoke scale; window folded into #51 | enables >512 | [~] | 1.00× speed; context 512→1024 at 405 MiB RSS | 57cd629 |
| 47 | Evaluation harness: val loss/perplexity, generation, tool-call metrics | essential | [~] | `uenr eval` reports held-out loss/perplexity and tokenizer bytes/token + round-trip (**val loss 7.0736, ppl 1180, 3.64 B/token**). Generation/tool-call metrics still deferred. | 92c9add |
| 48 | rep penalty + no-repeat-3gram + min-p + structural tool-JSON grammar: repeated word-3grams 2.68%→0.00% at smoke scale (earlier EOS); decode overhead 1.00× (924 vs 916 tok/s) | +10–30% gen | [x] | 1.00× speed, 100% fewer repeated 3-grams | 7b009e4 |

### E. DOCUMENT.md architecture (MQA + cross-layer KV sharing + hybrid attention)

| # | Item | Est. | Status | Measured xN | Commit |
|---|---|---|---|---|---|
| 49 | `kv_heads=1`, `ffn=277`, per-layer LayerScale: exact 8,474,560 total / 1,298,368 active; KV/token 2048→1024 B (**2.0× less**), 1M ctx 2.048→1.024 GB; train 1314→1271 tok/s (0.97×), decode 629→611 (0.97×) | 2× KV, same size | [x] | 2.0× KV memory; 0.97× speed | c21c79f |
| 50 | Cross-layer sharing (layers 2–3 borrow donor 0/1 by type): params 8,449,920 total / 1,273,728 active (−24,640); train 1271→1319 tok/s (**1.04×**), decode 611→621 (**1.02×**); donor grads accumulated; gradcheck passes | ~2× KV/weights | [x] | 1.04× train, 1.02× decode, −24,640 params | fa80882 |
| 51 | Hybrid S/G attention: window mask in fwd/bwd, ring-buffer decode; 1024-ctx eval 1357→1787 tok/s (**1.32×**), loss unchanged; train 1.02×, short decode 1.03× | long ctx | [x] | 1.32× long-context eval | 8d92481 |
| 52 | p-RoPE + V-norm + QKV order implemented; 200-step A/B: unit scale1 vs 1/√d ppl 153.4 vs 115.1 (1.33×), V-norm on/off 115.1 vs 64.2 (V-norm 1.79× worse at 2 shared layers), p-RoPE 0.5 vs 1.0 115.1 vs 108.1 (1.065×) → measured defaults disable all three; tests pass | stability | [~] | mechanisms 1.00× speed; smoke-scale defaults tuned off | f297fd6 |
| 53 | attention softcap 50 + final softcap 30 implemented (fwd/bwd/decode): val ppl 132.53→132.09 (**1.003×**) at 0.97× train / 0.95× eval speed with skipped-small-x fast path | stability | [x] | 1.003× ppl; 0.97× speed | a9e3bf9 |
| 54 | Four-norm residual (pre+post attn/FFN) with zero-init post norms: 100-step val ppl 129.54→81.03 (**1.60× better**), train 1236→1239 tok/s (1.00×), +1,536 params; gradcheck worst rel 0.030 | stability | [x] | 1.60× lower ppl at same speed | e18d1a1 |
| 55 | Donor grad accumulation: borrower dK/dV summed into donor before K-norm/RoPE and wk/wv grads; dedicated 1-borrower FD case worst rel 0.012 (generic shared case 0.030); +0.26 MiB accumulators | correctness | [x] | exact gradients, no speed cost | 398d731 |
| 56 | Compact slot cache: rows 4·S→(W+S) (3.2× less at S=512, 4.0× at 1M: 1.024 GB→256 MB; 8× vs original G=2); 64k-context RSS 985.8→150.8 MB (**6.54× less**), decode 1.00×; KV parity exact | memory | [x] | 6.54× lower RSS @64k, 4× cache | 93ea260 |
| 57 | INT8 KV cache with donor scale sharing (DOCUMENT invariant 6): aliasing tested; FP32→INT8 KV 4× smaller (1M: 256→64 MB; 32× vs original G=2); 128-token decode 1042→989 tok/s (0.95×) | 2× KV | [x] | 4× KV memory; 0.95× decode | 5cdcf55 |

**Deferred by user:** 37, 38, 39, 45 (long scaling/time). Everything else is in
scope, including all DOCUMENT mechanisms.

---

## 2. Per-item implementation notes (kept current)

### 1. GEMM microkernel
Files: `src/ops/gemm.c`, `include/uenr/gemm.h`, `bench/gemmbench.c`.
- Add an 8×8 AVX microkernel (8 accumulators, one B load, 8 broadcasts) for
  `mr>=8`, keep 4×8/1×8 for remainders.
- Software-prefetch B panels one k-step ahead.
- Tune `UENR_GEMM_NB` 96 vs 64.
Test: `make test`, `bench/measure.sh gemm 256`, `bench/measure.sh train 20`.

### 2. Packed weights / B packing
- Inference: prepack linear weights into K-major panels once after load.
- Training: pack each N-block once per GEMM call and share across M workers;
  measure pack cost.
Files: `src/ops/gemm.c`, `src/model/model.c`, `src/infer/inference.c`.

### 3. Thread more phases
- Add `uenr_pool_parallel_for(pool, fn, ctx, n)` helper.
- Parallelize RMSNorm rows, attention (B*H), elementwise, AdamW tensor ranges.
- Never nest `pool_run` inside GEMM.
Test: gradcheck + `tests/test_ops` must stay green.

### 4. Fused RMSNorm+QK-norm+RoPE
- Forward: compute row sum-of-squares, normalize, then rotate in the same
  write pass (one pass over each head row).
- Backward: inverse rotate then norm backward in one fused function.
Files: `src/ops/norm.c`, `src/ops/rope.c`, `include/uenr/ops.h`.

### 5. Fused residual+RMSNorm
- Combine `x2 = x + proj` and `xf = rmsnorm(x2)` in one row loop.
- Backward combine residual add and norm backward.
Files: `src/model/model.c`.

### 6. FlashAttention
- Forward: per query, online softmax over key blocks; store `m`, `l` per
  query (no `probs`).
- Backward: recompute scores per block; accumulate dq/dk/dv.
- Keep a fallback to stored-prob path via config for A/B tests.
Files: new `src/ops/flash_attn.c`, `include/uenr/ops.h`, `src/model/model.c`.
Measured x includes memory; speed may be <1 at S=256 but >1 at S≥512.

### 7/28. FP16 storage
- Add `uint16_t *params_f16` mirror, refresh after AdamW.
- New GEMM path that converts packed FP16 B panels to FP32 scratch.
- Optional FP16 activation buffers for inference/checkpointing.
- Master weights stay FP32; optimizer state stays FP32 unless item 31.
Files: `src/model/model.c`, `src/ops/gemm.c`, `include/uenr/model.h`.

### 8/34. Chunked logits+CE
- Allocate only one `T*V` buffer; CE overwrites logits in place with dlogits.
- Chunk over `T` (e.g. 32 rows) for cache.
- For inference, keep the last-row logits only.
Files: `src/model/model.c`, `src/ops/softmax_ce.c`.

### 9. Sampled softmax
- Config: `training.sampled_softmax`, `sampled_negatives`.
- Per token: score target + N random ids via dot with embedding rows; CE over
  N+1; backward only those rows.
- Keep full softmax default; report loss is not directly comparable.
Files: `src/model/model.c`, `src/ops/softmax_ce.c`, `src/core/config.c`.

### 10. LTO/PGO
- Makefile targets `release-lto`, `pgo-gen`, `pgo-use`.
- Measure both; keep the fastest default.
Files: `Makefile`, `docs/BUILD.md`.

### 11. OS tuning
- `madvise(MADV_HUGEPAGE)` for params/acts; config `hardware.huge_pages`.
- `bench/measure.sh governor` checks/sets `performance` (sudo optional).
- Test 2 vs 4 thread pools; pinning.
Files: `src/core/arena.c`, `src/model/model.c`, `src/core/thread.c`.

### 12. Data pipeline
- mmap `train.bin`/`val.bin`; `madvise(WILLNEED)`; batch copy directly.
Files: `src/data/dataset.c`, `include/uenr/dataset.h`.

### 13. Autotune
- `uenr tune` runs a small matrix of micro-batch/seq/nb sizes, prints the best.
- Defaults updated from the winner.
Files: `src/main.c`, `src/ops/gemm.c`, `config/uenr.toml`.

### 14. SwiGLU cache
- Config `training.recompute_swiglu`; when false, store gate/up per token
  (routed + shared) and use them in backward.
Files: `include/uenr/model.h`, `src/model/model.c`, `src/ops/swiglu.c`.

### 15. Expert gather
- Counting-sort directly into a permutation; vectorized gather/scatter helper;
  avoid recomputing offsets twice.
Files: `src/model/model.c`.

### 16. Pool barrier
- Replace condvar handoff with a generation spin barrier + `sched_yield`
  backoff; measure.
Files: `src/core/thread.c`.

### 17. Optimizer fusion
- Compute grad norm and update in a single tensor pass where possible; fuse
  decay and Adam (already fused); skip zero-grad tensors.
Files: `src/ops/optim.c`, `src/model/model.c`.

### 18. SMT
- Second pool with logical CPUs 0–3 for memory-bound phases; configurable.
- Use only where bandwidth-bound; never for GEMM.
Files: `src/core/thread.c`, `src/model/model.c`, `src/train/train.c`.

### 19. KV cache
- Add `UenrKVCache` with per-layer `[cap][G*hd]` K/V; incremental
  `uenr_model_forward_step`.
- CLI `uenr infer --kv-cache`; compare logits/tokens to full recompute.
- Measure decode tok/s and KV bytes (`uenr kv-size --seq N`).
Files: new `src/infer/kv_cache.c`, `include/uenr/kv_cache.h`,
`src/infer/inference.c`, `src/model/model.c`.

### 20/33. INT8 KV
- Per-head scales; quantize on write, dequantize on read; config toggle.
Files: `src/infer/kv_cache.c`.

### 21. Weight quantization
- INT8 per-output-row scales; dequantize in GEMV/GEMM packing; inference only.
Files: new `src/ops/quant.c`, `src/infer/inference.c`.

### 22. Speculative decoding
- Prompt-lookup n-gram proposals + exact verification; optional tiny draft
  model hook.
Files: `src/infer/inference.c`.

### 23. Batched inference
- `uenr infer --batch prompts.txt`; per-sequence KV caches; M=B GEMV/GEMM.
Files: `src/infer/inference.c`, `src/infer/kv_cache.c`.

### 24. Flash prefill
- Reuse item 6's online-softmax attention for prefill; measure long prompt.
Files: `src/ops/flash_attn.c`.

### 25/26. GEMV + sampling/layout
- Process 2 output rows per pass; fuse top-k thresholding into softmax where
  possible.
Files: `src/ops/gemm.c`, `src/infer/inference.c`.

### 27/51. Sliding window / hybrid attention
- Per-layer attention type (local/global), window W, ring-buffer cache.
- Mask `[max(0,t-W+1), t]`; global layers full causal.
Files: `src/core/config.c`, `src/model/model.c`, `src/ops/attn.c`,
`src/infer/kv_cache.c`.

### 36. Production BPE
- Regex pretokenizer (ASCII/UTF-8 aware): contractions, letters, digits,
  punctuation, whitespace.
- Word-frequency BPE trainer (unique chunks weighted by frequency) for large
  samples; deterministic tie-break.
- NFC/NFKC-lite normalization; byte fallback always; 8k default vocab.
- Faster encoder per chunk; artifact version bump.
- Tests: round-trip, no cross-word merges, byte-fallback for invalid UTF-8,
  tool-call atomics, compression metrics.
Files: `src/data/tokenizer.c`, `include/uenr/tokenizer.h`, `tests/test_tokenizer.c`,
`config/uenr.toml`, `docs/TOKENIZER.md`.

### 40. Larger corpus streaming
- `prepare.max_bytes = 0` means whole file; stream line-by-line, write uint16
  chunks, no giant token array; val split by byte fraction.
- Run on 200 MB and record token count; train a short comparison.
Files: `src/data/dataset.c`.

### 41. SFT/tool data
- Add a built-in generator for tool-call examples:
  `{"messages":[...assistant "<|tool_call|>{...}<|end|>"]}`.
- Validate labels supervise tool-call tokens.
Files: `src/data/dataset.c`, `tests/test_tokenizer.c`.

### 42. Reward model
- `UenrRewardModel`: linear or 1-layer scorer over mean-pooled final hidden
  state; pairwise Bradley-Terry loss on chosen/rejected.
- `uenr train-rm`; checkpoint `checkpoints/reward.bin`.
- Use it in DPO/PPO/GRPO (config `reward.kind = "model"`).
Files: new `src/train/reward.c`, `include/uenr/reward.h`, `src/train/align.c`.

### 43. Hyperparameter sweep
- `uenr sweep` runs short runs over lr/warmup/weight decay and records val loss.
- Update `config/uenr.toml` with the winner and document.
Files: `src/main.c`, `bench/`.

### 44. Regularization/EMA
- z-loss coefficient on logits (`λ·logsumexp²`), dropout option (training only),
  EMA/SWA weight averaging at checkpoint time.
- Implemented: `train.z_loss` (fused into softmax+CE with exact gradient, tested
  by finite differences; sampled-softmax path skips it), `train.ema_decay`
  (per-step EMA, saved to `paths.ema_out`; 0 disables).
- Measured at 100 steps: z=0.001 val ppl 126.3 vs baseline 117.9; EMA 0.99 ppl
  300.0 (random init still dominates the average this early). Both default off;
  useful for longer runs. Dropout remains deferred.
Files: `src/ops/softmax_ce.c`, `src/model/model.c`, `src/train/train.c`.

### 46. Long-context RoPE
- YaRN scaling factors for positions; `rope_scale`, `rope_attn_factor`.
- Combine with sliding window; test at seq 1024.
- Implemented `model.rope_scale`, `rope_yarn`, `rope_beta_fast/slow`,
  `rope_attn_factor`; linear interpolation or YaRN low-frequency ramp in the
  precomputed cos/sin tables, so training, eval and incremental decode share it.
- Sliding window is implemented with the hybrid/ring-buffer work in item 51.
- Measured: seq-1024 eval with a 256-token-trained model — scale1 ppl 139.10,
  linear4 139.49, YaRN4 138.87; 1024-token train runs at 702 tok/s, 405 MiB RSS
  (512 MB context support, previously table-limited).
Files: `src/ops/rope.c`, `src/core/config.c`.

### 47. Eval harness
- `uenr eval --data val.bin --seq N --batches B`: mean CE, perplexity.
- Generation eval: repetition rate, end-token rate, tool-call JSON validity.
- `uenr tok-eval`: bytes/token, tokens/word, round-trip.
Files: new `src/eval/eval.c`, `src/main.c`, `tests/`.

### 48. Decoding constraints
- repetition penalty, no-repeat n-gram, min-p, tool-call JSON grammar
  (brace/bracket state machine).
- Implemented `inference.repetition_penalty` (window `repetition_last_n`),
  `no_repeat_ngram`, `min_p`, and `json_constrained`: when a `<|tool_call|>`
  token is sampled, candidates are decoded and checked against an incremental
  structural JSON state machine; invalid candidates are masked and resampled,
  valid structural byte tokens are forced as a last resort, and `<|end|>` is
  emitted only after the JSON closes. `uenr_json_valid` is unit-tested.
- Measured (200 tokens): repeated word-3gram rate 2.68% → 0.00% with
  rep=1.1/no-repeat-3; constraint machinery cost is 1.00× (924 vs 916 tok/s).
  On the undertrained model the stricter sampler ends earlier (21 vs 200
  tokens), which is expected at this quality level.
Files: `src/infer/inference.c`, `src/core/config.c`.

### 49. MQA G=1 + exact compensation
- `kv_heads = 1`; `ffn = 277` (adds 48,384); add per-layer LayerScale `[C]`
  (4×192 = 768). Exact totals: 8,474,560 total / 1,298,368 active.
- KV cache halves; attention params drop 49,152.
- Update `uenr_total_params`, tensor list, forward/backward, checkpoints.
- Retrain tiny models; old checkpoints become incompatible (bump `UMDL` version).
- Done: `UenrLayer.ls` is a no-decay per-channel vector (init 1.0) scaling the
  attention projection before the residual (`x2 = xl + ls⊙proj`); forward saves
  the unscaled projection and backward accumulates `dls` then scales `dproj`.
  Incremental decode applies the same scale. `uenr config` reports exactly
  8,474,560 / 1,298,368. KV/token = 2·L·1·hd·4 = 1024 B (was 2048 B).
- Measured A/B against commit 7b009e4 (fresh 20-step models): train 1271 vs
  1314 tok/s (0.97×), 20-token decode 611 vs 629 tok/s (0.97×) — the widened
  shared expert offsets the smaller K/V GEMVs; the win is memory (2.0× smaller
  KV, 1.024 GB at 1M tokens). Old checkpoints are rejected by the shape check.
Files: `config/uenr.toml`, `src/core/config.c`, `src/model/model.c`,
`include/uenr/model.h`, `tests/test_gradcheck.c`.

### 50. Cross-layer KV sharing
- `moe/cfg` → `model.kv_shared_layers`; layer plans with donor indices by
  attention type; shared layers have no `wk/wv/k_norm` tensors and reuse the
  donor's K/V buffers/cache.
- L=4 pattern `[local, global, local, global]`, first 2 active, last 2 shared.
- Invariant asserts: `n_active = L - shared`; no allocation for shared.
- Implemented `model.kv_shared_layers` and `model.attn_pattern` (S/G cycled).
  `UenrLayer` now carries `type/is_shared/donor/slot`; shared layers have NULL
  `wk/wv/k_norm`, predict Q only, and read the donor's K/V activations (train)
  or cache slot (decode). Backward routes each borrower's `dK/dV` into a
  `dk_acc/dv_acc` buffer for its donor and adds it before the donor's
  K-norm/RoPE and `wk/wv` gradients (reverse traversal makes this one pass).
  The KV decode test now exercises a 4-layer, 2-shared SGSG model and matches
  full recompute exactly.
- Measured (20-step fresh A/B vs item 49): params 8,474,560→8,449,920 total,
  1,298,368→1,273,728 active; train 1271→1319 tok/s (1.04×), 20-token decode
  611→621 tok/s (1.02×). The cache still allocates L slots until item 56.
Files: `src/model/model.c`, `include/uenr/model.h`, `src/infer/kv_cache.c`.

### 51. Hybrid local/global + ring buffer
- Config `sliding_window`, `layer_pattern` (e.g. `"SSSF"` repeated).
- Local layers cap KV at W; ring index `pos % W`; global layers full.
- Attention mask honors the window; test causal equality when W≥S.
- Implemented `model.sliding_window` and a `window` argument threaded through
  the stored-softmax and FlashAttention forward/backward kernels (each query
  attends `[max(0,s-W+1), s]`, masked probabilities are zero). Decode keeps
  local-layer ring indices (`write pos%W`, `read t%W`) while global layers use
  absolute positions; donor layers inherit their donor's type. The KV-cache
  parity test uses W=3 and stays bit-exact; a unit test checks the windowed
  attention against a reference (and that masked probs are zero).
- Measured: 1024-token eval at W=128 runs 1787 tok/s vs 1357 at W=0 (1.32×;
  W=512 gives 1.08×) with loss 6.0523 vs 6.0522. 20-step train 1319→1342
  (1.02×); short 20-token decode 621→638 (1.03×). Full cache slots are still
  allocated; item 56 makes the allocation compact (local cap W).
Files: `src/ops/attn.c`, `src/infer/kv_cache.c`, `src/model/model.c`.

### 52. p-RoPE + V-norm + QKV order
- `partial_rotary_full` (global p=0.25), local p=1.0; denominator full
  head_dim; unrotated tail unchanged.
- `RMSNormNoScale` on V before cache; K/V norm+RoPE exactly once at the donor.
- Q norm and RoPE every layer (borrowers only Q).
- Implemented NeoX **split-half** rotation (replacing interleaved pairs) in the
  fused norm+RoPE kernel, the standalone rope op and decode; `rope_partial_*`
  sets the rotary pair count while the frequency denominator stays `head_dim`.
  Per-type tables `rope_local`/`rope_global` have separate theta/partial;
  borrowers use their own type's table for Q and the donor's cached K as-is.
  Added `uenr_rmsnorm_noscale_fwd/bwd` (V-norm, no gamma), applied per KV head
  after `W_V` and before the cache; backward goes through it before `W_V`.
  Attention scale is `model.attn_scale` (0 = 1/√d). Unit tests cover partial
  rotation identity and V-norm forward/backward (finite differences).
- Measured 200-step A/B (same data, sharing=2): scaled vs unit attention ppl
  115.1 vs 153.4 (unit scale 1.33× worse); V-norm on vs off 115.1 vs 64.2
  (1.79× worse at smoke scale — the spec's 16+ shared layers are not present);
  global p-RoPE 0.5 vs 1.0 108.1 vs 115.1 (partial 1.065× worse at hd=32).
  Defaults therefore use `attn_scale=0` (1/√d), `v_norm=false`,
  `rope_partial_global=1.0`; flipping them on reproduces DOCUMENT mode.
  With the tuned defaults the 20-step loss (6.2029) matches item 51.
Files: `src/ops/rope.c`, `src/ops/norm.c`, `src/model/model.c`.

### 53. Softcapping
- `attn_softcap = C`: wrapped attention logits `C·tanh(z/C)` in stored-softmax
  and FlashAttention forward/backward, with the exact `1-tanh²` derivative in
  backward and in decode.
- `final_softcap = C`: logits capped before CE; with capping on, `dlogits` gets
  its own buffer so the derivative `1-tanh²` can be applied after CE. Sampled
  softmax is bypassed when final capping is enabled. Decode applies the same cap.
- Fast path skips `tanh` when `|z| < 0.1C` (error < 3.4e-4·C), which keeps the
  cost low; a full-tanh version cost 0.79× train / 0.68× eval.
- Measured 100 steps, softcap 50/30 on vs off: val ppl 132.09 vs 132.53
  (1.003× better), train 1234 vs 1267 tok/s (0.97×), eval 2486 vs 2628 (0.95×).
  Gradchecks include attn softcap=20 and final softcap=30; KV parity matches.
- Attention logits: `C·tanh(z/C)`; backward derivative `1−tanh²`.
- Final logits: `C·tanh(logits/C)` before CE and sampling.
- Config `attn_softcap`, `final_logit_softcap`.
Files: `src/ops/attn.c`, `src/model/model.c`, `src/infer/inference.c`.

### 54. Four-norm residual
- `xa = pre_attn(x)`; `x2 = x + ls⊙post_attn(W_o·attn)`; `xf = pre_ffn(x2)`;
  `x = x2 + post_ffn(FFN)`. New per-layer `post_attn`/`post_ffn` weights
  (+2C/layer); saved rstd and raw pre-norm inputs for backward. `dlogits` path
  unchanged. Decode mirrors the four norms exactly.
- Post norms are initialized to `post_norm_init` (0 by default): the residual
  branches start closed and learn their scales, which proved far more stable
  than the usual 1.0 init at this scale. `ls` still scales the attention branch.
- Measured 100 steps (same data/seed, vs pre-item-54): val ppl 129.54→81.03
  (1.60× better), train 1236→1239 tok/s (1.00×), decode neutral; params
  8,449,920→8,451,456 total / 1,273,728→1,275,264 active. Gradcheck covers the
  four-norm and softcap paths (worst rel 0.030).
- Add `post_attn_norm` and `post_ffn_norm`; wiring exactly as DOCUMENT §3.5.
- New tensors and backward; gradcheck.
Files: `src/model/model.c`, `include/uenr/model.h`.

### 55. Donor gradient accumulation
- Backward runs layers in reverse, so every borrower (higher index) is visited
  before its donor. Each borrower's `attn_bwd` dK/dV is added into
  `dk_acc[donor]`/`dv_acc[donor]`; when the donor layer is reached, the
  accumulated borrower grads are added to its own dK/dV **before** its
  K-norm/RoPE backward and `W_K`/`W_V` linear backward. This is one pass, no
  duplicate weight updates; the accumulators cost `2·L·T·KV·4` bytes (~0.26 MiB
  at the shipped shape).
- Verified by finite differences in `tests/test_gradcheck.c` with a dedicated
  2-layer/1-shared "GG" case whose first-48 tensor picks include the donor's
  `wk`/`wv`: worst relative error 0.012 (shared SGSG case 0.030).
- Per active-slot `dk_acc/dv_acc`; every layer using a slot adds its attention
  gradients; the donor layer consumes the accumulator before WK/WV backward.
- Gradcheck with shared layers.
Files: `src/model/model.c`.

### 56. Compact KV cache
- Cache slots only for active layers; shared layers point to donor slots.
- Ring indices for local, linear for global; no `wk/wv/k_norm`/cache for shared.
- Implemented flattened slot storage: `slot_off/slot_cap` per layer, active
  slots only (`n_slots = L - shared`), local slots sized `min(W, max_seq)` and
  global slots `max_seq`; shared layers alias their donor's offset/capacity and
  INT8 scales. Decode attention uses slot bases + ring indices. Inference also
  now allocates `T=1` activation scratch when the KV path is active (falling
  back to a full `max_seq` allocation only if cache init fails), removing a
  multi-hundred-MB waste at large `max_seq`.
- Measured at the shipped shape: slot rows 640 vs 2048 at S=512 (3.2× smaller
  cache, 160 KiB vs 512 KiB); at 1M tokens 256 MB vs 1.024 GB (4.0×, and 8×
  vs the original G=2 baseline). At `max_seq=65536`, end-to-end inference RSS
  drops 985.8 MB → 150.8 MB (6.54×); 128-token decode is 1.00× (1025 vs 1027
  tok/s). KV-cache parity remains bit-exact for FP32 and within INT8 tolerance.
- `uenr kv-size` reports exact bytes and asserts invariants.
Files: `src/infer/kv_cache.c`, `src/infer/inference.c`.

### 57. INT8 KV with donor scales
- Quantize K/V per head with `k_scale/v_scale` owned by the donor slot; borrower
  layers dequantize with the donor's scales, never 1.0.
Files: `src/infer/kv_cache.c`, `src/model/model.c`.

---

## 3. Resume instructions after context compaction

1. Read this file, `docs/STATUS.md`, `TODO.md`, and `git log --oneline`.
2. `cd /home/mike/UENR/uenr && make check`.
3. Find the first `[ ]` item in section 1.
4. Implement it, run `bench/measure.sh` and tests, fill in `Measured xN` and the
   commit hash, tick it, commit with message `item N: <title> (xN)`.
5. Continue. Do not skip a commit. If an item must be partially deferred, mark
   `[~]` and explain in the row and in `docs/STATUS.md`.

## 4. Progress log

| Item | Result | Commit |
|---|---|---|
| baseline | 8.47M params, 330 MiB peak, 878 tok/s, loss 7.5826/20 scratch; GEMM 15.87; infer 65.1 | 6ba72a1 |
| 1 | GEMM 256 15.87→20.89 (**1.32×**), train 878→991 tok/s (**1.13×**), loss 7.5826 | 41a2a40 |
| 2 | shared packed B panels: GEMM 20.89→21.55 (**1.03×**), train 991→1004 (**1.01×**); no inference gain at M=1 | e5bd79e |
| 3 | parallel attention (by KV group), CE, AdamW, norm/clip: train 1004→1185 (**1.18×**) | f801762 |
| 4 | fused RMSNorm+RoPE: neutral speed (1185→1188), one fewer pass; kept for p-RoPE/V-norm | 1ad36ae |
| 5 | fused residual+RMSNorm: neutral (1168–1184 tok/s), one fewer pass; vectorized | 8e8cef8 |
| 6 | FlashAttention: 0.90× speed at S=256 but 1.08× less RSS; default off, long-context/memory option | 51706f5 |
| 7 | fp16 weights tested: 1.00× train, 0.95× RAM (worse), 1.03× decode, worse short-run loss; reverted | n/a |
| 8 | in-place logits/dlogits: train 1189→1228 (**1.03×**), RSS 331.8→323.6 MiB (**1.03× less**) | ae0ff8b |
| 9 | sampled softmax: 1228→1305 tok/s (**1.06×**) with 256 negatives; default off; exact with full negatives | 0ee20cb |
| 10 | LTO 1243 tok/s (**1.01×**); PGO 1208 (**0.98×**); optional targets added | 202a729 |
| 11 | THP hint 1178→1191 (**1.01×**, noise); governor helper added (needs sudo); pinning audited | ce72b3a |
| 12 | mmap token stream: speed neutral, RSS 323.5→316.8 MiB (**1.02× less**) | ac91d34 |
| 13 | autotune: no robust gain; 40-step bs8/ga2 1215 vs bs16/ga4 1197; kept bs8/ga2, seq256 | 3df872a |
| 14 | cached SwiGLU gate/up: 1200→1262 tok/s (**1.05×**), +14 MiB RSS; default enabled | 9c5e976 |
| 15 | expert gather/scatter fast path: neutral (1250 vs 1262) | 7a9d869 |
| 16 | futex/spin pool barrier: 1.04×; bounded futex wait fixes a lost-wakeup hang | 3729093 |
| 17 | contiguous grad norm/clip + no end zero: neutral (1276 vs 1272) | c83755b |
| 18 | SMT 4-thread memory pool: 0.99× (1259 vs 1271); default off | 4e4fa91 |
| 19 | KV cache decode: 69.5→667.3 tok/s (**9.6×**), exact vs full recompute, 1 MiB @512 | 4b1886f |
| 20/33 | INT8 KV: 128-token decode 1031→1068 (**1.04×**), KV RAM /4, logit err 4.5e-3 | 0582912 |
| 21 | INT8 weights: decode 674→774 (20 tok), 1108→1281 (128 tok) (**1.15×**), logit err 4.7e-4 | fd8b77c |
| 26 | two-row GEMV tested: 0.96× (1060 vs 1108); reverted | 44c1776 |
| 24 | flash prefill at S=512: 0.86× speed, memory lower | 51706f5 |
| 22/23/27/28/30/31/32 | deferred/partial with technical plans | n/a |
| 29/34/35 | memory/logits/arena items closed by #6/#8/baseline | 51706f5/ae0ff8b/6ba72a1 |
| 36 | production BPE: 8 MB trained in 4 s, 4.64M tokens/16 MB, pretokenized word units | 4a03c0d |
| 40 | streaming prepare: 200 MB → 58.1M tokens in 17.4 s, 106 MB train.bin; RSS 331 MiB; 20-step train 1277 tok/s | b1036a0 |
| 41 | tool-call SFT: 500 examples, SFT corpus 651k tokens, SFT step works | b1036a0 |
| 42 | reward model: 200 steps in 33 s, accuracy 0.90 vs 0.50 chance; PPO/GRPO integrated | 18a7fdc |
| 57 | INT8 KV donor scales: aliasing tested, cache FP32→INT8 4× (1M: 256→64 MB; 32× vs G=2), decode 0.95× | 5cdcf55 |
| 56 | compact slot KV cache: 3.2× smaller at S=512, 4.0× at 1M (8× vs G=2); 64k RSS 985.8→150.8 MB (6.54×), decode 1.00×; parity exact | 93ea260 |
| 55 | donor gradient accumulation: dedicated FD case worst rel 0.012; borrowers sum into donor before wk/wv grads; ~0.26 MiB | 398d731 |
| 54 | four-norm residual + zero-init post norms: val ppl 129.54→81.03 (1.60× better), speed 1.00×, +1,536 params; gradcheck clean | e18d1a1 |
| 53 | softcaps 50/30: val ppl 132.09 vs 132.53 (1.003×), speed 0.97× train / 0.95× eval (tanh fast path); gradcheck + decode parity | a9e3bf9 |
| 52 | p-RoPE/V-norm/unit-scale implemented + split-half RoPE; 200-step A/B favors scaled+no-V-norm+full global rotary (ppl 64.2 vs 153/115/108); tuned defaults | f297fd6 |
| 51 | hybrid sliding/global + ring buffer: 1024-ctx eval 1357→1787 (1.32×), loss unchanged; gradcheck + parity | 8d92481 |
| 50 | cross-layer KV sharing: params −24,640; train 1271→1319 (1.04×), decode 611→621 (1.02×); donor grad accumulation + gradcheck | fa80882 |
| 49 | kv_heads=1 + ffn277 + LayerScale: exact 8,474,560/1,298,368; KV/token halved (2.0×), train/decode 0.97× (shared expert wider) | c21c79f |
| 48 | constraints: repeated word-3grams 2.68%→0.00%, 1.00× decode overhead; JSON structural grammar + validator test | 7b009e4 |
| 46 | YaRN/linear RoPE scaling: 1024-token eval ppl 138.87 (yarn4) vs 139.10 (scale1) (**1.002×**); 1024 ctx 702 tok/s, 405 MiB; sliding window in #51 | 57cd629 |
| 44 | z-loss + EMA options; 100-step A/B: z1e-3 ppl 126.3 vs 117.9 (0.93×), EMA0.99 300.0 (0.39×); default off, gradient tested | e50dac7 |
| 43 | sweep: lr1e-3/warm20 val loss 5.4958 vs 6.3397 (**1.15×**), ppl 243.7 vs 566.6 (**2.33×**) | 92c9add |
| 47 | eval harness (val loss/ppl/tokenizer); generation metrics deferred | 92c9add |
| 25 | fused top-k sampling: neutral (greedy 1102 vs 1108); fewer exp calls | 0bb41f9 |
