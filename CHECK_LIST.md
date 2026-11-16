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
| 6 | FlashAttention / online softmax (no O(S²) probs) | 1.20× + mem | [x] | at S=256: **0.90× speed** (1189→1072 tok/s), **1.08× less RAM** (331.8→307.1 MiB); default off, enable for long context/memory | pending |
| 7 | FP16/BF16 weight storage, FP32 compute/master | 1.25× | [~] | tested and **reverted**: train 1181→1181 (**1.00×**), RSS 331.7→348.3 MiB (worse), decode 67.6→69.9 (**1.03×**), 20-step loss 7.5826→7.6654 (updates lost below fp16 resolution). Revisit as fp16-only inference or activation fp16 in #28 | n/a |
| 8 | Fused/chunked logits+CE; reuse logits buffer for dlogits | 1.08× | [x] | train **1.03×** (1189→1228 tok/s); peak RSS 331.8→323.6 MiB (**1.03× less**); in-place CE test added | pending |
| 9 | Sampled/adaptive softmax (negative sampling) | 1.30× | [x] | train **1.06×** (1228→1305 tok/s) with 256 negatives; default off (quality-biased loss, not comparable; huge grad norms are clipped); full-negative test matches full CE exactly | pending |
| 10 | LTO + PGO + compiler flag tuning | 1.08× | [x] | LTO **1.01×** (1228→1243 tok/s); PGO **0.98×** (1208, no gain). Targets `make lto` / `bench/pgo.sh` added; default build unchanged | pending |
| 11 | OS tuning: performance governor, huge pages, SMT policy | 1.10× | [ ] | | |
| 12 | mmap/prefetch/no-copy data pipeline | 1.03× | [ ] | | |
| 13 | Micro-batch/seq/cache-block autotune | 1.08× | [ ] | | |
| 14 | Store SwiGLU gate/up, no backward recompute | 1.06× | [ ] | | |
| 15 | Faster expert gather/scatter / counting-sort grouping | 1.05× | [ ] | | |
| 16 | Lower-overhead thread pool barrier | 1.03× | [ ] | | |
| 17 | Fuse optimizer/clip/weight-decay passes | 1.02× | [ ] | | |
| 18 | Use SMT for memory-bound phases only | 1.08× | [ ] | | |

### B. Inference / decode

| # | Item | Est. | Status | Measured xN | Commit |
|---|---|---|---|---|---|
| 19 | GQA KV cache incremental decode | 6× @512 | [ ] | | |
| 20 | INT8 KV cache with scales | 1.4× / 4× RAM | [ ] | | |
| 21 | INT8/INT4 weight quantization for inference | 2.0× | [ ] | | |
| 22 | Speculative decoding (prompt-lookup + draft option) | 2.0× | [ ] | | |
| 23 | Continuous/batched inference | 1.5–3× tput | [ ] | | |
| 24 | FlashAttention prefill path | 1.2× long | [ ] | | |
| 25 | Fused GEMV + top-k/top-p/sampling | 1.1× | [ ] | | |
| 26 | GEMV layout / multi-row vector path | 1.1× | [ ] | | |
| 27 | Sliding-window/attention sinks (long context) | long ctx | [ ] | | |

### C. Memory

| # | Item | Est. | Status | Measured xN | Commit |
|---|---|---|---|---|---|
| 28 | FP16 weights/activations memory path | 1.8× less | [ ] | | |
| 29 | FlashAttention removes O(S²) storage | 2–4× less | [ ] | | |
| 30 | Gradient checkpointing (recompute blocks) | 2–3× less | [ ] | | |
| 31 | 8-bit Adam / Adafactor-style optimizer state | 1.6× less | [ ] | | |
| 32 | LoRA/QLoRA memory path (LoRA exists; add QLoRA) | 3–10× less | [ ] | | |
| 33 | Quantized KV cache memory | 2–4× less | [ ] | | |
| 34 | Chunked logits/CE removes full dlogits | 1.2× less | [ ] | | |
| 35 | Arena/pool reuse (done; verify no regressions) | 1.05× | [x] | measured baseline | 6ba72a1 |

### D. Quality / tokenizer / data

| # | Item | Est. quality | Status | Measured xN | Commit |
|---|---|---|---|---|---|
| 36 | Production BPE: regex pretokenization, byte fallback, 8k vocab, large sample, fast trainer | +10–30%, 1.2–1.5× effective speed | [ ] | | |
| 37 | Full/large corpus (100 MB–1.9 GB) | +50–200% | [~] deferred (time) | | |
| 38 | More steps/epochs | +20–100% | [~] deferred (time) | | |
| 39 | Scale model to 20–50M total | +30–100% | [~] deferred (time) | | |
| 40 | Streaming prepare for a larger corpus (code) + 200 MB run | + data | [ ] | | |
| 41 | Better SFT/tool-call dataset generator | +30–100% tool | [ ] | | |
| 42 | Trainable pairwise reward model for DPO/PPO/GRPO | +30–100% align | [ ] | | |
| 43 | Hyperparameter sweep / muP-style tuning | +10–30% | [ ] | | |
| 44 | z-loss, weight EMA/SWA, dropout option | +5–15% | [ ] | | |
| 45 | Distillation from teacher | +50–200% | [~] deferred | | |
| 46 | YaRN/long-context RoPE scaling + window | enables >512 | [ ] | | |
| 47 | Evaluation harness: val loss/perplexity, generation, tool-call metrics | essential | [ ] | | |
| 48 | Decoding constraints: repetition penalty, min-p, no-repeat n-gram, tool JSON grammar | +10–30% gen | [ ] | | |

### E. DOCUMENT.md architecture (MQA + cross-layer KV sharing + hybrid attention)

| # | Item | Est. | Status | Measured xN | Commit |
|---|---|---|---|---|---|
| 49 | Force `kv_heads=1`; compensate to exactly 8,474,560 total / 1,298,368 active (`ffn=277` + per-layer LayerScale `[C]`) | 2× KV, same size | [ ] | | |
| 50 | Cross-layer KV sharing (`num_kv_shared_layers`, donor mapping by local/global type) | ~2× KV/weights | [ ] | | |
| 51 | Hybrid sliding-window + global attention with ring-buffer KV | long ctx | [ ] | | |
| 52 | p-RoPE (partial rotary, full-head denominator), V-norm (no scale), QKV norm order | stability | [ ] | | |
| 53 | Attention logit softcapping + final logit softcapping | stability | [ ] | | |
| 54 | Four-norm residual block (pre+post attn/FFN) | stability | [ ] | | |
| 55 | Donor gradient accumulation for shared K/V | correctness | [ ] | | |
| 56 | Compact KV cache: active slots only, no shared allocations, ring indices, donor scale sharing | memory | [ ] | | |
| 57 | INT8 KV cache with donor scale sharing (DOCUMENT invariant 6) | 2× KV | [ ] | | |

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
Files: `src/ops/softmax_ce.c`, `src/model/model.c`, `src/train/train.c`.

### 46. Long-context RoPE
- YaRN scaling factors for positions; `rope_scale`, `rope_attn_factor`.
- Combine with sliding window; test at seq 1024.
Files: `src/ops/rope.c`, `src/core/config.c`.

### 47. Eval harness
- `uenr eval --data val.bin --seq N --batches B`: mean CE, perplexity.
- Generation eval: repetition rate, end-token rate, tool-call JSON validity.
- `uenr tok-eval`: bytes/token, tokens/word, round-trip.
Files: new `src/eval/eval.c`, `src/main.c`, `tests/`.

### 48. Decoding constraints
- repetition penalty, no-repeat n-gram, min-p, tool-call JSON grammar
  (brace/bracket state machine).
Files: `src/infer/inference.c`, `src/core/config.c`.

### 49. MQA G=1 + exact compensation
- `kv_heads = 1`; `ffn = 277` (adds 48,384); add per-layer LayerScale `[C]`
  (4×192 = 768). Exact totals: 8,474,560 total / 1,298,368 active.
- KV cache halves; attention params drop 49,152.
- Update `uenr_total_params`, tensor list, forward/backward, checkpoints.
- Retrain tiny models; old checkpoints become incompatible (bump `UMDL` version).
Files: `config/uenr.toml`, `src/core/config.c`, `src/model/model.c`,
`include/uenr/model.h`, `tests/test_gradcheck.c`.

### 50. Cross-layer KV sharing
- `moe/cfg` → `model.kv_shared_layers`; layer plans with donor indices by
  attention type; shared layers have no `wk/wv/k_norm` tensors and reuse the
  donor's K/V buffers/cache.
- L=4 pattern `[local, global, local, global]`, first 2 active, last 2 shared.
- Invariant asserts: `n_active = L - shared`; no allocation for shared.
Files: `src/model/model.c`, `include/uenr/model.h`, `src/infer/kv_cache.c`.

### 51. Hybrid local/global + ring buffer
- Config `sliding_window`, `layer_pattern` (e.g. `"SSSF"` repeated).
- Local layers cap KV at W; ring index `pos % W`; global layers full.
- Attention mask honors the window; test causal equality when W≥S.
Files: `src/ops/attn.c`, `src/infer/kv_cache.c`, `src/model/model.c`.

### 52. p-RoPE + V-norm + QKV order
- `partial_rotary_full` (global p=0.25), local p=1.0; denominator full
  head_dim; unrotated tail unchanged.
- `RMSNormNoScale` on V before cache; K/V norm+RoPE exactly once at the donor.
- Q norm and RoPE every layer (borrowers only Q).
Files: `src/ops/rope.c`, `src/ops/norm.c`, `src/model/model.c`.

### 53. Softcapping
- Attention logits: `C·tanh(z/C)`; backward derivative `1−tanh²`.
- Final logits: `C·tanh(logits/C)` before CE and sampling.
- Config `attn_softcap`, `final_logit_softcap`.
Files: `src/ops/attn.c`, `src/model/model.c`, `src/infer/inference.c`.

### 54. Four-norm residual
- Add `post_attn_norm` and `post_ffn_norm`; wiring exactly as DOCUMENT §3.5.
- New tensors and backward; gradcheck.
Files: `src/model/model.c`, `include/uenr/model.h`.

### 55. Donor gradient accumulation
- Per active-slot `dk_acc/dv_acc`; every layer using a slot adds its attention
  gradients; the donor layer consumes the accumulator before WK/WV backward.
- Gradcheck with shared layers.
Files: `src/model/model.c`.

### 56. Compact KV cache
- Cache slots only for active layers; shared layers point to donor slots.
- Ring indices for local, linear for global; no `wk/wv/k_norm`/cache for shared.
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
| 10 | LTO 1243 tok/s (**1.01×**); PGO 1208 (**0.98×**); optional targets added | pending |
