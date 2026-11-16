# UENR CHECK_LIST02 — 24 further pushes (item 20 skipped)

> Living execution plan for the second optimization program. **One item = one
> implementation + tests + a measured xN + one git commit.** Read this first
> after any context compaction.
>
> User instruction: include all 25 items from the previous answer, **skip #20
> (teacher distillation — needs a teacher model / long run)**, commit after each
> item, update every necessary file, prefer formulas that are cheap on this CPU
> (integer/bit ops, Hadamard add/sub, lookup tables, fixed point), and keep the
> system buildable and tested at every step.
>
> Target: i5-3210M, 2C/4T, AVX1+F16C, no FMA/AVX2. Pure C, no dependencies.
> Baseline at the start of this program (HEAD `4801e23`):
> 8,451,456 total / 1,275,264 active params; pretrain ~1,271 tok/s;
> 20-token decode 621 tok/s (full recompute 60); 128-token decode ~1,070;
> 300-step val loss 3.610 / ppl 36.98; training peak RSS 349,680 KiB;
> KV at 1M = 256 MB FP32 / 128 MB FP16 / 64 MB INT8 / 40 MB Q4;
> original GQA KV = 2.048 GB. `make check` green.

## Measurement protocol

Use the same scratch configs/data. Record peak RSS with `/usr/bin/time -v`,
decode with `uenr infer`, prefill/decode with the `kvbench` microbenchmark
(source kept in `bench/` if promoted), and accuracy with held-out val ppl/logit
error/argmax agreement. Report **x vs the stated baseline**, not an invented
one. Negative results are kept and documented; options default off unless they
win.

## Status table

| # | Push | x target | Status | Measured | Commit |
|---|---|---|---|---|---|
| 1 | 4-bit KV integer QK/V + Hadamard rotation + KV QAT | 6–8× KV, 1.1–1.4× long decode | [~] | 6.4× smaller KV (1M 40 MB); QAT-4 mean logit err 0.0223→0.0136, argmax 61→64/64 at val ppl 81.03→81.57 (1.007×); training speed neutral 616/617 tok/s; decode 0.77–0.83× FP32 (no speed win); rotation invariance exact, accuracy neutral | pending |
| 2 | Ternary KV + QAT + mixed precision | 10–16× KV, 1.0–1.3× long | [~] | 1M 24 MB (10.7× smaller); QAT-2 ppl 81.03→81.97 (1.012×); mixed sinks=4/recent=32: mean logit err 0.091→0.039, max 1.45, argmax 63/64; decode 0.32×/0.14× FP32 at 512/2048 (scalar 2-bit unpack) | pending |
| 3 | KV eviction / attention sinks / StreamingLLM | 2–4× beyond window, 1.1–1.5× long | [ ] | | |
| 4 | MLA-style low-rank KV latent in pure C | 2–8× KV, 1.2–1.6× long | [ ] | | |
| 5 | More aggressive cross-layer sharing + learned donor adapters | 1.3–2× KV/weights | [ ] | | |
| 6 | Paged KV + async DRAM/disk offload | unbounded context, 1.2–2× memory | [ ] | | |
| 7 | Prefix/radix cache for shared prompts | 1.2–3× serving throughput | [ ] | | |
| 8 | Speculative decoding with a trained 2-layer draft | 1.5–2.5× decode | [ ] | | |
| 9 | Continuous/paged batched inference | 1.5–2.5× throughput | [ ] | | |
| 10 | Fused decode kernel (RoPE + KV quant + attention + Wo) | 1.1–1.25× decode | [ ] | | |
| 11 | AVX1 integer/assembly GEMM, 6×16 tiles, A-packing | 1.15–1.35× GEMM/train | [ ] | | |
| 12 | INT4/ternary weights with QAT | 2–4× weight memory, 1.1–1.3× decode | [ ] | | |
| 13 | Mixture-of-depths / early exit | 1.2–1.8× decode | [ ] | | |
| 14 | Local + global + random sparse attention | 1.5–3× long-context compute | [ ] | | |
| 15 | Full block gradient checkpointing (single-layer slots) | 2–3× less train RAM, ~0.85× speed | [ ] | | |
| 16 | QLoRA / INT4 base fine-tuning | 2–3× less finetune RAM | [ ] | | |
| 17 | INT8/4-bit training matmul + error feedback | 1.2–1.8× train | [ ] | | |
| 18 | FP16 activation storage with FP32 compute | 1.2–1.4× less train RAM | [ ] | | |
| 19 | 4-bit / Adafactor optimizer state | 2–3× optimizer RAM | [ ] | | |
| 20 | **SKIPPED per user** — distillation from a teacher | — | [—] | needs teacher/long run | n/a |
| 21 | Long-context curriculum + YaRN finetune | 4–8× context at equal ppl | [ ] | | |
| 22 | 16k–32k vocab + NFC/NFKC + code pretokenizer | 1.1–1.3× effective tokens | [ ] | | |
| 23 | Data curriculum / dedup / streaming | 1.1–1.3× data efficiency | [ ] | | |
| 24 | Muon / orthogonalized optimizer | 1.1–1.3× fewer steps | [ ] | | |
| 25 | Combined serving stack (batch + prefix + spec + quant) | 3–6× throughput | [ ] | | |

## Per-item plans

### 1. 4-bit KV integer attention + rotation + QAT — DONE (partial)
- `kv_q4` decode now uses int8 Q + int4 K via `maddubs`/`madd`/`sign_epi8`,
  int32 per-dim V accumulation (p folded with per-row V scale at 1/256),
  optional normalized Hadamard rotation (add/sub only) on Q/K/V and inverse on
  the attention output.
- `training.kv_qat_bits=4` quantize-dequantizes K/V after norm/RoPE with a
  straight-through gradient. `bench/kvbench` and `bench/kvcheck` are committed.
- Measured at context 512/2048/4096: q4 decode 824/434/267 tok/s vs FP32
  993/553/346 (0.83×/0.78×/0.77×); KV@1M 40 MB vs 256 MB (6.4×). QAT-4 over
  100 steps: val ppl 81.57 vs 81.03, training 617 vs 616 tok/s (neutral), but
  q4 mean logit error 0.0223→0.0136 and argmax 61→64/64. Rotation is exact
  (unit test) but did not improve q4 accuracy here (63/64), so default off.
- Verdict: memory/accuracy mode, not a speed win on this AVX1 CPU. The path to
  speed needs a fully integer V path without p-scale rounding or a different
  layout; documented as incomplete versus the 1.1–1.4× target.
- Quantize Q to int8 per head; compute Q·K4 with `maddubs`/`madd`
  (unsigned |q| × signed K with `sign_epi8`), V accumulation in int32 with
  p-quantization (scale 128) and per-dim widening.
- Optional fast Walsh-Hadamard rotation (add/sub only, O(d log d)) after
  Q/K RoPE and V-norm; inverse-rotate attention output; dot/weighted-sum
  invariant, quantization error spread.
- `training.kv_qat_bits=4`: quantize-dequantize K/V after norm/RoPE with a
  straight-through gradient during training so weights adapt.
- Measure logit error, argmax agreement, decode tok/s at 512/2048/4096, and
  val ppl with QAT.

### 2. Ternary KV + mixed precision — DONE (partial)
- `kv_q2`: 2-bit codes {-1,0,+1} packed 4/byte with per-head/row FP32 scale;
  sinks (first `kv_sinks`) and newest `kv_recent` tokens kept FP32, read in
  preference to the packed value; `training.kv_qat_bits=2` is STE ternary QAT.
- Measured: QAT-2 100 steps val ppl 81.97 vs 81.03 (1.012×); with mixed
  sinks=4/recent=32 q2 mean logit error 0.039, max 1.45, argmax 63/64 (without
  mixed: mean 0.091, max 1.77). Cache ~24 MB at 1M (10.7× smaller than FP32).
- Decode is slow: 317 tok/s at 512 and 79 at 2048 (0.32×/0.14× FP32) because
  the 2-bit unpack is scalar. A vectorized pshufb unpack + integer kernel is
  required to reach the 1.0–1.3× target; kept as a memory/accuracy prototype.
- 2-bit packing, values {-1,0,1}, per-head/row scale; QAT via STE.
- Recent `kv_recent` tokens kept FP16 in a small ring; older tokens ternary
  (mixed precision), sinks FP16.
- Measure ppl/logit error and cache bytes.)

### 3. KV eviction / sinks
- StreamingLLM layout for global slots: `sink` first tokens + a ring of the
  last `W` tokens; fixed cache independent of S.
- Decode attention reads two ranges; optional H2O score accumulation.
- Measure long-context memory and generated quality vs full global.

### 4. Low-rank KV (MLA-style)
- Per active layer, learn down `[r, KV]` and up `[KV, r]` matrices; cache the
  latent `r` vector instead of K/V; reconstruct before attention.
- Include in params/gradcheck; compare ppl at rank 8/16 and KV bytes.

### 5. Aggressive sharing + donor adapters
- `kv_shared_layers` up to L-1; optional per-borrower low-rank K/V adapter
  `KV + A_l(B_l K)` (or additive bias) to recover type mismatch.
- Measure params, KV slots, ppl.

### 6. Paged / offloaded KV
- Block allocator (e.g., 64 rows) over a large mmap'ed file for global slots;
  hot pages in RAM, cold pages `madvise`/read on demand.
- Measure max context/RAM and decode penalty.

### 7. Prefix/radix cache
- Cache the rendered system prompt's KV once; clone prefix slots into each
  sequence cache in batch inference; later, trie shared user prefixes.
- Measure prefill time saved and throughput at B=8.

### 8. Draft-model speculation
- Train a 1–2 layer draft with the same tokenizer/config; propose k tokens
  with its cache, verify with the existing span kernel; accept matching target
  samples.
- Measure acceptance and decode x.

### 9. Continuous batching
- Stream prompts into `batch_max` active slots; when a sequence finishes,
  prefill the next prompt into the freed slot without waiting for the batch.
- Measure throughput vs lockstep batch.

### 10. Fused decode kernel
- Fuse per-head score/softmax/V accumulation with q4/int8 unpack and RoPE/Vnorm
  where possible; remove intermediate buffers.
- Measure decode x and logit parity.

### 11. AVX1 GEMM
- A-panel packing, 6×16/6×8 tiles, instruction-scheduled C intrinsics (no FMA),
  prefetch distance tuning; benchmark 128/256/512.
- Measure GFLOP/s and training x.

### 12. INT4/ternary weights + QAT
- Pack signed int4 weights (2/byte) with per-row/group scales; AVX1 dequant
  GEMV; QAT with STE/error feedback for training.
- Measure weight bytes, decode x, ppl.

### 13. MoD / early exit
- Inference early-exit when top-1 margin/entropy is confident after layer 2 or
  3, using the tied head; optional trained per-token gate.
- Measure average layers/token, decode x, quality.

### 14. Sparse attention
- Add local window + global sinks + deterministic random keys per query to the
  full-sequence attention mask/backward; config `attn_sparse_rand`.
- Measure long-context speed and ppl.

### 15. Full block checkpointing
- Refactor the layer forward into a callable block; in backward recompute each
  layer into single-layer-sized slots; keep donor K/V in a compact buffer.
- Measure train RSS at S=512/1024 and speed; gradcheck.

### 16. QLoRA / INT4 base
- Freeze base, quantize to int4/int8, run LoRA forward/backward by dequantizing
  per linear into scratch; save adapters + quantized base.
- Measure finetune RSS at seq 256/512 and loss vs fp32 LoRA.

### 17. Low-precision training matmul
- Simulate INT8/INT4 weights/activations in forward with STE and error
  feedback; optional integer dot kernels.
- Measure train x and ppl vs fp32.

### 18. FP16 activation storage
- Store the largest act buffers (probs/x/q/k/v) as FP16 with F16C convert at
  use; backward reads converted values; measure RSS/speed.

### 19. 4-bit/Adafactor optimizer
- Adafactor factored second moment for matrices + Adam for vectors; or int4
  first moment with block scales.
- Measure optimizer bytes, RSS, ppl, speed.

### 20. SKIPPED
Teacher distillation requires a larger trained teacher and a long run.

### 21. Long-context curriculum + YaRN
- Increase seq_len over training (e.g., 256→512→1024), set rope_scale/YaRN,
  finetune; evaluate at 1024/2048/4096.
- Measure context x at equal ppl.

### 22. Tokenizer 16k–32k + NFC/NFKC + code
- Add NFC-lite composition/decomposition, code-aware pretokenizer (operators,
  indentation), configurable 16k/32k vocab training; measure bytes/token and
  encoder speed on stories+code.

### 23. Data curriculum / dedup
- Exact/near-duplicate line filtering in `prepare`; length/difficulty bucketed
  sampling or progressive sequence length; measure tokens/s and val ppl.

### 24. Muon optimizer
- Momentum + Newton–Schulz orthogonalization for 2-D weights, AdamW for
  vectors/norms; CPU cost measured; compare loss at matched steps/time.

### 25. Combined serving stack
- Enable prefix cache + continuous batching + draft/spec + quantized KV
  together; measure aggregate throughput at B=8/16 and mixed prompts.

## Resume instructions

1. Read this file, `docs/STATUS.md`, `CHECK_LIST.md`, `git log --oneline`.
2. `cd /home/mike/UENR/uenr && make check`.
3. Find the first `[ ]` item.
4. Implement, test, measure, fill status/x/commit, commit `item N2: ...`.
5. Do not skip commits. Mark `[~]` with a concrete blocker if an item cannot
   reach production quality in this hardware/time, but land a real prototype.

## Progress log

| Item | Result | Commit |
|---|---|---|
| start | CHECK_LIST02 created; baseline 4801e23 | c16ed78 |
| 1 | Q4 integer QK/V + rotation + QAT: 6.4× KV, QAT argmax 64/64, decode 0.77–0.83× FP32; memory mode | d519042 |
| 2 | Ternary QAT + mixed precision: 10.7× KV, ppl 81.97, argmax 63/64, decode 0.14–0.32× (scalar unpack); prototype | pending |
