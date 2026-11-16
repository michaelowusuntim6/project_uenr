# Roadmap

## Done

- Measured hardware audit (`docs/AUDIT.md`): 2C/4T Ivy Bridge, AVX1+F16C,
  no FMA/AVX2; 33.9 GFLOP/s/core, 13.4 GB/s two-thread DRAM.
- FMA-free scalar/vector formulas and a 64-byte-aligned tensor arena.
- AVX1 GEMM (8×8 microkernel, packed panels), GEMV, futex/spin pool.
- Full transformer forward/backward: RMSNorm, QK-norm, split-half p-RoPE,
  MQA/GQA attention, SwiGLU, four-norm residual, softcaps.
- Sparse MoE: top-1 routing, shared expert, aux loss, grouped dispatch.
- Byte-level BPE with pretokenization, word-frequency trainer, atomics.
- Streaming `prepare` (200 MB → 55.2M train tokens), SFT/tool/reward/DPO data.
- Pretrain, SFT, LoRA, DPO, PPO, GRPO, trainable reward model.
- KV cache: compact donor slots, cross-layer sharing, sliding ring buffers,
  FP32/FP16/INT8, exact parity tests.
- Inference: top-k/p/min-p, repetition/no-repeat-ngram, constrained tool JSON,
  prompt-lookup speculation, batched multi-sequence decode.
- Eval: val loss/ppl, tokenizer metrics, generation end/repetition/JSON metrics.
- Memory: INT8 Adam, attention-prob checkpointing, lean inference scratch.
- Interactive launcher, CLI, tests, benchmarks, docs.

## Next

1. **Full block checkpointing** (`CHECK_LIST.md` item 30): single-layer-sized
   activation slots + per-layer forward recompute. Target 2–3× less RSS.
2. **QLoRA** (item 32): batched INT8 training forward/backward, quantized-base
   checkpoint format, LoRA-only grads. Low ceiling on the current 8.5M model.
3. **Speculative decoding that wins** (item 22): add a real draft model or
   train n-gram tables; prompt lookup accepted 0/37 on the smoke model.
4. **Display sanitizer** for invalid UTF-8 byte output; full NFC/NFKC.
5. **Token-level PPO with a value head**; better RL reward shaping.
6. **Scale-up run** (items 37–39/45): 20–50M total params, 100 MB–1.9 GB
   corpus, more steps, optional distillation. The epoch formula already scales
   the budget from `tokens_per_param` and active parameters.
7. **Kernel tuning**: pack A panels, 6×8 microtiles, measure packing crossover.
8. **Smarter SFT packing** (document-boundary masks, no padding waste).
