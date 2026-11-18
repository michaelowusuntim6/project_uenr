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
- Memory: INT8 Adam, attention-prob checkpointing, lean inference scratch,
  full block gradient checkpointing (1.54× less RSS at seq 512), FP16
  attention-probability storage (1.62× less with block), and a no-base-state
  QLoRA path (1.16× less LoRA/QLoRA RSS).
- Crash-safe UMDL v2 checkpoints (Adafactor/AdamW/INT8-Adam/Muon state),
  newest-3 rotation, SIGINT save, `--resume`, scale/draft profile.
- GitHub Actions cloud training workflow (ubuntu-latest, 4 vCPU/16 GB,
  manual dispatch, per-run checkpoint releases).
- Interactive launcher, CLI, tests, benchmarks, docs.

## Next

1. **Packed INT4 base for QLoRA** (item 32): batched quantized training
   forward/backward and a quantized-base checkpoint format; ~27 MB further
   saving over the current no-base-state path.
2. **Scale-profile draft retraining** so draft speculation wins at 64.8M total.
3. **Speculative decoding that wins** (item 22): add a real draft model or
   train n-gram tables; prompt lookup accepted 0/37 on the smoke model.
4. **Display sanitizer** for invalid UTF-8 byte output; full NFC/NFKC.
5. **Token-level PPO with a value head**; better RL reward shaping.
6. **Scale-up run** (items 37–39/45): 20–50M total params, 100 MB–1.9 GB
   corpus, more steps, optional distillation. The epoch formula already scales
   the budget from `tokens_per_param` and active parameters.
7. **Kernel tuning**: pack A panels, 6×8 microtiles, measure packing crossover.
8. **Smarter SFT packing** (document-boundary masks, no padding waste).
9. **4-thread CPU saturation on the cloud runner** (thread-pool extension;
   currently 2 of 4 vCPUs used).
10. **Dataset mirror** as a GitHub release asset so follow-up runs do not
    re-download 1.92 GB from HuggingFace.
11. **Cross-run checkpoint resume** from prior workflow artifacts.
