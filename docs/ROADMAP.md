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
- Four-thread training on the cloud runner (main + 3 workers, CPUs 0–3);
  the laptop profile remains two threads.
- Runtime AVX2+FMA dispatch (GEMM/GEMV, attention, RMSNorm, SwiGLU,
  softmax-CE, AdamW, RoPE, INT8/Q4 KV) and opt-in AVX-512 GEMM/attention;
  the Ivy Bridge path remains the laptop default.
- Cloud config tuned to batch 64 / LR 1.41e-3, step defaults sized to the
  ~26M-token budget; batch_size/lr are workflow inputs.
- Batch/LR sweep, AVX-512, and post-saturation measurements are recorded in
  `docs/ARCH_OPTIMIZATION.md`.
- Interactive launcher (25 actions + exit), CLI, tests, benchmarks, docs.

## Next

1. **Dispatch the first real `full-core` run** (1600 pretrain + 1000 SFT +
   1000 LoRA, ~2.5–3 h including prepare) and verify the three stable
   checkpoints improve on init.
2. **Dispatch `full-rl`** after full-core (600 DPO + 400 GRPO + 400 PPO,
   ~20 min plus prepare) and verify its checkpoints.
3. **Packed INT4 base for QLoRA** (item 32): batched quantized training
   forward/backward and a quantized-base checkpoint format.
4. **Scale-profile draft retraining** so draft speculation wins at 64.8M.
5. **Speculative decoding that wins**, display sanitizer/full NFC, and
   token-level PPO with a value head.
6. **Scale-up run** (20–50M total params / larger corpus): the epoch formula
   already scales the budget.
7. **Kernel follow-ups:** pack A panels, measure AVX-512 at 2 threads, and
   investigate the 1024-cube memory bound.
8. **Smarter SFT packing** (document-boundary masks, no padding waste).
