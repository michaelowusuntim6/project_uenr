# Roadmap

## Done

- Measured hardware audit (`docs/AUDIT.md`): 2C/4T Ivy Bridge, AVX but no
  FMA/AVX2, 33.9 GFLOP/s/core, 13.4 GB/s DRAM at two threads.
- FMA-free scalar and vector formula set (exp, log, sin/cos, sigmoid, SiLU,
  tanh, GELU, rcp, rsqrt) with derivation doc.
- AVX1-safe 2^k construction and F16C conversion helpers.
- Blocked GEMM with 4×8 microkernel, packed transposed-B panels, GEMV path,
  2-thread pool pinned to physical cores.
- RMSNorm (incl. per-head QK-norm), RoPE, causal GQA attention, SwiGLU,
  fused softmax+CE, AdamW, gradient clipping.
- Tiny sparse MoE: 8.47M total / 1.30M active params, 24 experts top-1,
  shared expert, load-balance auxiliary loss, full backward.
- Numerical gradient check on dense and MoE models.
- Byte-level BPE trainer (generation-stamped hash counting), heap encoder,
  decoder, sanitizer, chat/tool-call template, SFT/preference data synthesis.
- Pretrain, SFT, LoRA (with merge), DPO, PPO, GRPO in C.
- Inference with greedy/top-k/top-p sampling and interactive chat.
- Atomic checkpoints, resume, CSV metrics.
- Interactive numbered launcher, CLI, tests, benchmarks, docs.

## Next (ordered by impact on this machine)

1. **KV cache decode — done.** A GQA-aware per-layer K/V cache now decodes
   incrementally (69.5 → 667.3 tok/s, 9.6×) and matches full recompute exactly.
   The long-context work (cross-layer sharing, sliding windows, INT8 KV) is
   items 49–57.
2. **fp16 storage + fp32 master weights.** F16C conversion is implemented but
   weights are fp32 today. Halving parameter and activation traffic targets
   the measured ×1.30 DRAM scaling wall. Needs careful master-copy/Adam
   handling.
3. **Tokenizer display sanitizer.** Byte-level output can be invalid UTF-8;
   replace invalid sequences for terminal display.
4. **Full NFC/NFKC normalization** (currently UTF-8 validation + line endings)
   and an optional regex-style pretokenizer for better merges on code.
5. **Token-level PPO with a value head**, replacing the sequence-level ratio
   and group-mean baseline.
6. **Kernel tuning**: pack A panels as well as B, try 6×8 microtiles, measure
   the crossover where packing pays. Current GEMM is ~15–25 GFLOP/s vs the
   67.8 two-core roof.
7. **Smarter data packing**: avoid padding in SFT batches; document-boundary
   masks inside a packed stream.
8. **More tests**: arena/RNG/IO unit tests, model shape tests, a short
   deterministic loss-decrease regression, and `opbench` bandwidth checks.
9. **Tool-call JSON schema samples** and a tougher alignment reward (schema
   validity, exact tool names) instead of the current length/format heuristic.
10. **Larger corpus run** once the above land; the epoch formula already
    scales the step budget from `tokens_per_param` and active parameters.
