# Adapting DeepSeek V3 / R1 / V4 / V4.1 to UENR

**Scope.** This is an engineering adaptation study for UENR, the 8.45M-total /
1.28M-active sparse-MoE CPU model in this repository. It analyses all four
papers in `/home/mike/PDF`:

- `DeepSeek-V3 Technical Report` (V3)
- `DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL` (R1)
- `DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence` (V4)
- `DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression` (V4.1)

It is **not** a claim that an 8M laptop model can reproduce 671B–1.6T results.
Every number below is labelled as:

- **M** = measured in this repository,
- **R** = reported in a paper,
- **P** = projection for UENR, derived from bytes/FLOPs and the paper's
  mechanism.

The projections are ranges, not promises. A projection that depends on quality
(perplexity, benchmark score) cannot be known before an ablation on the same
data and token budget.

---

## 1. Executive verdict

**Adapt V3 as the primary architectural reference, R1 only for the
post-training loop, and V4.1 selectively for long-context inference. Do not
port V4 wholesale.**

| Paper | What it is best at | UENR decision |
|---|---|---|
| **V3** | MLA, DeepSeekMoE with auxiliary-loss-free balancing, MTP, FP8, GRPO | **Primary source.** Adopt aux-free balancing and SwiGLU clamp; prototype MTP-lite and true Muon; skip FP8/DualPipe/EP. |
| **R1** | Pure-RL reasoning with GRPO, rule-based verifiable rewards, cold-start/rejection-sample pipeline | **Adopt the recipe, not the scale.** Replace heuristic rewards with task verifiers; keep the existing sequence-level GRPO math, add reference refresh and zero-variance filtering. |
| **V4** | 1M context with CSA/HCA, mHC, Muon, FP4 QAT, heterogeneous KV | **Take HCA-lite, KV layout/bounded replay, clamp; reject mHC and native FP4; test Muon only as an ablation.** |
| **V4.1** | CED (half prefill), CSA2 cross-layer reuse, hierarchical indexer, single-pass mHC, Engram, DSpark, FP4 KV, SWA bounded replay | **Take bounded replay/persistent KV and perhaps CED-lite; reject Engram-as-published and DSpark for this model size; treat CSA2 as future work above 64K context.** |

The largest *credible* CPU gains are **inference memory and long-context
management**, not training throughput:

- compressed global KV: **8–16×** less KV at long context in FP32 before
  further quantization (up to **32–64×** with INT8/Q4);
- long-context decode: **1.5–3×** when KV bytes dominate;
- agent-prefix cache hits with bounded replay: **2–4×** lower TTFT;
- CED-style prefill: **1.1–1.4×** on a 4-layer model (paper's 2× is for 40+
  layers);
- training: **no credible 5× speedup**. Aux-free balancing and SwiGLU clamp
  improve quality/stability, not throughput; MTP costs more per step; Muon is
  unproven on this CPU.

There is no honest route from these papers to a `x5.3` CPU training speedup.
The nearest `x` claims are KV size and long-context decode, where the
compression ratios are dimensional and can be computed.

---

## 2. UENR baseline used for every comparison

| Quantity | Current UENR | Source |
|---|---|---|
| Parameters | 8,451,456 total / 1,275,264 active | `docs/DETAILED.md`, config |
| Shape | 4 layers, C=192, H=6, 1 KV head (MQA), head_dim 32, 24 experts top-1 + shared | config |
| Context default | `max_seq = 512`; long-context tests up to 65,536 | config/docs |
| KV, FP32 | **256 B/token** at 1M (256 MB) | `docs/DETAILED.md` §6 |
| KV, INT8 / Q4 | 64 B/token / 40 B/token | `docs/DETAILED.md` §6 |
| KV with eviction | 388 rows ≈ 0.1 MB at any length | `docs/DETAILED.md` §6 |
| Low-rank KV experiment | rank 8 = **4× smaller**, decode **1.02×**, val ppl 82.61 vs 81.03 | `docs/CHECK_LIST02.md` item 4 |
| Pretrain throughput, cloud | **7,363 tok/s** (batch 64, AVX-512, 4 threads) | `docs/ARCH_OPTIMIZATION.md` |
| Pretrain throughput, laptop | ≈1.15–1.2k tok/s | README/v0.1.0 |
| GEMM 384-cube | 229 GFLOP/s AVX-512 4t; 195 AVX2 4t | `docs/ARCH_OPTIMIZATION.md` |
| Decode, laptop | ≈1.0–1.1k tok/s at 128 cached tokens | `docs/INFERENCE.md` |
| Cloud KV bench, 512 ctx | INT8 4,758 tok/s; Q4 4,200 tok/s | `mode=bench` artifact |
| RL | sequence-level GRPO, heuristic length/format/repetition reward, no value head, k1 KL; PPO retired | `docs/FORMULAS.md` §15 |
| MoE balance | Switch-style auxiliary loss, coef 5e-4, top-1 of 24 | `docs/MOE.md` |
| Tokenizer | byte-level BPE, 2,048 vocab | config |
| Optimizer | Adafactor default; AdamW/INT8-Adam; Muon-lite tested and off | `docs/DETAILED.md` |

**Scale gap.** UENR's 1.275M active parameters are about 1/29,000 of V3's
37B active and 1/38,000 of V4.1-Flash's 16B decode-active parameters. Fixed
overheads that are small on a GPU (Sinkhorn, indexers, extra residual streams,
large embedding tables) can dominate a 4-layer CPU model. This is the main
reason the verdict is selective rather than "port V4". The paper-reported
multipliers are not transferable across that gap; only the dimensional
arguments (bytes, FLOPs) and small model-level changes are.

Two pre-existing results matter for adaptation decisions:

1. **Low-rank KV already exists.** The Hadamard-truncation path gives 4×
   smaller KV at 1.02× decode and 1.02× val ppl, but is off by default because
   it needs low-rank QAT. A new MLA implementation must beat this, not the
   FP32 baseline.
2. **Muon-lite already failed.** Row-RMS-normalized momentum gave ppl 413 vs
   67.7 for Adafactor at the same speed. Only true Newton–Schulz is
   interesting, and it has not been benchmarked on this CPU.

---

## 3. Paper summaries and what actually transfers

### 3.1 DeepSeek-V3 (671B total / 37B active, 14.8T tokens)

V3's four architecture/training ideas:

- **Multi-head Latent Attention (MLA).** Cache the low-rank latent
  `c_t^KV = W_DKV h_t` plus a decoupled RoPE key; reconstruct K/V on the fly.
  The paper's cache is much smaller than MHA while preserving quality.
- **DeepSeekMoE with auxiliary-loss-free balancing.** Fine-grained routed
  experts plus shared experts; a per-expert bias `b_i` is added only when
  choosing top-k. The bias is decreased for overloaded experts and increased
  for underloaded experts. A tiny sequence-wise auxiliary loss remains as a
  safety net.
- **Multi-Token Prediction (MTP).** One or more sequential prediction modules
  predict the next tokens with a shared embedding/head; discarded at inference.
  At small scale it improved HumanEval 20.7→26.8, GSM8K 25.4→31.4 and MMLU
  50.0→53.3 (same data/tokens), at extra training cost.
- **Infrastructure**: FP8 mixed precision, DualPipe, expert parallelism,
  node-limited routing. These are GPU-cluster features, not CPU features.

What transfers to UENR: aux-free balancing (cheap, model-level), MTP-lite
(optional, cost-sensitive), MLA ideas (but see existing low-rank KV), and the
GRPO post-training recipe. What does not: FP8, DualPipe, EP, node routing.

### 3.2 DeepSeek-R1 (reasoning via RL)

R1-Zero starts from V3-Base and applies **GRPO without SFT**. For question
`q`, sample `G` outputs; reward each with rule-based **accuracy + format**
rewards; normalize advantages within the group:

```
A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)
J   = E[ 1/G Σ min(ρ_i A_i, clip(ρ_i,1-ε,1+ε) A_i) - β KL(πθ || πref) ]
ρ_i = πθ(o_i|q) / πold(o_i|q)
```

R1-Zero rose from **15.6% to 77.9% pass@1 on AIME 2024** (86.7% with
self-consistency) over 10,400 steps, with response length increasing from
roughly 2k to 20k tokens. The full R1 adds: cold-start long-CoT SFT, a first RL
stage with a language-consistency reward, rejection sampling + SFT on reasoning
and non-reasoning data, then a second RL stage with rule + preference rewards.
The paper explicitly avoids neural reward models for reasoning because they are
hackable, and refreshes the reference model every 400 steps.

What transfers: the **objective and reward design**. What does not: the scale,
the 32k–65k token rollouts, and the assumption that a 8M model has latent
reasoning to unlock. For UENR, R1 is a post-training recipe, not an
architecture.

### 3.3 DeepSeek-V4 (1.6T / 49B and 284B / 13B, 1M context)

V4's new mechanisms:

- **CSA (Compressed Sparse Attention)**: compress every `m` KV entries into one
  weighted entry (overlapping 2m window), then use a low-rank "lightning
  indexer" to top-k select compressed entries; always add an exact local
  sliding-window branch and attention sinks.
- **HCA (Heavily Compressed Attention)**: compress every `m' ≫ m` entries into
  one, with no sparse selector. Cheaper and simpler than CSA, less precise.
- **mHC (Manifold-Constrained Hyper-Connections)**: widen the residual stream
  by `n_hc`, learn A/B/C mappings, constrain B to a doubly stochastic matrix
  via Sinkhorn-Knopp (20 iterations), and use non-negative sigmoid A/C. The
  reported wall-time overhead is **6.7%** after fused GPU kernels.
- **Muon optimizer**: Nesterov momentum + hybrid Newton–Schulz (8 fast + 2
  stable iterations) for matrices; AdamW for norms/biases.
- **FP4 QAT** for expert weights and the indexer QK path; index scores BF16;
  top-k selector speedup 2× at 99.7% recall.
- **KV layout**: separate state cache (SWA + uncompressed tail) and
  CSA/HCA compressed cache, block sizes lcm(m,m'), on-disk prefix cache.

Efficiency: at 1M context, V4-Pro uses **27% of single-token FLOPs and 10% of
the KV cache of V3.2**; V4-Flash uses **10% / 7%**. The paper claims KV can be
about **2% of a BF16 GQA8 baseline** at 1M. Those percentages are against
hundreds-of-billions-parameter GPU models with 5120-wide hidden states and
512-wide KV latents, not against UENR's 32-wide MQA cache. Do not transfer the
percentage directly.

### 3.4 DeepSeek-V4.1-Flash (552B backbone, 8B prefill / 16B decode, 45T tokens)

V4.1 adds:

- **CED (Causal Encoder–Decoder)**: bottom half layers are the encoder; upper
  half global KV is projected from the encoder's final hidden state using
  layer-specific `W_l^KV, W_l^Z`. This nearly halves prefill compute in the
  paper (40 layers). SWA stays per-layer.
- **CSA2**: three layer modes — Full (compute KV + index), Reindex (reuse KV,
  recompute indices), Reuse (reuse KV and indices). Cross-layer KV/index reuse
  attacks the layer dimension; a hierarchical indexer builds a shared candidate
  pool so deeper indexers do not rescore the whole context.
- **Single-pass mHC**: shifts the input-mixing coefficients by one block so
  residual update, input mixing and coefficient prediction can be fused; halves
  activation memory traffic in the Mega-mHC kernel.
- **Engram**: 196B parameters of hashed 2/3/4-gram conditional memory, separate
  from compute; deterministic addressing and prefetch.
- **DSpark**: 3-layer drafter, 5 parallel draft positions, Markov head,
  confidence head, dynamic verification length.
- **FP4 main KV cache**: MXFP4 E2M1 with one E4M3 scale per 16 channels,
  quantized after RoPE; **890 B/token** global KV and roughly 1/4 of V4-Flash;
  persistent KV (global + SWA policy) is roughly **1/8** after removing SWA
  from durable storage and using bounded replay.
- **SWA Bounded Replay**: when SWA KV is missing on a prefix hit, replay only
  the last `n_win` tokens instead of `L * n_win`; accept approximate state.
- **Head-wise Muon** and Sinkhorn-balanced embedding updates.

Again: 890 B/token is for a 512-channel latent with 2 KV slots and multiple
compressed layers. UENR's whole FP32 KV is already 256 B/token. CED's 2× is a
40-layer result; on 4 layers the ceiling is much lower.

---

## 4. Mechanism-by-mechanism decision table

Legend: **Adopt** = implement next; **Prototype** = build behind config and
measure; **Defer** = only after the first two phases; **Reject** = poor fit.

| # | Mechanism | Paper evidence | UENR fit | Projected UENR effect | Decision |
|---|---|---|---|---|---|
| 1 | Aux-loss-free MoE bias | V3 Table 5: small MoE val 2.258→2.253; better BBH/GSM8K | High: router already exists | 0.2–0.3% lower val loss; <1% speed | **Adopt** |
| 2 | SwiGLU clamp | V4 §4.2.3: stabilizes outlier-heavy runs, no quality loss | High: 4-line change in `swiglu.c` | No speed; fewer loss spikes; may allow slightly higher LR | **Adopt** |
| 3 | R1 rule rewards + GRPO refinements | R1 AIME 15.6→77.9; format/accuracy rewards | High for verifiable tasks; current reward heuristic | 1.2–2× task success per rollout budget on narrow tasks; no speed | **Adopt** |
| 4 | SFT best-fit packing + document masks | V4.1 padding ≤1e-4 | High: SFT currently samples conversations | 1.2–1.5× SFT token throughput | **Adopt** |
| 5 | MLA-lite latent KV | V3 §2.1.1; GPU cache reduction | Medium: existing rank-8 KV is 4× smaller at 1.02× ppl | 2.7× KV (d_c=16+8 RoPE); 1.05–1.25× long decode; quality needs retrain | **Prototype** |
| 6 | HCA-lite block-compressed global KV | V4 §2.3.2; V4.1 CSA2 | Medium-high: replaces full global slot | 8–16× less global KV; 1.5–3× long decode; quality risk | **Prototype** |
| 7 | On-disk KV + SWA bounded replay | V4.1 §3.2; persistent KV 1/8 | High: prefix cache exists but is in-batch only | 2–4× lower TTFT on agent cache hits; 8× less persistent storage | **Prototype** |
| 8 | CED-lite (global KV from layer L/2) | V4.1 §2.2; ~2× prefill at 40 layers | Low-medium: only 4 layers and cross-layer sharing already | 1.1–1.4× prefill; quality risk | **Defer** |
| 9 | CSA2 cross-layer KV/index reuse + hierarchical indexer | V4.1 §2.3 | Low at 512 context; high at 1M | 2–4× long-attention compute; large implementation | **Defer** (only if >64K is a product requirement) |
| 10 | QAT for low-bit KV | V4.1 §2.4.4; existing UENR QAT bits | High: `kv_qat_bits` already exists | Better Q4 quality; no size change | **Adopt** (extend tests) |
| 11 | Native FP4 | V4 §5.2.1; V4.1 890 B/token | None on Ivy Bridge/AVX2 CPU | No speed; Q4 already 40 B/token; FP4 ~34 B for 64 values | **Reject** |
| 12 | mHC | V4 §2.2; 6.7% GPU wall overhead | Low: depth 4, C=192 | CPU overhead 10–20%; quality benefit unproven | **Reject** for now |
| 13 | Muon/Newton–Schulz | V4 §2.4; faster convergence | Low: existing Muon-lite failed 6.1× worse | Maybe 1.1–1.3× fewer steps; CPU NS overhead; wall 0.9–1.15× | **Defer** (true-NS ablation only) |
| 14 | MTP (one depth) | V3 Table 4: small MMLU +3.3, GSM8K +6.0, HumanEval +6.1 | Medium: adds one block to a 4-layer model | +10–25% train FLOPs; 1.1–1.25× data efficiency; discard at inference | **Prototype** |
| 15 | DSpark speculative decoding | V4.1 §2.4.3 | Low: current draft model 0.47–0.85× | Needs high acceptance; current 0/37 prompt lookup | **Reject** until acceptance >80% |
| 16 | Engram (full) | V4.1 §2.4.2; 196B parameters | None | 196B is 23,000× UENR total; mini version unproven | **Reject**; revisit mini research |
| 17 | Anticipatory routing | V4 §4.2.3; ~20% GPU overhead | Low | Stale routing, extra forward; bad for CPU | **Reject** |
| 18 | Grouped output projection | V4 §2.3.1; for large c·n_h | None: H·hd = C = 192 | No saving | **Reject** |
| 19 | EP/DualPipe/FP8/BF16 | V3/V4 infrastructure | None on 2C CPU | No transfer | **Reject** |
| 20 | Attention sink, QK-norm, partial RoPE | V3/V4 details | Already implemented | Validates current design | **No change** |

---

## 5. Detailed adaptations and recommended formulas

### 5.1 Auxiliary-loss-free MoE (highest near-term value)

**Use V3's formula, with two CPU/robustness improvements.**

Paper formula:

```
s_i   = sigmoid(u_t^T e_i)
g'_i  = s_i if s_i + b_i ∈ TopK({s_j + b_j}, K_r), else 0
g_i   = g'_i / Σ_j g'_j
b_i  <- b_i - γ  if expert i is overloaded
b_i  <- b_i + γ  if expert i is underloaded
```

UENR currently uses softmax routing plus a Switch auxiliary loss. Adaptation:

1. Keep softmax/top-1 routing for compatibility, but add a per-expert bias
   before the top-1 comparison **only**. The gate used for the expert output is
   still the unbiased softmax/sigmoid value.
2. Update after each optimizer step from the batch's expert counts:
   `b_i += γ * (target_i - load_i)`, `target_i = 1/E`; clamp `b_i` to
   `[-0.5, 0.5]` to avoid runaway on a 24-expert model.
3. Keep the sequence-wise auxiliary loss at a very small coefficient
   (V3 uses a small value; UENR's current 5e-4 is a reasonable starting point,
   lower it to 1e-4 if bias balancing works).
4. Do not backpropagate through `b_i`. It is a controller, not a parameter.

Why: it removes the pressure that makes every sequence use every expert
equally, which the paper shows improves specialization and benchmarks. At
UENR's scale the paper's exact gains (0.2–0.3% val loss) are small; the main
benefit is removing a conflicting auxiliary gradient and making load balance
observable. **Expected: 0.2–0.3% lower val loss; speed unchanged.**

### 5.2 SwiGLU clamp

V4 clamps the SwiGLU linear component to `[-10, 10]` and the gate upper bound
to 10. Adopt an equivalent in `uenr_swiglu_*`:

```
gate = min(gate, 10)
up   = clamp(up, -10, 10)
h    = silu(gate) * up
```

This is a numerical-stability change, not a speed change. It should reduce
rare large-activation spikes and may permit a slightly higher LR. Keep it
configurable (`training.swiglu_clamp = 10.0`, 0 disables) so the baseline
remains reproducible.

### 5.3 R1-style GRPO for UENR

Keep the existing GRPO objective (it already matches R1 eq. 1–3). Improve the
**reward and sampling**, not the algebra:

- **Verifiable task curriculum.** Add a small C-side verifier for tasks the
  model can actually do: arithmetic with a boxed final answer, JSON/tool-call
  grammar, unit conversion, and exact-format extraction. Reward =
  `1.0 * correctness + 0.2 * format`, no neural reward model for these tasks.
- **Group size** `G=8` (not 16) to control CPU rollout cost; use `G=16` only
  for a hard task where all groups currently have zero variance.
- **Zero-variance filtering.** If `std(r)=0`, skip the group (no learning
  signal). This is a cheap improvement over the paper's fixed sampling and
  avoids wasted rollouts. Optionally repeat the prompt once with a higher
  temperature.
- **Reference refresh.** Follow R1's every-400-step reference replacement, but
  refresh a *frozen copy* only when the policy has moved; on CPU, refreshing
  too often doubles checkpoint memory. Start with 400.
- **Clip.** Keep UENR's sequence-level `clip=0.2`; do **not** copy R1's `ε=10`
  (that is a token-level GRPO setting on a very different scale). Sequence-level
  ratios have a different variance.
- **KL.** Keep the k1 estimator and a small coefficient (`0.001`); monitor it.
- **No value head.** R1's point is that the group baseline replaces the critic;
  UENR already does this. Do not add PPO's value model.
- **Cold start.** Use 500–2,000 short, verified reasoning examples before RL;
  do not start from a random model. Rejection-sample correct completions and
  SFT them as R1 does, but keep the corpus small enough for CPU.

Projected: **1.2–2× task success per rollout budget** on the verifiable
curriculum, not a 5× AIME jump. The 5× R1 number is scale-dependent.

### 5.4 SFT packing (V4.1 data lesson)

V4.1 reports padding ≤1e-4 with best-fit packing. UENR's SFT loader currently
builds one sample at a time and pads to `seq_len`. Implement:

- sort/bucket conversations by length (best-fit decreasing);
- pack multiple documents into one sequence up to `seq_len`;
- apply a block-diagonal causal mask so document `i` cannot attend to document
  `j`;
- keep the assistant-only label mask;
- report the padding fraction.

Projected: **1.2–1.5× SFT token throughput** at equal optimizer steps, with no
math change. This is one of the safest wins.

### 5.5 MLA-lite (only if learned latent beats existing rank-8 KV)

UENR already has a 4× KV compression via Hadamard truncation. The V3 MLA
alternative is:

```
c_t   = W_DKV h_t                    # d_c = 16 (choose)
k_C   = W_UK c_t                     # 32
v     = W_UV c_t                     # 32
k_R   = RoPE(W_KR h_t)               # 8 RoPE dims
cache: c_t (16) + k_R (8) = 24 floats = 96 B/token
```

vs 256 B/token FP32 now. That is **2.67×** less, not 4× like the existing
rank-8 path, but it is learned and should preserve quality better. With FP16
RoPE dims + INT8 latent, cache can be **~32 B/token (8×)**.

Why it is only a prototype:

- decode must re-expand K/V from `c` for every attended entry, adding small
  GEMVs that hurt short-context CPU decode;
- 2.67× is less than the existing 4× low-rank path;
- a 4-layer model may not tolerate a 16-dimensional latent.

Recommendation: implement as an alternative to the **existing low-rank** path,
not as a new default. Success criterion: better cached ppl than rank-8 at the
same bytes and ≥1.1× long decode; otherwise leave low-rank as the compressed
mode. Projected **1.05–1.25× long decode, 1.0× short decode, 2.7× KV**.

### 5.6 HCA-lite: compressed global KV with exact local SWA

This is the best long-context idea for UENR. Keep the exact sliding-window
ring; replace only the global donor slot with block-compressed entries:

```
for each block of m=8 global tokens:
    z_j = w_z^T h_j + b_j
    s_j = softmax(z over the block)
    C_i = Σ_j s_j ⊙ C_j            # C_j = W_KV h_j, dim 32
    store C_i                      # one entry per 8 tokens
```

At m=8, global KV is **16 B/token FP32** (or 8 B FP16, 4 B INT8, 2 B Q4) vs
256 B/token. At 1M that is **16 MB / 8 MB / 4 MB / 2 MB** instead of 256 MB.
The local SWA ring stays exact. This is V4's HCA simplified for CPU: no
lightning indexer, no top-k sort, contiguous block reduction.

Quality risk: block pooling loses within-block detail. Mitigation:

- keep `m=4` for the first experiment (32 B/token FP32, 8× smaller);
- add a small per-block residual term (my improvement): store the block mean
  plus a rank-2 residual `R_i` of the same 32-dim space, so the cache is
  `mean + U V^T` with `r=2`; bytes 32+16=48 B/token at m=8, still 5.3×
  smaller, with much lower error;
- fine-tune the model with the compressed cache (QAT-style straight-through),
  not just truncate at inference.

Projected: **8–16× KV reduction**, **1.5–3× long decode**, **1.2–2× prefill
at ≥32K**. At 512 context, almost no benefit because the local ring dominates.

### 5.7 On-disk KV + SWA Bounded Replay

UENR's prefix cache is in-batch; V4.1 persists KV and replays only `n_win` SWA
tokens on a miss. Adapt:

- persist the compressed global KV blocks and the last `n_win` exact local KV
  per donor layer to a file (or mmap region) keyed by a prefix hash;
- on a prefix hit, load global KV, then replay only the last
  `n_win=128` prompt tokens through the layers to reconstruct SWA state;
- keep the replay *approximate* and document it; V4.1 explicitly accepts this
  and simulates it during post-training;
- evict SWA from persistent storage first, global compressed entries last.

Projected: **2–4× lower TTFT** for a repeated long system prompt/agent session;
**8× smaller persistent cache** if SWA is not stored. For single-turn prompts
with no hit, no change.

### 5.8 CED-lite

V4.1's CED computes upper-layer global KV from the encoder's final hidden
state. For UENR:

- layers 0–1: normal global/local KV;
- layers 2–3: global KV projected from layer-1 hidden:
  `C_l = H_1 W_l^KV`, `Z_l = H_1 W_l^Z`, while SWA remains per-layer;
- training remains a normal full forward; only inference prefill can skip
  upper-layer global-KV construction.

Benefit on 4 layers is much smaller than the paper's 40-layer 2×. Projected
**1.1–1.4× prefill**, no decode gain, quality risk from a shallower encoder.
Do this only after HCA-lite/bounded replay, and only if agentic prefill is a
measured product bottleneck.

### 5.9 KV precision / QAT

UENR already supports `training.kv_qat_bits = 4` and Q4/INT8 cache. V4.1's
lesson is **quantize after RoPE and QAT the cache**, not merely quantize at
inference. Extend the existing QAT:

- quantize the 8 RoPE dimensions in FP16 and the remaining latent in INT8/4-bit
  (hybrid precision, as V4 does with BF16/FP8);
- per-16-channel scales for 32-wide caches, not per-head single scale;
- straight-through gradients during pretrain/SFT;
- test max logit error and argmax agreement at every context length.

Native FP4 is rejected: Ivy Bridge/AVX2 has no FP4 multiply, and Q4 is already
40 B/token. V4.1's 890 B/token is not a target for a 32-wide cache. Expected:
**better Q4 quality at the same 40 B/token**, not a speedup.

### 5.10 MTP-lite (optional)

Add one prediction depth with a separate small block sharing the embedding and
head (V3 eq. 21–25, `λ=0.3`):

```
h'_1 = M_1 [RMSNorm(h_main); RMSNorm(Emb(t+1))]
h_1  = TRM_1(h'_1)
L    = L_main + 0.3 * CE(OutHead(h_1), t+2)
```

UENR cost: one extra 192-wide transformer block on a 4-layer model is a
**+10–25% training FLOP** increase. V3's small-model gains were MMLU +3.3,
GSM8K +6.0, HumanEval +6.1, so the effect is real but it is a quality/data
efficiency trade, not a speed win. Prototype only after the long-context work;
discard the module at inference. Do not use the MTP head as a drafter unless
acceptance is measured.

### 5.11 Muon: test true Newton–Schulz, do not ship the existing approximation

V4's Muon uses 10 hybrid Newton–Schulz iterations on 2-D matrices. UENR's
current Muon is a row-RMS momentum approximation that failed badly (ppl 413 vs
67.7). If tested, it must be **true NS**, with:

- 5 iterations for small matrices (192×192) and 10 only for large expert
  matrices;
- fallback to AdamW for norms, embeddings, head, biases;
- a per-step wall-clock budget; if NS takes >10% of a step, reject;
- matched-token and matched-wall comparisons, not just step counts.

Projected: **1.1–1.3× fewer steps** (paper) but **0.9–1.15× wall** on CPU; no
quality guarantee. **Defer.**

### 5.12 What to reject and why

- **mHC**: 6.7% GPU overhead even with fused kernels; on a 4-layer CPU model
  the residual mixing adds memory passes and Sinkhorn work without a depth
  problem to solve. Reject.
- **Engram**: 196B parameters; a scaled-down 1M×32 table is 128 MB FP32 and
  random-access heavy. Reject as published; a 262k-entry INT8 research variant
  is possible but unlikely to pay at 8M parameters.
- **DSpark/draft**: UENR's 2-layer draft achieved ~70% acceptance but
  0.47–0.85× decode because verification overhead dominates at this size.
  DSpark's 3-layer drafter is even heavier. Reject until a drafter is <1 layer
  or acceptance >80%.
- **CSA2/hierarchical indexer**: 2–4× long-attention compute at 64K–1M, but
  large implementation and indexer overhead; only after HCA/bounded replay
  prove insufficient.
- **FP8/FP4/BF16/EP/DualPipe/grouped output/multimodal**: no CPU transfer.

### 5.13 Things UENR already does right

- MQA + cross-layer KV sharing is an 8× KV reduction vs original GQA—already
  the spirit of MLA.
- Attention sinks and sliding-window eviction match V4's SWA/sink design.
- QK-norm, partial RoPE, V-norm, logit softcaps, and four-norm residuals are
  the same stability family V3/V4 use.
- Low-rank KV QAT and Q4/INT8 caches already cover the practical CPU version
  of V4.1's FP4 storage.
- Sequence-level GRPO already removes the critic as R1 does.

---

## 6. Proposed UENR-C configuration (my recommended combined design)

Do not implement all of this at once. The target design is:

```
Local layers:    exact SWA ring, W=128, FP16/FP32 KV
Global layers:   block-compressed HCA-lite, m=4..8, plus rank-2 residual
KV precision:    RoPE dims FP16; latent dims INT8/Q4 with QAT
Prefix cache:    compressed global blocks on disk; replay last 128 SWA tokens
Attention:       exact local + compressed global; optional top-k above 64K
MoE routing:     top-1 of 24 + shared; auxiliary-loss-free bias (γ=1e-3)
FFN:             SwiGLU clamp [-10,10]
Optimizer:       Adafactor default; true Muon only as an ablation
Post-training:   R1-style verifiable GRPO + cold-start SFT
```

Expected combined long-context profile (not multiplied blindly):

| Metric | Current | UENR-C projection | Basis |
|---|---:|---:|---|
| KV at 1M, FP32 | 256 MB | 16–32 MB | m=8 with residual; 8–16× |
| KV at 1M, INT8 | 64 MB | 4–8 MB | 8–16× |
| KV at 1M, Q4 | 40 MB | 2–4 MB | 10–20× |
| Long decode at 64K | 1.0× | 1.5–3× | KV bytes dominate |
| Prefill at 64K | 1.0× | 1.2–2× | HCA + maybe CED |
| Agent cache-hit TTFT | 1.0× | 2–4× lower | bounded replay |
| Training throughput | 1.0× | 1.0–1.1× | only packing/clamp; MTP costs |
| Val loss same tokens | baseline | −0.2 to −0.5% | aux-free + clamp + packing |

The `x5.3`-style number does **not** exist for training here. The largest
defensible CPU multiplier is the **KV-size reduction** (8–16× FP32, up to
20× Q4) and the **long-decode/TTFT** range above.

---

## 7. Priority roadmap and validation

### Phase 0 — cheap, high confidence (1–2 weeks)

1. Auxiliary-loss-free bias; keep tiny sequence aux loss.
2. SwiGLU clamp (config-gated).
3. SFT best-fit packing + document mask.
4. R1-style verifier reward for the existing GRPO; zero-variance filtering;
   reference refresh.
5. Tests: `test_align` finite differences, `test_ops` clamp, SFT leakage test,
   packed-attention equivalence (packed vs unpacked logits).

### Phase 1 — long-context KV (3–5 weeks)

6. HCA-lite block compressor behind `inference.kv_compress = 0|4|8` with
   exact SWA ring; train/fine-tune with compressed-cache straight-through.
7. QAT for the compressed cache; hybrid FP16 RoPE / INT8 latent.
8. On-disk compressed global KV + SWA bounded replay.
9. `kvbench` cases at 512/2K/8K/32K/64K, cached ppl vs FP32, logit error,
   argmax agreement, bytes/token, TTFT with/without replay.

### Phase 2 — optional architecture experiments (4–8 weeks)

10. MLA-lite as an alternative to rank-8 KV; must beat rank-8 on quality at
    equal bytes.
11. CED-lite inference only; measure prefill at 8K–64K.
12. MTP-lite one depth; measure val loss at matched tokens and wall time.
13. True Muon/NS ablation with a hard 10% step-time budget.

### Phase 3 — research only

14. CSA2/hierarchical indexer above 64K.
15. Engram-mini (262k entries, INT8, 2/3-grams); protocol first.
16. DSpark only if a <1-layer drafter reaches >80% acceptance on CPU.

### Implementation map (where each change would live)

| Area | Current files | Adaptation |
|---|---|---|
| MoE router/bias | `src/model/model.c`, `config/uenr.toml` | aux-free bias, sequence loss |
| SwiGLU | `src/ops/swiglu.c` | clamp option |
| Attention/KV | `src/ops/attn.c`, `src/model/decode.c` | HCA-lite, bounded replay |
| KV layout/QAT | `include/uenr/model.h`, `src/model/decode.c`, config | compressed blocks, hybrid precision |
| SFT data | `src/data/dataset.c`, `src/train/train.c` | best-fit packing, document mask |
| RL | `src/train/align.c`, `src/train/reward.c`, config | verifier rewards, zero-variance filter, reference refresh |
| Optimizer | `src/ops/optim.c`, `src/train/train.c` | true-NS Muon ablation only |
| Benchmarks/tests | `bench/kvbench.c`, `tests/test_align.c`, `tests/test_ops.c` | KV bytes/ppl/TTFT and gradient checks |

### Acceptance rules

- Any compression must be evaluated on **cached perplexity at the same
  context**, not just KV bytes.
- Any FMA/SIMD kernel must keep the AVX1 path bit-clean and pass
  `test_simd_dispatch`.
- No new kernel without a `mode=bench` A/B (AVX1/AVX2/AVX-512) and a smoke
  dispatch.
- No pretraining claims from step counts alone; report tokens, wall time, and
  val loss.
- Revert a kernel/mechanism if it does not beat the existing baseline
  (especially rank-8 KV and Adafactor) within measurement noise.

---

## 8. Direct answers

**Which paper should UENR adapt?**
V3 first. Its MoE balancing and stability changes are small, testable, and
model-level. R1 supplies the post-training recipe. V4.1 supplies the only
long-context ideas worth implementing on CPU (HCA-style compression, bounded
replay/persistent KV, QAT). V4 is a reference for why those ideas work and for
what not to port (mHC, FP4, GPU infrastructure).

**What does each help?**
- V3: better expert specialization, denser training signal (MTP), MLA KV
  compression, GRPO.
- R1: verifiable-reward RL, cold start, rejection sampling, reference refresh.
- V4: compressed sparse attention, heterogeneous KV layout, mHC stability,
  Muon convergence, FP4 QAT.
- V4.1: half prefill (CED), cross-layer KV/index reuse (CSA2), hierarchical
  indexing, single-pass mHC, bounded replay, FP4 KV, Engram/DSpark.

**Same formulas or improved versions?**
Use V3's aux-free bias and V4.1's SWA replay formulas almost verbatim. Use the
R1 GRPO objective but with UENR's sequence-level clip, zero-variance filtering,
and verifier rewards. Replace HCA/CSA's GPU indexer with a block-max / threshold
selector; add a rank-2 residual to the block compression. Do not use mHC, full
Engram, DSpark, or native FP4 on this CPU. For Muon, only true Newton–Schulz is
worth an ablation; the current approximation is known to fail.

**What is the realistic x?**
KV size: **8–16×** at m=8 (up to **~20×** with Q4). Long-context decode:
**1.5–3×**. Agent cache-hit TTFT: **2–4× lower**. Prefill: **1.2–2×** at long
context (CED-lite 1.1–1.4× on four layers). Training throughput: **1.0–1.1×**;
quality: **0.2–0.5%** val-loss improvement from aux-free/clamp/packing. There
is no credible **x5.3 training** multiplier in these papers for an 8M CPU
model.

---

## 9. Evidence ledger (paper claims, not UENR measurements)

| Claim | Paper | Section |
|---|---|---|
| 671B total / 37B active, 14.8T tokens, 2.788M H800 hours | V3 | abstract |
| MLA caches latent + RoPE key only | V3 | §2.1.1 |
| Aux-loss-free bias; small model val 2.258→2.253, 3B 2.085→2.080 | V3 | §2.1.2, §4.5.2 |
| MTP small model: MMLU 50.0→53.3, GSM8K 25.4→31.4, HumanEval 20.7→26.8 | V3 | Table 4 |
| R1-Zero AIME pass@1 15.6→77.9, 86.7 with self-consistency | R1 | §2.3, Fig. 1 |
| GRPO group-normalized advantage and KL | R1 | §2.1 |
| Rule-based accuracy+format rewards; no neural RM for reasoning | R1 | §2.2 |
| V4-Pro 27% FLOPs / 10% KV; Flash 10% / 7% vs V3.2 at 1M | V4 | abstract, §1 |
| V4 KV ≈2% of BF16 GQA8 at 1M | V4 | §2.3.4 |
| mHC Sinkhorn 20 iterations; 6.7% wall overhead | V4 | §2.2, §3.4.2 |
| FP4 QAT top-k 2× speed, 99.7% recall | V4 | §5.2.1 |
| V4.1 CED halves prefill O(NL)→O(NL/2) | V4.1 | §2.2 |
| V4.1 global KV 890 B/token, 1/4 V4-Flash; persistent 1/8 | V4.1 | abstract, §3.2.1 |
| V4.1 single-pass mHC halves activation traffic | V4.1 | §2.4.1 |
| V4.1 Engram 196B params; DSpark 5 draft positions | V4.1 | §2.4.2–3 |
| V4.1 padding ≤1e-4 with best-fit packing | V4.1 | §4.1 |

**Bottom line:** adapt the small, model-level ideas first (aux-free routing,
clamp, packing, verifier GRPO). Then build a CPU-specific HCA-lite + bounded
replay KV path, because that is where the real `x` numbers live. Leave the
GPU-scale machinery in the papers.


## Measured outcomes

- SwiGLU clamp is applied in scalar, AVX2 and quantized paths.
- INT8 is the default KV format (64 B/token at 1M, ~1.04x FP32 decode).
- SFT document-boundary packing measured 733 vs 643 tok/s (1.14x) at seq 128.
- True Newton-Schulz Muon is implemented but 75x the Adafactor step cost on
  this CPU (105.3 s vs 1.41 s for two steps), so it is opt-in only.
- HCA/CED/replay/MTP remain deferred or already covered by cross-layer KV
  sharing; MLA-lite would change checkpoint layout and is not pursued.
