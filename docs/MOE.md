# Sparse MoE

UENR-TinyMoE is a **top-1 sparse mixture of experts** at a scale that a
two-core Ivy Bridge can train from scratch: 8.47M total parameters,
1.30M active parameters per token.

## Shapes (default config)

```
V = 2048, C = 192, L = 4, H = 6, G = 2, hd = 32
E = 24 experts, top-k = 1, expert width EF = 128
shared expert width FF = 256 (always active)
```

Per token, each layer activates one routed expert plus the shared expert:

```
attn projections   98,304
router              4,608
shared SwiGLU     147,456
routed SwiGLU      73,728
norms                  ~448
----------------------------
per layer         324,544  × 4 layers = 1,298,176
```

## Routing

For each token in the flattened micro-batch:

```
r = xf · Wr^T                [E]
p = softmax(r)
idx, gate = top-k(p)
```

Top-k selection is deterministic: repeated argmax excluding already-selected
experts, ties broken by lower expert id. The gate is the softmax probability,
not renormalized over the top-k subset.

## Dispatch

The forward pass groups tokens by selected expert:

1. Count assignments per expert (`ecount`), prefix-sum to offsets.
2. Fill a token list per expert.
3. For each expert: gather its tokens into a contiguous buffer, run SwiGLU with
   that expert's `[EF,C]` gate/up and `[C,EF]` down matrices, scatter
   `gate · expert(x)` back into the token's output, and save the ungated expert
   output for the router gradient.
4. Add the shared expert for all tokens.
5. If `moe.enabled` is false, the same SwiGLU code runs densely over all tokens
   with the shared weights — the dense baseline.

Grouping is recomputed from saved indices during backward; no per-expert
activation tensors are stored beyond the routed outputs needed for gate
gradients.

## Backward

For each expert, the gradient of its output is `gate · dy`; SwiGLU backward
produces `dWg`, `dWu`, `dWd`, and `dx` (scattered back to tokens). The gate
gradient is `dp = ⟨expert(x), dy⟩`. The router softmax gradient is

```
denom = Σ_{k∈topK} gate_k dp_k
∂L/∂r_j = p_j (Σ_{k:idx_k=j} dp_k - denom)
```

and the router weight gradient is `dWr += dlogits^T xf`.

## Load balancing

The optional Switch-style auxiliary loss keeps expert usage from collapsing:

```
f_i = count_i / (T·K)
P_i = mean_t p_{t,i}
aux = coef · E · Σ_i f_i P_i
```

Its gradient is added to the router logits during backward
(`docs/FORMULAS.md` §13). With `router_aux_coef = 5e-4` the effect is gentle at
this scale; set it to 0 to train pure top-1 routing.

## Why top-1 at this scale

- Active compute stays tiny (~1.3M parameters/token), so the 2-core CPU can
  afford real training steps.
- One expert per token means deterministic, cache-friendly grouping: each
  expert processes a small contiguous batch.
- Total capacity (24 experts) is large relative to the dense active path, so
  the model can scale by increasing `num_experts` with almost no extra
  per-token FLOPs — only memory and checkpoint size grow.

Top-k > 1 is implemented in forward/backward and validated by the config
checker (`experts_per_tok ≤ num_experts`), but the shipped model uses top-1
because that is the compute budget this CPU can sustain.
