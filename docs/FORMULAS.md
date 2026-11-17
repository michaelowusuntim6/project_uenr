# UENR Formulas

Every formula in this file is implemented in C in the tree. None of them assumes
FMA, AVX2, a fast hardware divide, or a generic x86 microarchitecture. The target
is the i5-3210M: AVX1 (8-wide FP32), F16C, SSE4.2, **no FMA, no AVX2**, measured
sustained clock ~2.2 GHz.

Notation: `v` is an 8-lane FP32 vector, `*` is `vmulps`, `+` is `vaddps`. The
compiler is instructed to never contract these into FMA (there is none), and we
never use `-ffast-math`.

---

## 1. exp

Forward (`uenr_expf`, `uenr_v_exp`):

```
x = k·ln2 + r,   k = round(x / ln2),   |r| ≤ ln2/2
exp(x) = 2^k · P(r)
P(r) = 1 + r(1 + r(1/2 + r(1/6 + r(1/24 + r(1/120 + r/720)))))   [degree 6]
2^k  = bit pattern ((k+127) << 23)
```

Relative error over the reduced range is < 2.5e-7. `x` is clamped to
[-88, 88] to avoid exponent-field overflow. In the vector path `2^k` is built
with **two 128-bit integer halves** because `vpslld` on 256-bit registers is
AVX2-only; `vcvtps2dq`, `vpaddd`, `vpslld` are run on `__m128i` halves and
recombined with `vinsertf128`.

Cost: ~9 mul/add + 1 convert + ~6 integer ops per 8 lanes. No divide.

## 2. log

`uenr_logf` is used only once per token for cross-entropy, but it is also
dependency-free:

```
x = m · 2^e,  m ∈ [1,2)        (bit split: exponent field, mantissa field)
if m > sqrt(2): m /= 2, e += 1
s = (m-1)/(m+1)                 |s| ≤ 0.1716
log(m) = 2·(s + s³/3 + s⁵/5 + s⁷/7 + s⁹/9 + s¹¹/11)
log(x) = e·ln2 + log(m)
```

The atanh series converges fast on this narrow interval; degree 11 gives < 2e-7
absolute error. No libm.

## 3. sin / cos (RoPE tables)

Quadrant reduction to `|r| ≤ π/4`:

```
k = round(x · 2/π),  r = x - k·(π/2),  q = k mod 4
sin(r) = r - r³/6 + r⁵/120 - r⁷/5040 + r⁹/362880
cos(r) = 1 - r²/2 + r⁴/24 - r⁶/720 + r⁸/40320
```

Then the standard quadrant swap/sign. Error < 1e-7 on the reduced range.
RoPE tables are built once at model init: for position `p` and pair `j`,

```
inv_freq[j] = theta^(-2j/hd)
cos_tab[p][j] = cos(p · inv_freq[j])
sin_tab[p][j] = sin(p · inv_freq[j])
```

## 4. sigmoid, SiLU, tanh, GELU

```
sigmoid(x) = 1 / (1 + exp(-x))          [rcp + 1 Newton step]
silu(x)    = x · sigmoid(x)
tanh(x)    = 2·sigmoid(2x) - 1
gelu(x)    = 0.5x(1 + tanh(0.79788456(x + 0.044715x³)))
```

`rcp` is `vrcpps` followed by `y ← y(2 - x·y)`. Throughput ~4× `vdivps`,
accuracy to ~1 ulp for our use.

## 5. rsqrt

```
y = vrsqrtps(x)                  // ~11-bit estimate
y = y(1.5 - 0.5·x·y²)            // one Newton step => ~fp32 accuracy
sqrt(x) = x · rsqrt(x)           // used only when x > 0
```

In RMSNorm the input is `mean(x²)+eps > 0`, so `x·rsqrt(x)` is safe. In AdamW
we use `vsqrtps` directly because `v` can be exactly 0 (a parameter that has
received no gradient yet): `0 · inf = NaN`, which was a real bug caught by the
checkpoint NaN audit.

## 6. RMSNorm

Forward, per token row of width `D`:

```
r = 1/sqrt((1/D)Σ x_i² + eps)
y_i = x_i · r · w_i
```

Backward, given `dy`:

```
s   = Σ dy_i · w_i · x_i
c   = r³ / D · s
dx_i = r · w_i · dy_i - c · x_i
dw_i += r · dy_i · x_i        (accumulated over tokens)
```

Derivation: `dr/dx_j = -r³ x_j / D`, then product rule on `y_i = x_i r w_i`.
The saved `r` avoids recomputing the norm in backward. For QK-norm the same
function is called on `T·H` rows of width `head_dim`.

## 7. RoPE

For each head and pair `j`, with `(a, b) = (x[2j], x[2j+1])`:

```
forward:  a' = a·c - b·s
          b' = a·s + b·c
backward: ga' = ga·c + gb·s      (transpose rotation)
          gb' = -ga·s + gb·c
forward followed by backward is the identity (verified in `tests/test_ops.c`).
```

Positions reset per sequence: `p = t mod S`.

## 8. Causal GQA attention

For batch `b`, query head `h`, position `s`; KV head `g = h / (H/G)`:

```
score_j = (q[b,s,h] · k[b,j,g]) · scale,  scale = 1/sqrt(head_dim), j ≤ s
p_j     = softmax(score)_j                (max-subtracted)
out     = Σ_{j≤s} p_j · v[b,j,g]
```

Backward:

```
dprob_j = dout · v_j
dscore_j = p_j (dprob_j - Σ_k p_k dprob_k)
dq     += scale Σ_j dscore_j k_j
dk_j   += scale dscore_j q
dv_j   += p_j dout
```

KV heads accumulate gradients from every query head in their group. The
forward stores the attention probabilities; seq length is capped by
`model.max_seq`, and the buffer size is `L·B·H·S²`.

## 9. SwiGLU

```
gate = X Wg^T            [n,F]
up   = X Wu^T            [n,F]
h    = silu(gate) · up
Y    = h Wd^T            [n,C]
```

Backward:

```
dWd += dY^T h
dh   = dY Wd
dg   = dh · up · silu'(gate)
du   = dh · silu(gate)
silu'(g) = sigmoid(g) · (1 + g·(1 - sigmoid(g)))
dWg += dg^T X ;  dWu += du^T X
dX   = dg Wg + du Wu
```

Gate/up/h are recomputed from the saved normed input during backward, trading
compute for activation memory. SwiGLU backward scratch is `5·n·F` floats
(gate, up, h/dh, dg, du).

## 10. Fused softmax + cross-entropy

For logits `z`, target `y`:

```
m = max(z);  S = Σ exp(z_i - m)
loss = log(S) + m - z_y
dlogits_i = exp(z_i - m)/S - 1{i=y}
```

Ignored labels produce zero loss and zero `dlogits`. The loss never needs a
materialized probability buffer separate from `dlogits`; the same buffer is
reused for the softmax values and then the one-hot subtraction.

## 11. AdamW

With step counter `t`, `bc1 = 1 - b1^t`, `bc2 = 1 - b2^t`:

```
m = b1 m + (1-b1) g
v = b2 v + (1-b2) g²
c1 = lr·sqrt(bc2)/bc1
c2 = eps·sqrt(bc2)
p = p - lr·wd·p - c1·m / (sqrt(v) + c2)
```

`b1^t` and `b2^t` are evaluated with our own `exp(t·log(b))`. Weight decay is
decoupled and skipped for norm weights (tensor flag `no_decay`).

## 12. Schedule and epoch formula

**Step budget.** Instead of a fixed epoch count:

```
target_tokens = tokens_per_param · active_params
epochs        = ceil(target_tokens / corpus_tokens), clamped [min_epochs, max_epochs]
total_steps   = epochs · corpus_tokens / (batch_size · seq_len)
total_steps   = clamp(total_steps, min_steps_before_stop, max_steps)
```

For UENR-TinyMoE (`active ≈ 1.30M`, `tokens_per_param = 20`) the target is
~26M tokens. The shipped `prepare.max_bytes = 200 MiB` yields ~55.2M training
tokens, so the formula chooses 1 epoch (~26.9k steps) and then clamps to
`max_steps = 20000` (~41M tokens). A 16 MiB experiment slice has ~3.85M
training tokens, which instead selects the maximum 3 epochs.

**LR curve.**

```
warmup: lr_t = lr · t / warmup_steps
then:   p = (t - warmup)/(total - warmup)
        lr_t = min_lr + (lr - min_lr)·0.5·(1 + cos(π p))
adaptive (kind="adaptive"):
        lr_t *= clamp(1 + 0.2·(ema_loss/target_loss - 1), 0.75, 1.25)
```

The adaptive factor raises the LR slightly when the EMA loss is above target
and lowers it when below, within a safe band.

**Early stop.** EMA `e = βe + (1-β)loss` with `β = ema_beta`. Track the best
EMA; if it has not improved by more than `plateau_delta` for
`plateau_window` steps, stop. Stop immediately when `e ≤ target_loss` and
`step ≥ min_steps_before_stop`.

## 13. MoE router

For token `x`, router logits `r = x Wr^T`; `p = softmax(r)`; select the top
`K` experts. The layer output for the routed part is

```
y = Σ_{k∈topK} p_k · expert_k(x)
```

Router gradient, with `dp_k = ⟨expert_k(x), dy⟩`:

```
denom = Σ_{k∈topK} p_k dp_k
∂L/∂r_j = p_j · ( Σ_{k: idx_k=j} dp_k - denom )
```

Switch-style load-balance auxiliary loss (optional, coefficient `α`):

```
f_i = count_i / (T·K)          fraction of assignments to expert i
P_i = (1/T) Σ_t p_{t,i}        mean gate probability
aux = α · E · Σ_i f_i P_i
∂aux/∂r_{t,j} = α·E·(1/T)·p_{t,j}·( f_j - Σ_i f_i p_{t,i} )
```

The auxiliary gradient is added to the router gradient in backward; it is not
part of the reported CE loss. Top-1 routing is piecewise constant, so numerical
gradient checks in `tests/test_gradcheck.c` use dense and single-expert models
for the smooth cases, and the router formula is derived above.

## 14. DPO

For a preference pair `(x, y_w, y_l)`, sequence log-probs under policy `π` and
frozen reference `π_ref`:

```
Δ = (logπ(y_w|x) - logπ_ref(y_w|x)) - (logπ(y_l|x) - logπ_ref(y_l|x))
L = -log σ(βΔ)
∂L/∂logπ(y_w|x) = -β(1-σ(βΔ))
∂L/∂logπ(y_l|x) = +β(1-σ(βΔ))
```

Per-token `dlogits = softmax - onehot` is scaled by these coefficients and fed
to the ordinary transformer backward pass.

## 15. PPO and GRPO

Rollouts are sampled from the policy; reward is a configurable heuristic
(format bonus, length target, repetition penalty).

Let `r = exp(logπ_new(y) − logπ_old(y))` (sequence level) and `A` the
advantage. The loss is written with the sign convention
`dL/dlogπ = coefficient`, and the trainer scales the CE-style `dlogits`
buffer (`softmax − onehot = −dlogπ/dlogits`) by `−coefficient`.

**GRPO** normalizes advantages inside a group of `G` completions:

```
A_g = (r_g - mean(r)) / (std(r) + eps)
L   = -A·logπθ(y) + kl_coef·KL
coefficient = -A + kl_coef
```

**PPO** uses the clipped surrogate:

```
ratio = exp(logπ_new(y) - logπ_old(y))
L      = -min(ratio·A, clamp(ratio, 1-clip, 1+clip)·A) + kl_coef·KL
dS/dlogπ = A·ratio   if A>0 and ratio ≤ 1+clip
                     or A<0 and ratio ≥ 1-clip
         = 0         otherwise (the clipped term is the minimum)
coefficient = -dS/dlogπ + kl_coef
```

Both signs are handled explicitly: for `A>0` clipping binds above `1+clip`,
for `A<0` it binds below `1-clip`; inside the trust region (and in the
improving direction outside it) the update is active.

**KL** uses the frozen reference and the single-sample k1 estimator
`KL ≈ logπθ(y) − logπ_ref(y)`; the reference log-prob is constant, so its
gradient contribution is `+kl_coef` on the policy log-probability.

`rl.temperature` affects sampling only; log-probabilities and PPO ratios are
computed at T=1 against the T=1 behaviour policy. This is a deliberately small,
dependency-free policy-gradient implementation; a learned value head, entropy
bonus, GAE and token-level ratios are roadmap items.

## 16. Throughput accounting

```
training FLOPs/token ≈ 6 · (active_params + V·C)   (fwd + bwd, tied head)
MFU = achieved_FLOP/s / (33.9e9 · threads)
```

The measured 2-core roof is 67.8 GFLOP/s. The GEMM microkernel reaches
~15–25 GFLOP/s depending on shape and cache state; end-to-end training is
memory-bound and reports ~9 GFLOP/s at the default micro-batch on this laptop.
