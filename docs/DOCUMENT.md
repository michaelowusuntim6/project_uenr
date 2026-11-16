# Architectural Specification: >90% KV Cache Reduction via MQA + Cross-Layer KV Sharing (Non-DeepSeek / Gemma 4 Style) for Pure C LLM Engines

## 1. Executive Summary & Architectural Goal

This specification details how to implement the **90%+ KV Cache Memory Reduction** strategy used in modern edge-efficient Transformers (specifically **Google Gemma 4 E2B / E4B**, **Gemma 3n**, and **Cross-Layer Attention [CLA]**) inside a **pure C (`C99`/`C11`) LLM engine from scratch**—without relying on DeepSeek's Multi-Head Latent Attention (MLA).

Instead of compressing Keys and Values through low-rank latent matrix factorization (MLA), this architecture achieves a **14x (~93%) reduction in KV cache memory** and eliminates over **50% of K/V projection weights and FLOPs** in upper layers by combining four synergistic mechanisms:

1. **Multi-Query Attention ($n_{kv} = 1$) with Enlarged Head Dimensions ($d_{head}$):** Collapses all Query heads onto a single shared Key/Value head while compensating for capacity loss by doubling/quadrupling $d_{head}$ ($256$ for local layers, $512$ for global layers).
2. **Cross-Layer KV Sharing (`num_kv_shared_layers`):** Trailing decoder layers ($L_{shared}$) completely omit $W_K$, $W_V$, and KV cache writes, directly borrowing the already-computed, post-norm, post-RoPE $K$ and $V$ tensors from the last non-shared layer of the matching attention type (`sliding_attention` vs. `full_attention`).
3. **Hybrid Local Sliding-Window + Global Attention (4:1 or 5:1 Ratio):** Restricts 80% of layers to a fixed sliding window ($W = 512$ or $1024$ tokens) using a circular ring-buffer KV cache, reserving full-sequence global KV storage for only 1 in every 5 layers.
4. **Proportional Partial RoPE (`p-RoPE`) + QKV-Normalization + Softcapping:** Stabilizes single-head cross-layer training and inference against attention logit explosion, rank collapse, and out-of-distribution long-context extrapolation.

---

## 2. Exact Memory & Compute Mathematics

### 2.1 KV Cache Footprint Formula

For a standard Transformer with $L$ layers, $n_{kv}$ KV heads, head dimension $d_{head}$, sequence length $S$, and element byte-width $B_{elem}$ (e.g., `2` bytes for `FP16`/`BF16`, `4` bytes for `FP32`):

$$\text{Memory}_{\text{standard}} = 2 \times L \times n_{kv} \times d_{head} \times S \times B_{elem}$$

When applying **Cross-Layer KV Sharing + Hybrid Local/Global Attention**, layers are partitioned into:
* **$L_{\text{active}}$**: The first $L - L_{\text{shared}}$ layers that own $W_K, W_V$ weights and allocate KV cache buffers.
  * $L_{\text{active, local}}$: Active sliding-window layers (capped at $\min(S, W)$ tokens).
  * $L_{\text{active, global}}$: Active full-attention layers (scales linearly with $S$).
* **$L_{\text{shared}}$**: The final `num_kv_shared_layers` layers that have **zero** KV cache allocation and **zero** $W_K, W_V$ parameters.

The exact KV cache memory in bytes at sequence length $S$ is:

$$\text{Memory}_{\text{optimized}}(S) = 2 \times B_{elem} \times n_{kv} \times \left[ L_{\text{active, local}} \cdot d_{\text{head, local}} \cdot \min(S, W) + L_{\text{active, global}} \cdot d_{\text{head, global}} \cdot S \right]$$

### 2.2 Concrete Benchmark Comparison (Gemma 4 E2B vs. Llama-3 8B Baseline)

| Metric | Standard GQA Baseline (Llama-3 8B Style) | MQA Only ($n_{kv}=1$) | MQA + Cross-Layer Sharing (Gemma 4 E2B Spec) |
| :--- | :--- | :--- | :--- |
| **Total Layers ($L$)** | 32 | 32 | 35 |
| **Active KV Layers ($L_{\text{active}}$)** | 32 (all global) | 32 (all global) | **15** (12 local + 3 global) |
| **Shared KV Layers ($L_{\text{shared}}$)** | 0 | 0 | **20** (16 local + 4 global) |
| **KV Heads ($n_{kv}$)** | 8 | 1 | **1** |
| **Head Dimension ($d_{head}$)** | 128 | 128 | **256** (local) / **512** (global) |
| **Bytes / Token (Long Context $S \gg W$, BF16)** | $2 \times 32 \times 8 \times 128 \times 2 = \mathbf{131,072\text{ B}}$ (128 KB) | $2 \times 32 \times 1 \times 128 \times 2 = \mathbf{16,384\text{ B}}$ (16 KB) | $2 \times 2 \times 1 \times (3 \times 512) = \mathbf{6,144\text{ B}}$ (6 KB) + fixed 12.6 MB local ring buffer |
| **KV Cache at $S = 32,768$ Tokens (BF16)** | **4,294.97 MB** (4.19 GB) | **536.87 MB** | **213.91 MB** (**95.02% reduction**) |

*(Note: Even when comparing strictly per-layer global+local active slots before sliding window saturation, $2 \times 2 \times 1 \times (12 \times 256 + 3 \times 512) = 18,432\text{ B/token}$ vs. $131,072\text{ B/token}$ is an immediate **85.9% reduction** at $S=1$, crossing **93%–95% reduction** once $S > 2,048$).*

---

## 3. Complete Mathematical Specification of the Forward Pass

To prevent Claude Code from introducing subtle numerical bugs, implement the exact mathematical operations below in your C forward pass.

### 3.1 Layer Routing & Donor Mapping Table

Let $L$ be `num_hidden_layers` and $L_{\text{shared}}$ be `num_kv_shared_layers`.
The boundary index is:

$$L_{\text{boundary}} = L - L_{\text{shared}}$$

* **Active (Owner) Layers ($l \in [0, L_{\text{boundary}} - 1]$):** Compute $Q_l, K_l, V_l$, normalize them, apply RoPE to $Q_l$ and $K_l$, write $(K_l, V_l)$ into cache slot $c = l$, and compute attention.
* **Shared (Borrower) Layers ($l \in [L_{\text{boundary}}, L - 1]$):** Have **no** $W_K^{(l)}, W_V^{(l)}, \gamma_K^{(l)}$ weights. They compute only $Q_l = x_l W_Q^{(l)}$, normalize $Q_l$, apply RoPE to $Q_l$, and read $(K_{\text{donor}(l)}, V_{\text{donor}(l)})$ directly from the cache of the **last active layer with the same attention type**:

$$\text{donor}(l) = \max \left\{ j \in [0, L_{\text{boundary}} - 1] \;\middle|\; \text{type}(j) = \text{type}(l) \right\}$$

**Example (Gemma 4 E2B, $L=35, L_{\text{shared}}=20 \implies L_{\text{boundary}}=15$):**
* Pattern: 4 `sliding_attention` layers followed by 1 `full_attention` layer (`[S, S, S, S, F]` repeated 7 times).
* Layers `0..14` are active owners. Layer `13` is the last active `sliding_attention` layer; Layer `14` is the last active `full_attention` layer.
* For layers `15..34`:
  * Every `sliding_attention` layer reads KV cache slot `13`.
  * Every `full_attention` layer reads KV cache slot `14`.

### 3.2 RMSNorm (Standard & Scale-Free Variants)

Gemma 4 uses two variants of RMSNorm with $\epsilon = 10^{-6}$:

1. **Standard Weighted RMSNorm** (used for Pre/Post-Attention norms, Pre/Post-FFN norms, $Q$-norm, and $K$-norm):

$$\text{RMSNorm}(x; \gamma) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$

2. **Scale-Free RMSNorm (`RMSNormNoScale`, used for $V$-norm):**
Normalizes the Value vectors to unit RMS magnitude without a learnable $\gamma$ parameter, preventing Value variance drift when $V$ is reused across 16+ layers:

$$\text{RMSNormNoScale}(v) = \frac{v}{\sqrt{\frac{1}{d_{head}} \sum_{i=1}^{d_{head}} v_i^2 + \epsilon}}$$

> **Critical Implementation Order:**
> 1. Project $Q = x W_Q$, $K = x W_K$, $V = x W_V$.
> 2. Reshape into heads: $Q \in \mathbb{R}^{n_q \times d_{head}}$, $K \in \mathbb{R}^{1 \times d_{head}}$, $V \in \mathbb{R}^{1 \times d_{head}}$.
> 3. Apply per-head norms **BEFORE RoPE**:
>    * $Q \leftarrow \text{RMSNorm}(Q; \gamma_Q)$
>    * $K \leftarrow \text{RMSNorm}(K; \gamma_K)$
>    * $V \leftarrow \text{RMSNormNoScale}(V)$
> 4. Apply RoPE to $Q$ and $K$ (do **not** apply RoPE to $V$).
> 5. Store post-norm, post-RoPE $K$ and post-norm $V$ into the KV cache.
>
> *Why this matters:* Because $K$ is cached *after* $K$-Norm and RoPE, borrower layers $l \ge L_{\text{boundary}}$ do **not** re-apply $K$-Norm or RoPE to the borrowed $K$ tensor. They only apply $Q$-Norm and RoPE (using position $t$) to their own newly computed $Q_l$.

### 3.3 Proportional Partial RoPE (`p-RoPE`)

In `full_attention` layers ($d_{head} = 512$), rotating all dimensions causes high-frequency angle aliasing over 128k contexts and restricts semantic matching across shared layers. Gemma 4 uses **Proportional RoPE** with `partial_rotary_factor` $p = 0.25$ on global layers ($	heta_{\text{global}} = 1,000,000$) and $p = 1.0$ on sliding layers ($	heta_{\text{local}} = 10,000$).

Let $d = d_{head}$ and $r = \lfloor p \cdot d / 2 \rfloor$ be the number of rotated frequency pairs:
* For `sliding_attention` ($d = 256, p = 1.0$): $r = 128$ pairs (all 256 dims rotated).
* For `full_attention` ($d = 512, p = 0.25$): $r = 64$ pairs (first 128 dims rotated; remaining 384 dims have $\omega_k = 0 \implies \cos(0)=1, \sin(0)=0$, acting as NoPE / unrotated semantic channels).

**Exact Inverse Frequency Formula (Note: denominator is full $d_{head}$, NOT $2r$):**

$$\omega_k = \begin{cases} \theta^{-\frac{2k}{d_{head}}} & \text{for } 0 \le k < r \\ 0 & \text{for } r \le k < \frac{d_{head}}{2} \end{cases}$$

Using NeoX split-half rotation on a head vector $u \in \mathbb{R}^{d_{head}}$ at sequence position $t$:

$$\begin{aligned}
u'_k &= u_k \cos(t \cdot \omega_k) - u_{k + d/2} \sin(t \cdot \omega_k) \\
u'_{k + d/2} &= u_{k + d/2} \cos(t \cdot \omega_k) + u_k \sin(t \cdot \omega_k)
\end{aligned} \quad \text{for } 0 \le k < r$$

For $r \le k < d/2$, $u'_k = u_k$ and $u'_{k + d/2} = u_{k + d/2}$ unchanged.

### 3.4 Attention Score Computation (Unit Scale + Optional Logit Softcapping)

Because both $Q$ and $K$ are RMS-normalized (`QKNorm`), their dot product variance is already bounded around $O(1)$. Therefore, Gemma 4 sets the attention scaling factor to **$s = 1.0$** (replacing the traditional $1/\sqrt{d_{head}}$ scale, or folding it into $\gamma_Q, \gamma_K$).

For query head $h \in [0, n_q - 1]$ at current position $t$, attending over valid cached positions $\tau \in \mathcal{W}_l(t)$:

$$z_{h, \tau} = s \cdot \left( Q_{l, h, t} \cdot K_{\text{donor}(l), 0, \tau}^T \right)$$

Where the causal window $\mathcal{W}_l(t)$ is:
* **Full Attention:** $\mathcal{W}_l(t) = [0, t]$
* **Sliding Window Attention ($W = 512$):** $\mathcal{W}_l(t) = [\max(0, t - W + 1), t]$

If `attn_logit_softcapping` ($C_{\text{attn}}$, e.g. $50.0$) is enabled in your model variant:

$$z'_{h, \tau} = C_{\text{attn}} \cdot \tanh\left(\frac{z_{h, \tau}}{C_{\text{attn}}}\right)$$

Then apply causal softmax and weighted Value aggregation using the single shared Value head ($kv\_head = 0$):

$$\alpha_{h, \tau} = \frac{\exp(z'_{h, \tau} - \max_j z'_{h, j})}{\sum_{m \in \mathcal{W}_l(t)} \exp(z'_{h, m} - \max_j z'_{h, j})}, \qquad O_{l, h, t} = \sum_{\tau \in \mathcal{W}_l(t)} \alpha_{h, \tau} V_{\text{donor}(l), 0, \tau}$$

### 3.5 Gemma 4 Four-Norm Residual Block Structure

To prevent gradient and activation magnitude explosion when 20 consecutive layers read from the same frozen KV representations, each decoder block wraps both sub-layers with **both Pre-Norm and Post-Norm**:

$$\begin{aligned}
h_{\text{in}} &= \text{RMSNorm}(x_l; \gamma_{\text{pre\_attn}}) \\
a_l &= \text{SelfAttn}(h_{\text{in}}) \\
x'_l &= x_l + \text{RMSNorm}(a_l; \gamma_{\text{post\_attn}}) \\
m_{\text{in}} &= \text{RMSNorm}(x'_l; \gamma_{\text{pre\_ffn}}) \\
f_l &= \text{GeGLU\_MLP}(m_{\text{in}}) \\
x_{l+1} &= x'_l + \text{RMSNorm}(f_l; \gamma_{\text{post\_ffn}})
\end{aligned}$$

And at the final LM head, apply **Final Logit Softcapping** ($C_{\text{final}} = 30.0$):

$$\text{logits} = 30.0 \cdot \tanh\left(\frac{\text{RMSNorm}(x_L; \gamma_{\text{final}}) W_{\text{embed}}^T}{30.0}\right)$$

---

## 4. Why This Achieves Stability (And How Not to Break It)

If you train or run a pure C model with $n_{kv}=1$ and cross-layer sharing without the safeguards below, it will suffer from **attention entropy collapse** (one token dominates softmax across 20 layers) or **NaN gradients**. Each mechanism solves a specific failure mode:

1. **Why QK-Norm + V-Norm are Mandatory:**
   When 8 Query heads across 20 layers (160 effective query heads!) attend to a *single* Key vector, unconstrained dot products grow huge in magnitude, saturating softmax into a one-hot spike. `QKNorm` projects $Q$ and $K$ onto hyperspheres of radius $\sqrt{d_{head}}$, capping maximum dot products regardless of how many layers share the key. `V-Norm` prevents downstream residual streams from accumulating runaway variance from repeated additions of the same $V$ tensor.
2. **Why `d_head` Must Expand ($128 \to 256 / 512$):**
   Reducing $n_{kv}$ from 8 to 1 cuts the bottleneck rank of $W_K$ and $W_V$ by 8x. Doubling $d_{head}$ to $256$ on local layers and quadrupling it to $512$ on global layers restores the subspace dimensionality of the single KV head while still saving 50%–75% of parameters compared to 8-head MHA.
3. **Why Same-Type Donor Matching (`local -> local`, `global -> global`) is Critical:**
   A sliding-window layer's $K$ tensor is trained with $\theta = 10,000$ and short-range positional decay. If a global layer ($\theta = 1,000,000$, 128k context) tries to read a sliding layer's KV cache, positional phase mismatch destroys long-range retrieval. Always route `sliding -> last active sliding` and `full -> last active full`.
4. **Training Stability (If Training From Scratch in C):**
   * **Knowledge Distillation Loss:** If training a small 1-KV-head model from scratch, add a KL-divergence loss against a larger multi-head teacher's logits: $\mathcal{L} = (1 - \alpha)\mathcal{L}_{\text{CE}} + \alpha T^2 D_{\text{KL}}(p_{\text{teacher}} \parallel p_{\text{student}})$.
   * **Gradient Accumulation on Donor Layers:** During backpropagation in C, a donor layer $d = \text{donor}(l)$ receives gradients not only from its own attention block, but also the sum of $\frac{\partial \mathcal{L}}{\partial K_d}$ and $\frac{\partial \mathcal{L}}{\partial V_d}$ accumulated from all borrower layers $l \in \text{Borrowers}(d)$ before backpropagating through $W_K^{(d)}$ and $W_V^{(d)}$.

---

## 5. Pure C (`C99`/`C11`) Implementation Blueprint for Claude Code

Give the following data structures and execution blueprint directly to Claude Code to refactor your C engine.

### 5.1 Configuration & Layer Descriptor Structs

```c
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef enum {
    ATTN_SLIDING = 0,
    ATTN_FULL    = 1
} AttnType;

typedef struct {
    int d_model;                 // e.g., 1536
    int intermediate_size;       // e.g., 6144
    int n_layers;                // e.g., 35
    int n_kv_shared_layers;      // e.g., 20 (layers 15..34 borrow KV)
    int n_q_heads;               // e.g., 8
    int n_kv_heads;              // 1 (Multi-Query Attention)
    int head_dim_sliding;        // e.g., 256
    int head_dim_full;           // e.g., 512
    int sliding_window;          // e.g., 512
    int max_seq_len;             // e.g., 32768 or 131072
    float rope_theta_sliding;    // 10000.0f
    float rope_theta_full;       // 1000000.0f
    float partial_rotary_full;   // 0.25f (p-RoPE on global layers)
    float rms_norm_eps;          // 1e-6f
    float attn_softcap;          // 0.0f (or 50.0f if enabled)
    float final_logit_softcap;   // 30.0f
    AttnType *layer_types;       // Array of length n_layers
} ModelConfig;

typedef struct {
    int layer_idx;
    AttnType type;
    int head_dim;                // 256 for sliding, 512 for full
    bool is_kv_shared;           // true if layer_idx >= (n_layers - n_kv_shared_layers)
    int kv_cache_slot;           // For active layers: 0..n_active-1. For shared: donor's slot!
    int donor_layer_idx;         // Self if !is_kv_shared, else last active layer of same AttnType
} LayerPlan;
```

### 5.2 Building the Donor Routing Table & Compact KV Cache

Only allocate KV memory for `n_active_layers = n_layers - n_kv_shared_layers`. Sliding layers only allocate `sliding_window` slots (ring buffer), whereas global layers allocate `max_seq_len` slots.

```c
typedef struct {
    float *k;        // [capacity * n_kv_heads * head_dim] (n_kv_heads == 1)
    float *v;        // [capacity * n_kv_heads * head_dim]
    int capacity;    // sliding_window for ATTN_SLIDING, max_seq_len for ATTN_FULL
    int head_dim;
    AttnType type;
} LayerKVCache;

typedef struct {
    int n_active_layers;
    LayerKVCache *slots; // Length: n_active_layers (NOT n_layers!)
} KVCache;

void build_layer_plans(const ModelConfig *cfg, LayerPlan *plans) {
    int boundary = cfg->n_layers - cfg->n_kv_shared_layers;
    int last_sliding_slot = -1;
    int last_sliding_layer = -1;
    int last_full_slot = -1;
    int last_full_layer = -1;

    for (int l = 0; l < cfg->n_layers; l++) {
        AttnType t = cfg->layer_types[l];
        plans[l].layer_idx = l;
        plans[l].type = t;
        plans[l].head_dim = (t == ATTN_FULL) ? cfg->head_dim_full : cfg->head_dim_sliding;

        if (l < boundary) {
            plans[l].is_kv_shared = false;
            plans[l].kv_cache_slot = l;
            plans[l].donor_layer_idx = l;
            if (t == ATTN_SLIDING) {
                last_sliding_slot = l;
                last_sliding_layer = l;
            } else {
                last_full_slot = l;
                last_full_layer = l;
            }
        } else {
            plans[l].is_kv_shared = true;
            if (t == ATTN_SLIDING) {
                plans[l].kv_cache_slot = last_sliding_slot;
                plans[l].donor_layer_idx = last_sliding_layer;
            } else {
                plans[l].kv_cache_slot = last_full_slot;
                plans[l].donor_layer_idx = last_full_layer;
            }
        }
    }
}
```

### 5.3 Layer Weights Struct (Zero K/V Allocations on Shared Layers)

```c
typedef struct {
    float *pre_attn_norm;   // [d_model]
    float *post_attn_norm;  // [d_model]
    float *pre_ffn_norm;    // [d_model]
    float *post_ffn_norm;   // [d_model]

    float *wq;              // [n_q_heads * head_dim, d_model]
    float *q_norm;          // [head_dim] (shared across Q heads)
    float *wo;              // [d_model, n_q_heads * head_dim]

    // NULL when plan.is_kv_shared == true! Saves ~57% of K/V weights in a 35-layer model
    float *wk;              // [1 * head_dim, d_model] or NULL
    float *wv;              // [1 * head_dim, d_model] or NULL
    float *k_norm;          // [head_dim] or NULL (V-norm has no learnable gamma!)

    // GeGLU MLP
    float *w_gate;          // [intermediate_size, d_model]
    float *w_up;            // [intermediate_size, d_model]
    float *w_down;          // [d_model, intermediate_size]
} LayerWeights;
```

### 5.4 Core C Kernels: QKV-Norm, p-RoPE, and Shared-Cache Attention

```c
#include <math.h>

// 1. RMSNorm with optional scale weight (pass gamma = NULL for V-Norm)
void rms_norm(float *out, const float *x, const float *gamma, int dim, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) {
        ss += x[i] * x[i];
    }
    float inv_rms = 1.0f / sqrtf((ss / (float)dim) + eps);
    if (gamma != NULL) {
        for (int i = 0; i < dim; i++) {
            out[i] = x[i] * inv_rms * gamma[i];
        }
    } else {
        for (int i = 0; i < dim; i++) {
            out[i] = x[i] * inv_rms;
        }
    }
}

// 2. Proportional Partial RoPE (NeoX split-half style)
// Note: inv_freq denominator is always full head_dim, even when partial_factor < 1.0!
void apply_proportional_rope(
    float *vec,           // [head_dim]
    int head_dim,
    int pos,
    float theta,
    float partial_factor  // 1.0f for sliding, 0.25f for full
) {
    int half_dim = head_dim / 2;
    int rot_pairs = (int)(partial_factor * (float)half_dim);

    for (int k = 0; k < rot_pairs; k++) {
        float freq = 1.0f / powf(theta, (float)(2 * k) / (float)head_dim);
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);

        float x0 = vec[k];
        float x1 = vec[k + half_dim];
        vec[k]            = x0 * c - x1 * s;
        vec[k + half_dim] = x1 * c + x0 * s;
    }
    // Pairs k in [rot_pairs, half_dim - 1] have freq = 0 -> identity (NoPE)
}

// 3. Forward Attention Step for Token at Position `pos`
void attention_forward_step(
    float *attn_out,              // [n_q_heads * head_dim]
    const float *x_norm,          // [d_model]
    const LayerWeights *w,
    const LayerPlan *plan,
    const ModelConfig *cfg,
    KVCache *cache,
    int pos,
    float *q_buf,                 // scratch [n_q_heads * head_dim]
    float *scores_buf             // scratch [max_seq_len]
) {
    int hd = plan->head_dim;
    int n_q = cfg->n_q_heads;
    float p_factor = (plan->type == ATTN_FULL) ? cfg->partial_rotary_full : 1.0f;
    float theta    = (plan->type == ATTN_FULL) ? cfg->rope_theta_full : cfg->rope_theta_sliding;

    // Step A: Compute Q for ALL layers (active and shared)
    matmul(q_buf, w->wq, x_norm, n_q * hd, cfg->d_model);
    for (int h = 0; h < n_q; h++) {
        float *qh = q_buf + h * hd;
        rms_norm(qh, qh, w->q_norm, hd, cfg->rms_norm_eps);
        apply_proportional_rope(qh, hd, pos, theta, p_factor);
    }

    // Step B: Get target KV cache slot (own slot if active, donor slot if shared)
    LayerKVCache *kv = &cache->slots[plan->kv_cache_slot];
    int write_idx = (plan->type == ATTN_SLIDING) ? (pos % kv->capacity) : pos;

    // Step C: Compute & store K, V ONLY if this is an active (non-shared) layer
    if (!plan->is_kv_shared) {
        float *k_dst = kv->k + (size_t)write_idx * hd;
        float *v_dst = kv->v + (size_t)write_idx * hd;

        matmul(k_dst, w->wk, x_norm, hd, cfg->d_model);
        matmul(v_dst, w->wv, x_norm, hd, cfg->d_model);

        // Normalize K (with gamma) and V (without gamma), then RoPE K only
        rms_norm(k_dst, k_dst, w->k_norm, hd, cfg->rms_norm_eps);
        rms_norm(v_dst, v_dst, NULL,      hd, cfg->rms_norm_eps);
        apply_proportional_rope(k_dst, hd, pos, theta, p_factor);
    }

    // Step D: Determine valid causal window [start_pos, pos]
    int start_pos = 0;
    if (plan->type == ATTN_SLIDING && pos >= cfg->sliding_window) {
        start_pos = pos - cfg->sliding_window + 1;
    }
    int valid_len = pos - start_pos + 1;

    // Step E: Multi-Query Attention over the single KV head (n_kv_heads = 1)
    for (int h = 0; h < n_q; h++) {
        const float *qh = q_buf + h * hd;
        float max_score = -INFINITY;

        for (int i = 0; i < valid_len; i++) {
            int t = start_pos + i;
            int read_idx = (plan->type == ATTN_SLIDING) ? (t % kv->capacity) : t;
            const float *kt = kv->k + (size_t)read_idx * hd;

            // Dot product with scale = 1.0 (since Q and K are RMS-normalized)
            float score = 0.0f;
            for (int d = 0; d < hd; d++) {
                score += qh[d] * kt[d];
            }
            if (cfg->attn_softcap > 0.0f) {
                score = cfg->attn_softcap * tanhf(score / cfg->attn_softcap);
            }
            scores_buf[i] = score;
            if (score > max_score) max_score = score;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int i = 0; i < valid_len; i++) {
            scores_buf[i] = expf(scores_buf[i] - max_score);
            sum_exp += scores_buf[i];
        }
        float inv_sum = 1.0f / sum_exp;

        // Weighted sum of V
        float *oh = attn_out + h * hd;
        for (int d = 0; d < hd; d++) oh[d] = 0.0f;

        for (int i = 0; i < valid_len; i++) {
            int t = start_pos + i;
            int read_idx = (plan->type == ATTN_SLIDING) ? (t % kv->capacity) : t;
            const float *vt = kv->v + (size_t)read_idx * hd;
            float prob = scores_buf[i] * inv_sum;

            for (int d = 0; d < hd; d++) {
                oh[d] += prob * vt[d];
            }
        }
    }
}
```

---

## 6. Checklist for Claude Code Implementation

When handing this document to Claude Code, instruct it to verify these 6 invariants:

1. **Zero KV Allocation for Shared Layers:** Assert `cache->n_active_layers == cfg->n_layers - cfg->n_kv_shared_layers`. Never allocate `wk`, `wv`, `k_norm`, or `KVCache` buffers for `l >= n_active_layers`.
2. **No Double-Norm or Double-RoPE:** Verify `k_dst` is normalized and rotated *once* when written by the active donor layer. Borrower layers must read `kv->k` as-is.
3. **Ring-Buffer Modulo Indexing for Sliding Layers:** Both writes (`pos % sliding_window`) and reads (`t % sliding_window`) in `ATTN_SLIDING` layers must use ring-buffer indexing so sliding KV memory stays $O(1)$ regardless of sequence length.
4. **Proportional RoPE Denominator:** Ensure `powf(theta, (2.0f * k) / head_dim)` divides by full `head_dim` (512), *not* `rotary_dim` (128), and leaves indices `[rot_pairs, head_dim/2)` untouched.
5. **Pre- + Post-Norm Residual Wiring:** Ensure every sub-layer (Attention and MLP) applies `RMSNorm` both *before* entering the sub-layer and *after* the sub-layer output before adding to the residual stream.
6. **Quantization Scale Sharing (If Using INT8/FP8 KV Cache):** If quantizing the KV cache, borrower layers must dequantize using `donor_layer`'s `k_scale` and `v_scale`, never a default `1.0f` scale.
