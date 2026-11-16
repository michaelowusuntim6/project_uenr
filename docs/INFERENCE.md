# Inference

The inference engine is `src/infer/inference.c`, exposed as `uenr infer` and
as menu option 13.

## Loading

- Load `artifacts/tokenizer.bin` and the checkpoint (`--model`, default
  `paths.model_out`).
- The model is re-initialized from the config (same shapes), then parameters
  are overwritten from the `UMDL` file; a shape mismatch is rejected.
- Inference uses a single thread (decode is memory-bound; a second thread only
  adds contention).

## Prompt and chat

`uenr_chat_infer` renders:

```
<bos> [<system> system <end>] <user> user <end> <assistant>
```

Special-token markup in user/assistant text is preserved atomically by the
tokenizer. The default system prompt is `inference.system` in the config.

## Generation

With `inference.kv_cache = true` (default), prompt tokens are prefilled one
at a time into a per-layer GQA K/V cache; each generated token runs a single
incremental step: project Q/K/V, append K/V at `pos`, attend over the cached
`0..pos` window, run the MoE block, and emit logits. The cache is
`2·L·G·head_dim·S` floats (1 MiB at S=512 for the shipped config).

Measured on the target laptop: **69.5 → 667.3 tok/s (9.6×)** for a 20-token
generation. A unit test compares every cached step's logits against the
full-context forward and gets bit-identical results.

Setting `kv_cache = false` keeps the full-context recompute path as a
reference/fallback. `kv_int8 = true` quantizes K/V per KV head with per-head
scales (4x less cache RAM, 1.04x decode at 128 tokens, max logit error
4.5e-3). The KV cache also enables the long-context work in items 49–57.

## Sampling

- `greedy = true`: argmax.
- Otherwise softmax at `temperature`, then:
  - **top-k**: select the k-th largest probability by repeated max scans and
    zero everything below it;
  - **top-p** (nucleus): sort `(prob, id)` pairs, keep the smallest prefix with
    cumulative probability ≥ `top_p`, zero the rest;
  - renormalize and sample with the xorshift128+ RNG.

Special tokens are in the vocabulary and can be sampled; generation stops on
`<eos>`/`<end>`. Byte tokens can decode to arbitrary bytes, so output is not
guaranteed to be valid UTF-8.

## Interactive chat

`uenr infer --chat` (and menu option 13) reads lines until an empty line. The
menu calls the library directly rather than spawning a child so the menu and
chat share one stdin buffer.

## CLI examples

```bash
./uenr infer --prompt "Once upon a time" --tokens 64 --greedy
./uenr infer --chat
./uenr infer --model checkpoints/sft.bin --prompt "Tell me a story"
```
