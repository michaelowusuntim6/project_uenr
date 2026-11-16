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

For each new token the engine runs the full transformer forward over the
current sequence (`T = S = length`, batch 1), takes the logits of the last
position, samples, appends, and repeats until `max_new_tokens`, `<eos>`,
`<end>`, or `max_seq`.

This full-context recompute is simple and exact. Measured on the target
machine: ~65 tokens/s for short contexts; it slows as the context grows
because attention is O(S²) per token. A GQA KV-cache decode path is the main
roadmap item; the cache layout and config flag (`inference.kv_cache`) are
already reserved.

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
