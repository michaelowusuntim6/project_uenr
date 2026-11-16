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
Only donor layers own cache slots. Sliding slots hold `window` rows in a ring
buffer (`pos % window`); global slots hold `max_seq`. At the shipped shape the
cache is 640 rows at S=512 (160 KiB FP32) and 1,000,128 rows at 1M tokens
(256 MB FP32 / 128 MB FP16 / 64 MB INT8 / 40 MB Q4). Q4 stores every K/V value
in 4 signed bits with one FP32 scale per KV head and row; decode unpacks with
AVX1 integer/float ops. Measured on the real model: max logit error 0.061,
mean 0.003, 64/64 greedy argmax agreement; at 512/2048/4096-token contexts,
Q4 decode is 0.76×/0.64×/0.60× FP32 (memory-bound caches do not benefit on
this CPU), while using 6.4× less KV memory. FP16 is the best compromise for
bandwidth; FP32 remains the fastest at short context.

Measured on the target laptop: **~1.0–1.1k tok/s** for 128-token cached decode
(the original full-context path was ~65 tok/s). A unit test compares cached
steps, batched spans and multi-sequence decode against full recompute; FP32 is
bit-identical, FP16/INT8 match within tolerance.

Setting `kv_cache = false` keeps the full-context recompute path as a
reference/fallback. `kv_int8 = true` quantizes K/V per KV head with per-head
scales (4x less cache RAM, 1.04x decode at 128 tokens, max logit error
4.5e-3). `weight_int8 = true` quantizes linear weights per output row and
dequantizes in an AVX1 GEMV (1.15x decode, ~8.5 MiB extra quantized copies,
max logit error 4.7e-4 on the test model). The KV cache also enables the long-context work in items 49–57.

## Batched inference

`uenr infer --batch prompts.txt --tokens N` reads up to `inference.batch_max`
prompts, prefills one compact KV cache per sequence, and decodes in lockstep
with batched projections and shared-expert GEMMs (MoE tokens are grouped across
the batch). Greedy outputs match sequential inference. Measured 8 prompts × 20
tokens at **1718.5 tok/s** vs 538.5 tok/s aggregate sequential (**3.19×**);
B=4 gives 1.17× because top-1 experts differ per token.

## Speculative decoding

`inference.speculative` adds prompt-lookup proposals verified by
`uenr_model_forward_span` (k cached tokens in one batched pass). Greedy output
is exactly identical to normal decode. On the smoke model acceptance was 0/37,
so it defaults **off** (0.38×); it is useful with a draft model or repetitive
text.

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

## Decoding constraints

`[inference]` supports `repetition_penalty` (with `repetition_last_n`),
`no_repeat_ngram`, `min_p`, and `json_constrained`. The first three act on the
logits/probabilities before sampling. With `json_constrained`, sampling a
`<|tool_call|>` switches on an incremental structural JSON grammar: candidate
tokens are decoded and accepted only if they keep braces/brackets balanced and
strings well-formed; invalid candidates are masked and resampled, valid
structural byte tokens are forced if needed, and generation can only end after
the JSON value closes. A structural validator (`uenr_json_valid`) is exposed
for tests. Measured at smoke scale, repetition penalty + no-repeat-3 cuts the
repeated word-trigram rate from 2.68% to 0.00% at 1.00x decode cost.

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
