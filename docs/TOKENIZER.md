# Tokenizer

UENR ships a handwritten **byte-level BPE** tokenizer. It is trained from the
corpus, stored in a compact binary artifact, and used for pretraining data,
SFT chat samples, DPO pairs, and inference. There is no Python tokenizer
dependency.

## Id layout

| Range | Meaning |
|---|---|
| `0 .. 15` | Reserved special tokens (ids chosen in `config/uenr.toml`). |
| `16 .. 271` | The 256 raw bytes (`id = 16 + byte`). |
| `272 .. V-1` | BPE merges in rank order (`id = 272 + rank`). |

Default specials: pad=0, bos=1, eos=2, unk=3, system=4, user=5, assistant=6,
tool_call=7, tool_result=8, end=9. Ids 10–15 are free. Because merges only
reference ids ≥ 16, special ids are natural **barriers**: the encoder never
merges across a special token.

## Training (`uenr tok-train`)

1. Read up to `tokenizer.train_bytes` bytes of the corpus (`data` path).
2. Sanitize the sample (see below).
3. Map every byte to `16 + byte`.
4. For each merge rank:
   - count adjacent pairs with an open-addressing hash table keyed by
     `(a << 32) | b`;
   - the table uses a **generation stamp** so each iteration is cleared in
     O(1) instead of zeroing 64 MiB;
   - pick the most frequent pair, tie-broken by the smallest numeric key;
   - replace every occurrence in one linear pass, assigning the new id.
5. Write `artifacts/tokenizer.bin` (magic `UTOK`, version, vocab size, merge
   count, merge pairs) plus human-readable `vocab.txt` and `merges.txt`.

The default config trains 1776 merges from 1.5 MB of TinyStories, producing
353,769 tokens (~4.2 bytes/token) in a few seconds.

## Encoding

Encoding is a linked list + min-heap algorithm, not a repeated full scan:

- Build a doubly linked list over the byte ids.
- Push every adjacent pair that exists in the merge table into a min-heap
  ordered by **merge rank**, then position.
- Pop the best pair; verify it is still adjacent and its rank has not changed;
  merge it in O(1); push the two new neighbouring pairs.
- The result is O(n log n) per sequence, and it never merges across a special.

The merge table is an open-addressing hash map `(a,b) → rank`, built when the
artifact is loaded.

## Decoding

Special ids render as their markup strings (`<|user|>`, `<|tool_call|>`, …).
Byte ids render the raw byte. Merge ids recursively expand their two children.
The decoder returns a NUL-terminated byte buffer.

At inference, arbitrary byte output can produce invalid UTF-8; the engine
prints the raw bytes. A display sanitizer is a roadmap item.

## Sanitization

`uenr_tok_sanitize` is applied to every input before encoding:

- `\r\n` and `\r` become `\n`;
- runs of more than two newlines collapse to two;
- C0 control bytes except `\n` and `\t` are dropped when `strip_control`
  is set (DEL too);
- ASCII lowercase when `lowercase = true`;
- UTF-8 validation: overlong encodings, surrogates, out-of-range code points,
  and truncated sequences are replaced by `?`;
- the transform is otherwise lossless at the byte level.

`normalize_unicode` currently means UTF-8 validation and line-ending
normalisation; full NFC/NFKC composition is not implemented (roadmap).

## Chat template and tool calls

`uenr_chat_infer` builds:

```
<bos> [<system> sys <end>] <user> user <end> <assistant>
```

`uenr_chat_sft` builds the same plus the assistant completion and `<end>`, and
returns per-token labels where **only the assistant span and its `<end>` are
supervised**; system/user text is `ignore_index`.

When content contains literal markup such as `<|tool_call|>`,
`<|tool_result|>`, or `<|end|>`, the tokenizer emits the **atomic special id**
instead of tokenizing the characters. This keeps tool-call payloads intact and
lets SFT supervise the tool-call boundaries. An assistant message like

```
Sure.<|tool_call|>{"name":"weather","args":{"city":"Accra"}}<|end|>
```

is encoded as text tokens for `Sure.`, the tool-call special id, the JSON body,
and the end special id, with all of it supervised.

## SFT / preference data

`uenr prepare` also writes:

- `data/sft.jsonl`: `{"messages":[{"role":"user",...},{"role":"assistant",...}]}`
  synthesized from TinyStories prompts/completions (configurable count).
- `data/pairs.jsonl`: `{"prompt":..., "chosen":..., "rejected":...}` for DPO.
  The rejected completion is the previous story's continuation, which is a
  cheap but valid preference signal for a smoke-scale alignment demo.

The JSONL reader is a small handwritten JSON parser: strings, escapes,
`\uXXXX`, nested skip, top-level objects, and `messages` arrays. It preserves
tool-call content verbatim.
