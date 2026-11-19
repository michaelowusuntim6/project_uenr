# Datasets: configuration, formats and fetching

Dataset selection lives only in **`config/uenr.toml`** under `[datasets.*]`.
The training binaries read the prepared files named in `[paths]`; the workflow
and `scripts/bootstrap_datasets.sh` read the recipes in `[datasets.*]`.
Changing a dataset does **not** require editing `.github/workflows/train.yml`.

## How the pieces fit

```
[datasets.<phase>].paths ──► uenr prepare --phase <phase> ──► [paths].* file
        │                              ▲
        │ fetch recipe                 │ tokenizer (pretrain only)
        ▼                              │
scripts/bootstrap_datasets.sh ──► ./uenr datasets
```

1. `./uenr datasets` prints one unit-separator-separated (0x1f) recipe per
   configured path. It is generated from the fully validated TOML, not a
   second config file.
2. `scripts/bootstrap_datasets.sh` downloads/caches each path (release asset
   first, then `url`, then `derive_from`) and publishes new assets.
3. `./uenr prepare --phase <phase>` converts the source files into the exact
   files the trainer consumes.
4. The GitHub workflow calls the same script in both `restore-datasets` and
   `prepare-all`, so a browser-triggered run uses the TOML of the selected ref.

## Common keys

| Key | Meaning |
|---|---|
| `paths` | Up to 8 source files, concatenated in order then shuffled. |
| `format` | One of the formats below. Unknown formats fail config validation. |
| `shuffle_seed` | Deterministic shuffle seed (0 uses `[run].seed`). |
| `max_samples` | Cap after shuffling (0 = no cap). |
| `max_bytes` | Pretrain-only raw byte budget. |
| `release_tag` | GitHub release tag holding the cached source asset. |
| `release_asset` | Asset name on that tag. |
| `url` | Direct download URL used when no cached asset exists. |
| `derive_from` | Phase whose first path is the source (e.g. `"sft"`). |
| `convert_from` / `convert_to` | `uenr convert` names for materialising the file. |
| `limit` | `uenr convert --limit` (0 = all). |
| `bootstrap` | `true` allows fetch/convert; `false` requires the file to exist. |

`derive_from` accepts `pretrain`, `sft`, `lora`, `dpo`, `rl` or `reward`.
`convert_from`/`convert_to` are only used for cold-cache materialisation; a
cached release asset or an existing local file is used verbatim.

The fetch recipe (`release_*`, `url`, `derive_from`, `convert_*`) applies to
the **first** entry in `paths`. Additional paths must already exist locally
(or be produced by another phase); this keeps multi-file corpora explicit.

## Phase formats

| Phase | Section | `format` | Prepared output | Consumed by |
|---|---|---|---|---|
| pretrain | `[datasets.pretrain]` | `text_lines` | `data/train.bin`, `data/val.bin` (UDAT) | `uenr train` |
| SFT | `[datasets.sft]` | `jsonl_prompt_completion`, `jsonl_messages` | `data/sft.jsonl` | `uenr sft` |
| DPO | `[datasets.dpo]` | `jsonl_prompt_chosen_rejected` | `data/pairs.jsonl` | `uenr dpo` |
| GRPO | `[datasets.rl]` | `jsonl_prompt_only` | `data/rl_prompts.jsonl` | `uenr grpo` |
| reward (legacy) | `[datasets.reward]` | `jsonl_prompt_response_label` | `data/reward_pairs.jsonl` | `uenr train-rm` (retired) |
| eval | `[datasets.eval]` | `bin` / no paths | uses `data/val.bin` | `uenr eval` |
| LoRA (legacy) | `[datasets.lora]` | `inherit`, JSONL messages/prompt-completion | `data/lora.jsonl` | `uenr lora` (retired) |

### pretrain — `text_lines`

Plain UTF-8 text, **one document per line** (blank lines are skipped):

```text
Once upon a time there was a little robot who counted apples.
The sun was warm and the robot walked home.
```

`max_bytes` bounds the bytes read; `[prepare].val_percent` holds out the
validation split. Requires `uenr tok-train` before `uenr prepare --phase
pretrain` (or `--phase all`).

### SFT — `jsonl_prompt_completion`

One JSON object per line with non-empty `prompt` and `completion` strings:

```json
{"prompt":"Say hello","completion":"Hello there!"}
{"prompt":"What is the weather?","completion":"It is sunny today."}
```

The preparer rewrites these into chat messages (`user` + `assistant`).
Missing/empty fields make that line invalid and it is skipped.

### SFT — `jsonl_messages`

One JSON object per line in chat form; it is copied through unchanged:

```json
{"messages":[{"role":"system","content":"You help."},{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"}]}
{"messages":[{"role":"user","content":"Add 2 and 3"},{"role":"assistant","content":"5"}]}
```

Roles understood by the trainer: `system`, `user`, `assistant`, `tool`.
Only assistant content and the closing `<end>` are supervised.

### DPO — `jsonl_prompt_chosen_rejected`

All three fields are required and non-empty:

```json
{"prompt":"Tell me a story","chosen":"Once upon a time a fox ran home.","rejected":"home ran fox a time upon Once."}
```

### GRPO — `jsonl_prompt_only`

Only a prompt is required; completions are sampled live:

```json
{"prompt":"What is 12 + 7?"}
{"prompt":"Tell me about the moon."}
```

Arithmetic prompts of the form `a + b`, `a - b`, `a * b` (also `x`/`×`)
activate the R1 exact-answer verifier (`[rl].reward_verify`).

### reward (legacy) — `jsonl_prompt_response_label`

```json
{"prompt":"Say hi","response":"Hi!","label":1}
```

### eval — `bin`

Empty `paths` uses the pretrain `val.bin`. A non-empty entry must be a `UDAT`
binary produced by `uenr prepare`.

## Changing a dataset: local

```toml
[datasets.sft]
paths = ["data/my_sft.jsonl"]
format = "jsonl_prompt_completion"
shuffle_seed = 42
max_samples = 200000
bootstrap = true          # already-present file is used as-is
```

```bash
./uenr tok-train           # only for pretrain / --phase all
./uenr prepare --phase sft
```

To fetch a new dataset from a URL, set `url` and (optionally) `convert_from`
/`convert_to`:

```toml
[datasets.sft]
paths = ["data/my_sft.jsonl"]
format = "jsonl_prompt_completion"
url = "https://example.org/my_sft.jsonl"
derive_from = ""
convert_from = ""
convert_to = ""
bootstrap = true
```

```bash
scripts/bootstrap_datasets.sh
./uenr prepare --phase sft
```

To cache it as a release asset, set `release_tag` and `release_asset`; the
script uploads the file when the asset is missing and downloads it later.

## Changing a dataset: GitHub Actions

Edit `config/uenr.toml` on the branch you dispatch, commit, and start the
workflow from the browser or `scripts/dispatch.sh`. Both `restore-datasets`
and `prepare-all` build the binary and call `scripts/bootstrap_datasets.sh`,
which reads `./uenr datasets` from that checkout. No workflow edit is needed.

Order matters for derived sets: `bootstrap_datasets.sh` follows the phase
order pretrain → sft → dpo → rl, so `derive_from = "sft"` sees the SFT file
that was materialised just before it.

If `bootstrap = false` and the file is missing, the workflow fails early with
the exact path and phase instead of silently training on stale data.

## Converting external corpora

The C converter supports the canned paths used by the defaults:

```bash
# TinyStoriesInstruct text blocks -> prompt/completion JSONL
./uenr convert --from tinystories_instruct --to prompt_completion \
    --in TinyStories-Instruct-valid.txt --out data/sft.jsonl --limit 200000

# prompt/completion JSONL -> DPO pairs (chosen vs word-reversed rejected)
./uenr convert --from prompt_completion --to prefs \
    --in data/sft.jsonl --out data/pairs.jsonl --limit 50000

# prompt/completion JSONL -> prompt-only RL data
./uenr convert --from prompt_completion --to prompts \
    --in data/sft.jsonl --out data/rl_prompts.jsonl --limit 5000
```

Parquet sources must be converted to these formats offline first. The
workflow never runs Python or reads parquet.

## Defaults shipped in `config/uenr.toml`

| Phase | Path | Release tag / asset |
|---|---|---|
| pretrain | `data/tinystories_train.txt` | `dataset-pretrain` / `tinystories_train.txt` |
| SFT | `data/sft_tinystories.jsonl` | `dataset-sft` / `sft_tinystories.jsonl` |
| DPO | `data/prefs_tinystories.jsonl` | `dataset-dpo` / `prefs_tinystories.jsonl` |
| GRPO | `data/rl_prompts.jsonl` | `dataset-rl` / `rl_prompts.jsonl` |

These are the datasets used by the current full run; override any of them in
the TOML and the next workflow dispatch follows the new recipe.
