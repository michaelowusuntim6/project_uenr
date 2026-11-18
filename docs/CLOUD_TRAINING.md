# Cloud training on GitHub Actions

## What runs

- Runner: `ubuntu-latest` (4 vCPU AMD EPYC 7763v, Zen 3, AVX2, 16 GB RAM,
  14 GB SSD).
- Job timeout: 350 minutes (GitHub cap is 360; margin added).
- Trigger: manual `workflow_dispatch` from the Actions UI.
- Build flags: `-O3 -march=native -mavx2 -mf16c -pthread`, `make -j$(nproc)`.
- Training threads: `run.threads = 4` (main + 3 workers, CPUs 0–3).
- Pipeline: `./uenr tok-train` → `./uenr prepare` → `./uenr train --steps N`.
- Workflow: `.github/workflows/train.yml` on `master`.

## How to trigger

1. Open https://github.com/michaelowusuntim6/project_uenr/actions/workflows/train.yml
2. Click **Run workflow**, choose a `mode` (`smoke` first), branch `master`,
   step counts, reward source and whether to bootstrap datasets.
3. Watch the live log. Each phase publishes a stable `<phase>-checkpoint`
   release and a per-run `<phase>-checkpoint-run` artifact.

## Config for the cloud runner

| Key | Cloud value | Laptop default | Rationale |
|---|---:|---:|---|
| `run.threads` | 4 | 2 | use all four vCPUs on the runner |
| `run.pin_cpu0` | 0 | 0 | first distinct vCPU on the 4-vCPU runner |
| `run.pin_cpu1` | 1 | 2 | second distinct vCPU (laptop assumed physical 0/2) |
| `training.batch_size` | 16 | 8 | 16 GB runner; micro-batch becomes 8 sequences |
| `training.grad_accum` | 2 | 2 | unchanged; effective batch is 16 sequences |
| `prepare.max_bytes` | 2000000000 | 209715200 | covers the full ~1.92 GB TinyStories file |
| `schedule.target_loss` | 0.0 | 2.2 | CI runs exactly the requested steps |
| `schedule.plateau_window` | 100000 | 300 | no plateau stop before the step budget |

`training.block_checkpoint` remains `false` (16 GB is ample for the 8.45M
model) and `run.checkpoint_every` remains `500`. With `run.threads = 4`, the
two configured pins (0, 1) are respected and the extra workers are
auto-pinned to the next unused CPUs (2 and 3).

The laptop default is preserved as an inline `# laptop default:` comment on
each changed line in `config/uenr.toml`, so reverting is one edit per key.

## Dataset

Downloaded from
https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
(~1.92 GB) on every run. **Gap:** this is re-downloaded each time (~10–25 min).
A follow-up will mirror it as a release asset on tag `tiny-stories-dataset`
and switch to `gh release download` on subsequent runs.

## Checkpoints and outputs

- Stable per-phase release tags: `pretrain-checkpoint`, `sft-checkpoint`,
  `lora-checkpoint`, `dpo-checkpoint`, `grpo-checkpoint`, `ppo-checkpoint`,
  `train-rm-checkpoint`. Each holds the latest `<phase>.bin` with `--clobber`.
- Per-run workflow artifacts: `<phase>-checkpoint-run` (final phase file only)
  and the run summary table.
- Periodic `checkpoints/step_*.bin` files are **not** uploaded; they stay on
  the ephemeral runner.
- Pretrain resumes from its stable checkpoint; other phases load their own
  previous checkpoint as the starting weights. Cross-run *optimizer* resume is
  implemented only for pretrain.

## Known gaps (follow-up work)

1. **First-run bootstrap cost:** the first run downloads and uploads the
   ~1.92 GB pretrain asset; later runs use `dataset-pretrain`/`dataset-sft`/
   `dataset-dpo`/`dataset-rl` release caches.
2. **Non-pretrain optimizer resume:** SFT/LoRA/DPO/GRPO/PPO load the previous
   phase checkpoint as weights, but their optimizer step counters are not
   resumed. Pretrain `--resume` restores optimizer state.
3. **Optional labelled reward data:** `[datasets.reward]` accepts
   `jsonl_prompt_response_label`, but the current trainer consumes chosen/
   rejected pairs; the heuristic reward is the default. Use DPO pairs for
   `uenr train-rm`.

## What this workflow does NOT do

- Does not touch `/home/mike/UENR/uenr` (laptop project).
- Does not pull large artifacts back to the laptop (slow link).
- Does not require any repository secrets (uses built-in `GITHUB_TOKEN`).

## Dataset recommendations by phase

The workflow defaults are the domain-specific choices below; alternatives are
documented for offline conversion.  The workflow itself never runs Python and
never reads parquet: parquet-only sources are converted once on a development
machine and published as stable release assets (see
`scripts/fetch_and_convert.sh`).

| Phase | Default in this repo | Alternatives | HF URL | Approx. size | On-disk format |
|---|---|---|---|---|---|
| Pretrain | `roneneldan/TinyStories` | `SimpleStories/SimpleStories` (2M English + 2M Japanese; reported better sample efficiency at small scale) | https://huggingface.co/datasets/roneneldan/TinyStories | ~1.92 GB | plain UTF-8 text, one story per line (`text_lines`) |
| SFT | `roneneldan/TinyStoriesInstruct` validation split (`TinyStories-Instruct-valid.txt`) | `joey00072/TinyStories-SFT`; `HydraLM/TinyStoriesInstruct-standardized`; `maomao88/TinyStoriesInstruct-Formatted` | https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct | ~27 MB (valid); ~2.66 GB full train | prompt/completion JSONL, converted from the text blocks |
| LoRA / QLoRA | inherits SFT | any SFT corpus | same as SFT | same as SFT | empty `[datasets.lora].paths` ⇒ fall back to SFT |
| Reward model | heuristic `[reward]` (no file) | DPO pairs relabelled chosen=1 / rejected=0, converted offline | DPO URL below | small | `jsonl_prompt_response_label` (optional) |
| DPO | synthetic domain pairs derived from the SFT set (chosen story vs word-reversed rejected story) | `HuggingFaceH4/ultrafeedback_binarized` (canonical, parquet; convert offline) | https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized | ~50k pairs JSONL (a few tens of MB) | prompt/chosen/rejected JSONL |
| GRPO / PPO prompts | prompts extracted from the SFT instruction set | `mlabonne/smoltldr` (2,000 summary prompts, parquet; convert offline) | https://huggingface.co/datasets/mlabonne/smoltldr | ~5k prompts (a few MB) | prompt-only JSONL, completions sampled live |
| Eval | held-out pretrain split | held-out SFT/DPO slice | derived from pretrain | data-dependent | `UDAT` uint16 `val.bin` |

Justification: TinyStories and its instruct variant match the 8.45M model's
vocabulary and scale; the DPO/RL defaults are derived from that same domain
instead of out-of-domain Reddit summaries, and avoid parquet readers in CI.
The alternatives remain available through the offline conversion script.

## Workflow modes

| Mode | Jobs that run | Default steps | Estimated wall time | Outputs |
|---|---|---|---|---|
| `pretrain` | restore → prepare → build → pretrain | 2000 | 1–3 h | `pretrain-checkpoint` |
| `sft` | prepare → sft (loads cached pretrain) | 300 | 20–45 min | `sft-checkpoint` |
| `lora` | prepare → lora (loads cached pretrain) | 300 | 20–45 min | `lora-checkpoint` |
| `dpo` | prepare → dpo (loads cached sft, pretrain ref) | 300 | 10–30 min | `dpo-checkpoint` |
| `grpo` | prepare → grpo (loads cached sft) | 200 | 20–60 min | `grpo-checkpoint` |
| `ppo` | prepare → ppo (loads cached sft) | 200 | 20–50 min | `ppo-checkpoint` |
| `train-rm` | prepare → reward model | 150 | 5–20 min | `train-rm-checkpoint` |
| `eval` | prepare → eval (loads cached pretrain) | n/a | < 5 min | run summary only |
| `full-core` | pretrain → sft → lora → eval | 2000/300/300 | < 300 min | pretrain/sft/lora checkpoints |
| `full-rl` | dpo → grpo → ppo → eval (cached pretrain + sft) | 300/200/200 | < 300 min | dpo/grpo/ppo checkpoints |
| `smoke` | all phases, tiny step counts | 10–20 each | target < 15 min (first bootstrap run can be longer) | all checkpoints |

Only stable per-phase release tags are created.  Per-run history is kept as
workflow artifacts named `<phase>-checkpoint-run`.
