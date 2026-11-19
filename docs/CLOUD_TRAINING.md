# Cloud training on GitHub Actions

## What runs

- Runner: `ubuntu-latest` (4 logical vCPUs over **2 physical cores**,
  2 SMT threads/core, 16 GB RAM). Observed hosts on 2026-11-18 were AMD
  EPYC 9V74 and 9V45 (Zen 4 class) with AVX2/FMA/AVX-512 and a shared 32 MiB
  L3. `thread_siblings_list` was `0-1` and `2-3`, i.e. CPUs 0/1 are SMT
  siblings of core 0 and 2/3 of core 1. The diagnostic job prints
  `lscpu -e` and the sibling lists on every run because the host model can
  change.
- Job timeout: 350 minutes (GitHub cap is 360; margin added).
- Trigger: manual `workflow_dispatch` from the Actions UI.
- Build: plain `make -j$(nproc)`. Normal objects use `-march=ivybridge`
  (the laptop path); `src/**/*_avx2.c` objects use
  `-march=x86-64-v3 -mavx2 -mfma` and are selected only after a runtime
  AVX2+FMA check. This avoids `-march=native` (build and train jobs may land
  on different hosts) and keeps one binary usable on both the runner and the
  Ivy Bridge laptop.
- Training threads: `run.threads = 4` (main + 3 workers, CPUs 0–3). Four
  logical CPUs over two physical cores measured faster than two threads in
  both GEMM and 100-step pretrain. A 2-thread run on this topology must pin
  CPUs 0 and 2. `[run] simd = "auto"` selects AVX2+FMA on the runner.
- Pipeline: dataset bootstrap → `./uenr prepare --phase all` → the selected
  training phases, each loading the previous phase's stable checkpoint.
- Workflow: `.github/workflows/train.yml` on `master`.

## How to trigger

1. Open https://github.com/michaelowusuntim6/project_uenr/actions/workflows/train.yml
2. Click **Run workflow**, choose a `mode` (`smoke` first), branch `master`,
   per-phase step counts, `batch_size` (default 64), `lr` (default 0.00141),
   reward source and whether to bootstrap datasets.
3. Or use the local orchestrator: `./uenr-menu` option **22. Cloud / GitHub
   trainer**, or `scripts/dispatch.sh <mode> [--steps N] [--watch]`. Both use
   the token from `github_token.txt` (or `gh auth`); see
   [GITHUB_ORCHESTRATION.md](GITHUB_ORCHESTRATION.md).
3. Watch the live log. Each phase publishes a stable `<phase>-checkpoint`
   release and a per-run `<phase>-checkpoint-run` artifact.

## Config for the cloud runner

| Key | Cloud value | Laptop default | Rationale |
|---|---:|---:|---|
| `run.threads` | 4 | 2 | use all four vCPUs on the runner |
| `run.pin_cpu0` | 0 | 0 | first distinct vCPU on the 4-vCPU runner |
| `run.pin_cpu1` | 1 | 2 | second distinct vCPU (laptop assumed physical 0/2) |
| `training.batch_size` | 64 | 16 | bench winner; 16,384 tokens/step at seq 256 |
| `training.grad_accum` | 1 | 2 | one micro-batch of 64; laptop 16/2 |
| `training.lr` | 0.00141 | 0.001 | sqrt(64/32) scaling for the batch-64 winner |
| `run.simd` | auto (AVX-512 on the runner) | auto (AVX1) | runtime dispatch |
| `prepare.max_bytes` | 2000000000 | 209715200 | covers the full ~1.92 GB TinyStories file |
| `schedule.target_loss` | 0.0 | 2.2 | CI runs exactly the requested steps |
| `schedule.plateau_window` | 100000 | 300 | no plateau stop before the step budget |

`training.block_checkpoint` remains `false` (16 GB is ample for the 8.45M
model) and `run.checkpoint_every` remains `500`. With `run.threads = 4`, the
two configured pins (0, 1) are respected and the extra workers are
auto-pinned to the next unused CPUs (2 and 3).

The laptop default is preserved as an inline `# laptop default:` comment on
each changed line in `config/uenr.toml`, so reverting is one edit per key.

## Dataset bootstrap and cache

`scripts/bootstrap_datasets.sh` is pure bash plus the C converter (no
Python). On the first run with `bootstrap_datasets=true` it downloads
TinyStories (~1.92 GB) and TinyStoriesInstruct-valid (~27 MB), converts the
phase files with `./uenr convert`, and uploads four stable release assets:

| Tag | Asset | Used by |
|---|---|---|
| `dataset-pretrain` | `tinystories_train.txt` | pretrain, eval |
| `dataset-sft` | `sft_tinystories.jsonl` | SFT |
| `dataset-dpo` | `prefs_tinystories.jsonl` | DPO |
| `dataset-rl` | `rl_prompts.jsonl` | GRPO |

Later runs download those assets directly (`gh release download`) instead of
re-fetching from HuggingFace. Set `bootstrap_datasets=false` to require the
cache and fail rather than download. The existing `tiny-stories-dataset`
release is untouched.

## Checkpoints and outputs

- Stable per-phase release tags: `pretrain-checkpoint`, `sft-checkpoint`,
  `dpo-checkpoint`, `grpo-checkpoint`. Each holds the latest `<phase>.bin`
  plus `tokenizer.bin` with `--clobber`.
- Per-run workflow artifacts: `<phase>-checkpoint-run` (final phase file only)
  and the run summary table.
- Periodic `checkpoints/step_*.bin` files are **not** uploaded; they stay on
  the ephemeral runner.
- Pretrain resumes from `pretrain-checkpoint` with optimizer state (`--resume`).
  Other phases load the dependency checkpoint as weights; cross-run
  *optimizer* resume is implemented only for pretrain.

## Verification

The full pipeline was smoke-tested on `cloud-optimization` on 2026-11-18:

| Run | Mode | Result | Wall time | Notes |
|---|---|---|---|---|
| [44305055510](https://github.com/michaelowusuntim6/project_uenr/actions/runs/44305055510) | `smoke` | all 13 jobs green | 5m32s | cold dataset cache; pretrain 40 steps |
| [44306118735](https://github.com/michaelowusuntim6/project_uenr/actions/runs/44306118735) | `smoke` | all 13 jobs green | 7m58s | dataset assets served from releases; pretrain loaded step 40 and resumed to 60 |
| [44340226213](https://github.com/michaelowusuntim6/project_uenr/actions/runs/44340226213) | `bench` | green | 11m19s | earlier AVX1/AVX2 GEMM, pretrain, KV baseline |
| [44375533674](https://github.com/michaelowusuntim6/project_uenr/actions/runs/44375533674) | `bench` | green | 43m | batch/LR sweep, AVX-512 columns, post-Part-C numbers; `bench-results` artifact |
| [44383949635](https://github.com/michaelowusuntim6/project_uenr/actions/runs/44383949635) | `smoke` | 13/13 green | 7m19s | batch 64, AVX-512 auto, 7363 tok/s pretrain |

Those runs predate the method retirement below: the workflow now runs only
pretrain -> SFT -> DPO -> GRPO -> eval, so a current `smoke` dispatches fewer
jobs. Smoke step counts are 20 per phase by default. Local
validation on the same tree: `make test` 19/19 executables (including
`test_simd_dispatch`), `make smoke` pass, `make check-isa` clean with the
AVX1 objects AVX2/FMA-free and the AVX2 objects FMA-bearing.

## Known gaps (follow-up work)

1. **Non-pretrain optimizer resume:** SFT/DPO/GRPO load the previous phase
   checkpoint as weights, but their optimizer step counters are not resumed.
   Pretrain `--resume` restores optimizer state.
2. **Reward is heuristic only:** GRPO uses `[rl]` rule-based rewards; there
   is no learned reward model in the pipeline.
3. **Full-length validation:** only `mode=smoke` has been dispatched so far;
   the default step counts in the mode table are estimates until a full run.

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
| DPO | synthetic domain pairs derived from the SFT set (chosen story vs word-reversed rejected story) | `HuggingFaceH4/ultrafeedback_binarized` (canonical, parquet; convert offline) | https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized | ~50k pairs JSONL (a few tens of MB) | prompt/chosen/rejected JSONL |
| GRPO prompts | prompts extracted from the SFT instruction set | `mlabonne/smoltldr` (2,000 summary prompts, parquet; convert offline) | https://huggingface.co/datasets/mlabonne/smoltldr | ~5k prompts (a few MB) | prompt-only JSONL, completions sampled live |
| Eval | held-out pretrain split | held-out SFT/DPO slice | derived from pretrain | data-dependent | `UDAT` uint16 `val.bin` |

Justification: TinyStories and its instruct variant match the 8.45M model's
vocabulary and scale; the DPO/RL defaults are derived from that same domain
instead of out-of-domain Reddit summaries, and avoid parquet readers in CI.
The alternatives remain available through the offline conversion script.

## Workflow modes

| Mode | Jobs that run | Default steps | Measured/scaled wall time | Outputs |
|---|---|---|---|---|
| `pipeline` | pretrain → sft → dpo → grpo → eval, strictly sequential | 1600/1000/600/400 | ~130 min + prepare (smoke-scaled) | all four checkpoints |
| `pretrain` | restore → prepare → build → pretrain | 1600 (26.2M tokens) | ~63 min (smoke-scaled 47 s/20 steps) | `pretrain-checkpoint` |
| `sft` | prepare → sft (loads cached pretrain) | 1000 | ~55 min (smoke-scaled) | `sft-checkpoint` |
| `dpo` | prepare → dpo (loads cached sft, pretrain ref) | 600 | ~5 min (smoke-scaled) | `dpo-checkpoint` |
| `grpo` | prepare → grpo (loads cached dpo, pretrain ref) | 400 | ~8 min (smoke-scaled) | `grpo-checkpoint` |
| `eval` | prepare → eval (loads GRPO/DPO/SFT/pretrain) | n/a | < 5 min | run summary only |
| `smoke` | pipeline at tiny step counts | 10–20 each | verified 7m19s (cached data, batch 64) | all checkpoints |
| `bench` | build, diagnostic, prepare, GEMM/KV/AVX-512, batch/LR sweep, 100-step A/B | n/a | verified 43 min (run 44375533674) | run summary + `bench-results` artifact |

Only stable per-phase release tags are created.  Per-run history is kept as
workflow artifacts named `<phase>-checkpoint-run`.
