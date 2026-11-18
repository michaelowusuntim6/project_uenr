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
2. Click **Run workflow**, pick branch `master` (or another branch) and the
   step count, then run.
3. Watch the live log. The final checkpoint and training CSV are uploaded as
   the `cloud-checkpoint` artifact and attached to a per-run release.

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

- Per-run release: `cloud-training-<run>-<attempt>` containing
  `checkpoints/model.bin` and `checkpoints/train.csv`.
- Workflow artifact: the same two files, name `cloud-checkpoint`.
- Periodic `checkpoints/step_*.bin` files are **not** uploaded; they stay on
  the ephemeral runner. Cross-run resume is not wired yet.

## Known gaps (follow-up work)

1. **Dataset re-download:** 1.92 GB from HuggingFace every run.
2. **No cross-run resume:** each dispatch starts from scratch. A full
   5,000-step run takes hours and can approach the 350-minute cap depending on
   runner throughput, so resume from the latest `cloud-checkpoint` artifact
   should be added before long runs.

## What this workflow does NOT do

- Does not touch `/home/mike/UENR/uenr` (laptop project).
- Does not pull large artifacts back to the laptop (slow link).
- Does not require any repository secrets (uses built-in `GITHUB_TOKEN`).
