# Handoff to OpenCode

State at handoff:

- Merge of `orchestration`: **8babb3b**; later docs/install follow-ups are
  on `master`. The branch can be deleted. `cloud-optimization` and
  `arch-optimization` were merged earlier and can be deleted too.
- Laptop tree `/home/mike/UENR/uenr` clean at `e4729f4` (untouched).
- Cloud working tree `/home/mike/UENR/uenr_github` on `master`, clean.
- `make test` green: **19/19 executables**. `make smoke` passes. `make
  check-isa` verifies AVX1 objects are AVX2/FMA-free, AVX2 objects contain
  FMA, and AVX-512 objects contain zmm instructions.
- Runtime SIMD: `[run] simd = "auto" | "avx1" | "avx2" | "avx512"`.
  `auto` selects AVX-512 on the runner, AVX2 on older AVX2 hosts, AVX1 on
  the laptop; forcing an unsupported level fails closed.
- Cloud defaults: batch 64, accum 1, LR 1.41e-3, pretrain 1600 steps
  (26.2M tokens). `config/uenr.laptop.toml` reverts to laptop values.
- Cloud training uses all four logical CPUs: `run.threads = 4` (main + 3
  workers, CPUs 0-3). The verified topology is 2 physical cores with SMT
  siblings `0-1` and `2-3`. A **two-thread** run must pin 0 and 2.
- Workflow: `.github/workflows/train.yml` on `master`, manual
  `workflow_dispatch` with `mode`, per-phase step counts, reward source and
  dataset bootstrap toggle. Modes: pretrain, sft, lora, dpo, grpo, ppo,
  train-rm, eval, full-core, full-rl, smoke, **bench**.
- Actions UI:
  https://github.com/michaelowusuntim6/project_uenr/actions/workflows/train.yml

## Verified pipeline (2026-11-18)

Smoke (all 13 jobs green):

- [44305055510](https://github.com/michaelowusuntim6/project_uenr/actions/runs/44305055510)
  — 5m32s, cold dataset bootstrap.
- [44306118735](https://github.com/michaelowusuntim6/project_uenr/actions/runs/44306118735)
  — 7m58s, dataset assets from release tags, pretrain resumed step 40 -> 60.
- Tuned batch-64/AVX-512 smoke: run
  [44383949635](https://github.com/michaelowusuntim6/project_uenr/actions/runs/44383949635)
  — 13/13 green, 7m19s, pretrain 7363 tok/s.
- Bench: [44375533674](https://github.com/michaelowusuntim6/project_uenr/actions/runs/44375533674)
  — 43m, batch/LR sweep, AVX-512 and post-saturation numbers.

SIMD/throughput (runs 44375533674 and 44383949635):

- 384 GEMM 4t: AVX2 195.15, AVX-512 229.38 GFLOP/s (1.18x).
- 1024 GEMM 4t: AVX2 107.32, AVX-512 141.45 GFLOP/s (1.32x).
- 100-step pretrain 4t: AVX1 4164, AVX2 7463, AVX-512 7887 tok/s.
- Locked batch-64 AVX-512 smoke: 7363 tok/s (about 6.1x the 1.2k laptop
  baseline).
- Batch sweep: 16/2 3.484, 32/1 3.295, 64/1 3.142, 64/1@1.41e-3 3.063
  val loss at 200 steps. Batch 64 + 1.41e-3 wins; throughput is 6668 tok/s.
- Details: `docs/ARCH_OPTIMIZATION.md` §§9–11.

Stable release assets: `dataset-pretrain`, `dataset-sft`, `dataset-dpo`,
`dataset-rl`; checkpoint tags `pretrain-checkpoint`, `sft-checkpoint`,
`lora-checkpoint`, `train-rm-checkpoint`, `dpo-checkpoint`,
`grpo-checkpoint`, `ppo-checkpoint`. `tiny-stories-dataset` was not modified.

## Local GitHub orchestration

- `./uenr-menu` option 24 is a Cloud / GitHub submenu; option 25 exits.
- `install.py` installs missing global dependencies; `scripts/github_auth.sh
  setup` stores a validated token in gitignored `github_token.txt`.
- `scripts/dispatch.sh <mode>` wraps every workflow mode with
  steps/batch/LR/ref/watch/list/dry-run.
- `scripts/cleanup_releases.sh` deletes checkpoint tags by default and keeps
  dataset caches; `--all-pre` includes them; v0.1.0 is protected.
- `scripts/github_auth.sh setup` writes gitignored `github_token.txt` (0600)
  after validation; `install.py` installs missing global tools.
- Full guide: `docs/GITHUB_ORCHESTRATION.md`.

## Remaining work

1. **Dispatch `mode=full-core` manually** — the first real training run
   (1600 pretrain + 1000 SFT + 1000 LoRA, ~2.5–3 h including prepare at the
   tuned batch-64/AVX-512 defaults).
2. **Dispatch `mode=full-rl`** after full-core succeeds (~20 min plus
   prepare; loads the cached SFT checkpoint).
3. **Verify the checkpoints** from those runs: `pretrain-checkpoint`,
   `sft-checkpoint`, `dpo-checkpoint` should contain trained models (val
   loss below init; SFT loss dropped by at least 1.0).

The pipeline is tuned and saturated. Nothing else remains before the first
real training run.

## Constraints for OpenCode

- Do NOT modify `/home/mike/UENR/uenr` (laptop).
- Do NOT pull datasets, prepared data, or checkpoints to the laptop
  (slow link; all heavy I/O stays on GitHub).
- Do NOT launch a long training run during development; use `mode=smoke`, a
  100-step pretrain, or `mode=bench`.
- Work on a branch off `master`; merge only after `make test` is green and a
  `mode=smoke` dispatch succeeds. Do not create per-run releases.
- Keep the AVX1 objects free of AVX2/FMA (`make check-isa` enforces this);
  new hot kernels need both paths or a runtime-gated AVX2 object.

See also: `docs/ARCH_OPTIMIZATION.md` and `docs/CLOUD_TRAINING.md`.
