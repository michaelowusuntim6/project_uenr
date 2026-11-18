# Handoff to OpenCode

State at handoff:

- `master` SHA: **4d2ae46** (merge of `cloud-optimization`; this handoff
  update follows as a docs-only commit on `master`).
- `cloud-optimization` is merged and can be deleted.
- Laptop tree `/home/mike/UENR/uenr` clean at `e4729f4` (untouched).
- Cloud working tree `/home/mike/UENR/uenr_github` on `master`.
- `make test` green: **18/18 executables**; `make smoke` passes;
  `make check-isa` clean on the Ivy Bridge default build.
- Cloud training uses all four vCPUs: `run.threads = 4` (main + 3 workers),
  pins 0/1 respected and workers auto-pinned to CPUs 2/3. The laptop profile
  remains two threads.
- Workflow: `.github/workflows/train.yml` on `master`, manual
  `workflow_dispatch` with `mode`, per-phase step counts, reward source and
  dataset bootstrap toggle. Modes: pretrain, sft, lora, dpo, grpo, ppo,
  train-rm, eval, full-core, full-rl, smoke.
- Actions UI:
  https://github.com/michaelowusuntim6/project_uenr/actions/workflows/train.yml

## Verified cloud pipeline (2026-11-18)

Two `mode=smoke` dispatches passed all 13 jobs on `cloud-optimization`:

- [44305055510](https://github.com/michaelowusuntim6/project_uenr/actions/runs/44305055510)
  — 5m32s, cold dataset bootstrap, pretrain 20 steps.
- [44306118735](https://github.com/michaelowusuntim6/project_uenr/actions/runs/44306118735)
  — 7m58s, dataset tags served from releases, pretrain resumed from step 40
  to 60. SFT/LoRA/DPO/GRPO/PPO/train-rm/eval all succeeded and published
  stable checkpoint tags.

Stable release assets now exist: `dataset-pretrain`, `dataset-sft`,
`dataset-dpo`, `dataset-rl`; checkpoint tags `pretrain-checkpoint`,
`sft-checkpoint`, `lora-checkpoint`, `train-rm-checkpoint`,
`dpo-checkpoint`, `grpo-checkpoint`, `ppo-checkpoint`. The pre-existing
`tiny-stories-dataset` release was not modified.

## Remaining gaps (follow-up work)

1. **Full-length runs:** only `mode=smoke` has been dispatched. The default
   step counts and wall-time estimates in `docs/CLOUD_TRAINING.md` are not yet
   measured; run `full-core`, then `full-rl`, to validate them.
2. **Non-pretrain optimizer resume:** SFT/LoRA/DPO/GRPO/PPO load dependency
   checkpoints as weights but do not restore optimizer step counters.
   Pretrain `--resume` restores optimizer state.
3. **Branch cleanup:** delete `cloud-optimization` from `origin` once no
   further cloud changes are pending.

## Constraints for OpenCode

- Do NOT modify `/home/mike/UENR/uenr` (laptop).
- Do NOT pull datasets, prepared data, or checkpoints to the laptop
  (slow link; all heavy I/O stays on GitHub).
- Do NOT launch a long training run during development; use a 100-step smoke
  test.
- Work on a branch off `master`; merge only after `make test` is green and a
  `mode=smoke` dispatch succeeds. Do not create per-run releases.

See also: `docs/CLOUD_TRAINING.md` (current state) and the OpenCode prompt
(two gaps).
