# Handoff to OpenCode

State at handoff:

- `master` tip: **81301fe** (merge **a2176d6** of `arch-optimization`, then a
  `gemmbench` dispatch follow-up). The documented SIMD work is in the merge.
- `arch-optimization` is merged and can be deleted.
- Laptop tree `/home/mike/UENR/uenr` clean at `e4729f4` (untouched).
- Cloud working tree `/home/mike/UENR/uenr_github` on `master`, clean.
- `make test` green: **19/19 executables**, including
  `tests/test_simd_dispatch.c`. `make smoke` passes. `make check-isa` verifies
  the AVX1 objects are AVX2/FMA-free and the AVX2 objects contain FMA.
- Runtime SIMD: `[run] simd = "auto" | "avx1" | "avx2"` (`--simd` override).
  The laptop runs AVX1+F16C; the runner selects AVX2+FMA. Both paths are
  linked; forcing AVX2 on an unsupported CPU fails closed.
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
- Post-merge smoke on `master`: run
  [44344021516](https://github.com/michaelowusuntim6/project_uenr/actions/runs/44344021516).

SIMD benchmark (`mode=bench`, run
[44340226213](https://github.com/michaelowusuntim6/project_uenr/actions/runs/44340226213),
11m19s):

- 384 GEMM: AVX1 136-138 GFLOP/s, AVX2+FMA 188-193 GFLOP/s (1.36-1.42x).
- 1024 GEMM: AVX1 43.9-52.5, AVX2 55.8-80.0 (1.27-1.52x).
- 100-step pretrain: AVX1 2642/4108 tok/s (2/4 threads), AVX2 4140/6506
  (1.57x/1.58x). Four threads wins.
- INT8 KV decode at 512: 2017 -> 2871 tok/s (1.42x); Q4: 2141 -> 2577
  (1.20x).
- Details, corrected assumptions and the kernel audit are in
  `docs/ARCH_OPTIMIZATION.md`.

Stable release assets: `dataset-pretrain`, `dataset-sft`, `dataset-dpo`,
`dataset-rl`; checkpoint tags `pretrain-checkpoint`, `sft-checkpoint`,
`lora-checkpoint`, `train-rm-checkpoint`, `dpo-checkpoint`,
`grpo-checkpoint`, `ppo-checkpoint`. `tiny-stories-dataset` was not modified.

## Remaining gaps (follow-up work)

1. **Full-length runs:** only `smoke` and the short `bench` pretrain have been
   dispatched. Run `full-core`, then `full-rl`, to validate the default step
   counts and wall-time estimates in `docs/CLOUD_TRAINING.md`.
2. **Non-pretrain optimizer resume:** SFT/LoRA/DPO/GRPO/PPO load dependency
   checkpoints as weights but do not restore optimizer step counters.
   Pretrain `--resume` restores optimizer state.
3. **AVX-512:** the observed hosts advertise AVX-512 but the project
   deliberately targets AVX2+FMA for portability. A future AVX-512 path would
   need its own runtime tier and benchmark.
4. **Branch cleanup:** delete `arch-optimization` and `cloud-optimization`
   from `origin` once no further changes are pending.

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
