# Handoff to OpenCode

State at handoff:

- `master` SHA: **54c08ef** (last content commit before this handoff file;
  the handoff commit is the tip containing this document).
- `github-training` branch has been deleted from `origin` and locally; all
  cloud work is consolidated on `master`.
- Laptop tree `/home/mike/UENR/uenr` clean at `e4729f4` (untouched).
- Cloud working tree `/home/mike/UENR/uenr_github` on `master`.
- `make test` green: **17/17 executables**; `make check-isa` clean on the
  Ivy Bridge default build.
- Cloud training uses all four vCPUs: `run.threads = 4` (main + 3 workers),
  pins 0/1 respected and workers auto-pinned to CPUs 2/3. The laptop profile
  remains two threads.
- Workflow: `.github/workflows/train.yml` on `master`, manual
  `workflow_dispatch` with `branch` and `steps` inputs.
- Actions UI:
  https://github.com/michaelowusuntim6/project_uenr/actions/workflows/train.yml

## Two gaps to close (in order)

1. **Dataset mirror:** single 1.92 GB upload to release tag
   `tiny-stories-dataset`; subsequent runs should use `gh release download`
   instead of re-downloading from HuggingFace.
2. **Cross-run resume:** upload a `cloud-checkpoint` artifact with
   `if: always()`, and download it at the start of the next run to resume via
   `./uenr train --resume`.

## Constraints for OpenCode

- Do NOT modify `/home/mike/UENR/uenr` (laptop).
- Do NOT pull datasets, prepared data, or checkpoints to the laptop
  (slow link; all heavy I/O stays on GitHub).
- Do NOT launch a long training run during development; use a 100-step smoke
  test.
- Push commits to a new branch `cloud-optimization` off `master`, then open a
  PR or merge to `master` once `make test` is green.

See also: `docs/CLOUD_TRAINING.md` (current state) and the OpenCode prompt
(two gaps).
