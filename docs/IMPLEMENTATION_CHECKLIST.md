# Implementation checklist — GitHub orchestration and onboarding

Status keys: `[ ]` pending, `[x]` done, `[~]` documented/partial.

## 1. Release cleanup

- [x] Audited GitHub releases (7 checkpoint tags + 4 dataset caches + v0.1.0).
- [x] Deleted the 7 checkpoint releases only; dataset caches and v0.1.0 kept.
- [x] Added `scripts/cleanup_releases.sh` and menu actions.

## 2. GitHub authentication

- [x] Token loader reads `github_token.txt` (gitignored), `.github_token`, or
      `config/github_token.txt`, with `GH_TOKEN`/`GITHUB_TOKEN` fallback.
- [x] Falls back to `gh auth status` when no token file exists.
- [x] Interactive `github_auth.sh` setup/check/logout with hidden input and
      `chmod 600`.
- [x] Committed `github_token.txt.example`; real token files ignored.
- [x] All dispatch/cleanup/menu commands use the loader and a detected repo.

## 3. Cloud dispatch

- [x] `scripts/dispatch.sh` supports mode, steps, batch, LR, ref, watch/list.
- [x] `scripts/cleanup_releases.sh` supports checkpoint/all-pre, dry-run, yes.
- [x] Scripts are warning-free under `bash -n`.

## 4. Interactive orchestrator

- [x] Clean "Cloud / GitHub" submenu in `uenr-menu`.
- [x] Actions: auth, status/runs, smoke, bench, pretrain, SFT, full-core,
      full-rl, custom dispatch, checkpoint cleanup, back.
- [x] Menu self-test updated; no interactive regressions.

## 5. install.py

- [x] Stdlib-only, idempotent, skips installed tools.
- [x] `--check`, `--yes`, `--global`, `--force`, `--dry-run`; distro package-manager map.
- [x] gh fallback binary install; sudo handling; clear diagnostics.
- [x] Executable bit and docs/quickstart entry.

## 6. README / docs

- [x] Removed machine-specific `/home/...` paths and benchmark laundry list from README cloud section.
- [x] Added production-ready cloud/onboarding summary.
- [x] Updated CLOUD_TRAINING, HANDOFF, CHANGELOG, docs index, BUILD.
- [x] Added `docs/GITHUB_ORCHESTRATION.md`.

## 7. Verification

- [x] `python3 -m py_compile install.py`; `./install.py --check` (0).
- [x] `bash -n` on every script; `make -j4`, `make test` (19/19), `make check-isa`.
- [x] `make smoke` passes.
- [x] Menu piped self-test (local + cloud auth).
- [x] Commit groups prepared; merge/push in final step.
