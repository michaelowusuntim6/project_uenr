# Implementation checklist — GitHub orchestration and onboarding

Status keys: `[ ]` pending, `[x]` done, `[~]` documented/partial.

## 1. Release cleanup

- [x] Audited GitHub releases (7 checkpoint tags + 4 dataset caches + v0.1.0).
- [x] Deleted the 7 checkpoint releases only; dataset caches and v0.1.0 kept.
- [ ] Added a repeatable cleanup script and interactive menu action.

## 2. GitHub authentication

- [ ] Token loader reads `github_token.txt` (gitignored), `.github_token`, or
      `config/github_token.txt`, with `GH_TOKEN`/`GITHUB_TOKEN` fallback.
- [ ] Falls back to `gh auth status` when no token file exists.
- [ ] Interactive `github_auth.sh` setup/check/logout with hidden input and
      `chmod 600`.
- [ ] Committed `github_token.txt.example`; real token files ignored.
- [ ] All dispatch/cleanup/menu commands use the loader and a detected repo.

## 3. Cloud dispatch

- [ ] `scripts/dispatch.sh` supports mode, steps, batch, LR, ref, watch/list.
- [ ] `scripts/cleanup_releases.sh` supports checkpoint/all-pre, dry-run, yes.
- [ ] Scripts are warning-free under `bash -n` and shellcheck-style quoting.

## 4. Interactive orchestrator

- [ ] Clean "Cloud / GitHub" submenu in `uenr-menu`.
- [ ] Actions: auth, status/runs, smoke, bench, pretrain, SFT, full-core,
      full-rl, custom dispatch, checkpoint cleanup, back.
- [ ] Menu self-test updated; no interactive regressions.

## 5. install.py

- [ ] Stdlib-only, idempotent, skips installed tools.
- [ ] `--check`, `--yes`, `--global`, `--force`; distro package-manager map.
- [ ] gh fallback binary install; sudo handling; clear diagnostics.
- [ ] Executable bit and docs/quickstart entry.

## 6. README / docs

- [ ] Remove machine-specific `/home/...` paths and benchmark laundry list
      from README cloud section.
- [ ] Add production-ready cloud/onboarding summary.
- [ ] Update CLOUD_TRAINING, HANDOFF, CHANGELOG, docs index, TECH docs.
- [ ] Add `docs/GITHUB_ORCHESTRATION.md`.

## 7. Verification

- [ ] `python3 -m py_compile install.py`; `./install.py --check`.
- [ ] `bash -n` on every script; `make -j4`, `make test`, `make check-isa`.
- [ ] `make smoke`.
- [ ] Menu piped self-test.
- [ ] Commit groups, merge to `master`, push; laptop tree untouched.
