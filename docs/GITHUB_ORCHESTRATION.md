# GitHub orchestration, authentication and onboarding

This page covers the local tooling that drives the GitHub Actions trainer:
dependency installation, token storage, the interactive cloud menu, scripted
dispatches, and release-cache cleanup. It does not contain a real token.

## 1. One-time setup

```bash
python3 install.py --check       # report missing tools
python3 install.py --global      # install missing tools system-wide
scripts/github_auth.sh setup     # hidden prompt; validates via api.github.com
scripts/github_auth.sh status    # show user, token source and repo
```

`install.py` checks for `git`, `make`, a C compiler, `curl` and `gh`. Existing
tools are skipped. `--global` requires root or `sudo`; `--force` reinstalls or
upgrades even when a tool is present (and also requests a global install).
`--yes` makes it non-interactive; `--dry-run` prints the commands;
`--break-dependencies` is an alias for `--force`. Supported
package managers: apt, dnf/yum, pacman, zypper, apk and Homebrew. If the
distro has no `gh` package, the script downloads the official GitHub CLI
release binary for x86_64/arm64.

## 2. Token storage

The preferred path is a single gitignored file:

```
github_token.txt          # chmod 600, one token, no quotes
```

`github_token.txt.example` is committed as a template. Never commit the real
file; `.gitignore` also excludes `.github_token` and
`config/github_token.txt`.

Lookup order used by every orchestration script:

1. `$GH_TOKEN` or `$GITHUB_TOKEN`
2. `github_token.txt`
3. `.github_token`
4. `config/github_token.txt`
5. an existing `gh auth login` session

Create the token at <https://github.com/settings/tokens> with `repo` and
`workflow` scopes. `scripts/github_auth.sh setup` writes it with mode 0600
after validating it, and `logout` removes the local token files. The scripts
never print the token; `gh auth status` masks it.

The repository is detected from `origin`; override with
`UENR_GH_REPO=owner/name` or `GITHUB_REPOSITORY`. The default workflow ref is
`master`; override with `UENR_REF` or `--ref`.

## 3. Interactive launcher

`./uenr-menu` option **22. Cloud / GitHub trainer** opens:

| Choice | Action |
|---|---|
| 1 | Check authentication, user and repo |
| 2 | Set/update the token in `github_token.txt` |
| 3 | Monitor the latest run live (Ctrl+C stops watching) |
| 4 | Dispatch the full `pipeline` (pretrain -> SFT -> DPO -> GRPO -> eval, watched) |
| 5 | Dispatch `smoke` (all surviving phases, tiny) |
| 6 | Dispatch `bench` |
| 7 | Delete checkpoint releases, keep dataset caches |
| 8 | Delete all pre-releases, including dataset caches |
| 9 | Back to the main menu |

All actions call the scripts below, so the behaviour is identical whether
driven from the menu or a shell. Option 23 exits the main menu.

## 4. Dataset selection

The dataset recipes are part of `config/uenr.toml` (`[datasets.*]`). Both the
`restore-datasets` and `prepare-all` jobs call
`scripts/bootstrap_datasets.sh`, which reads the validated TOML through
`./uenr datasets`. A browser dispatch therefore uses the TOML of the selected
ref; changing `paths`, `release_tag`/`release_asset`, `url` or `derive_from`
does not require touching the workflow. Formats and examples:
[DATASETS.md](DATASETS.md).

## 5. Scripted dispatch

```bash
scripts/dispatch.sh pipeline --watch
scripts/dispatch.sh smoke
scripts/dispatch.sh bench --watch
scripts/dispatch.sh pretrain --steps 1600
scripts/dispatch.sh sft --steps 1000 --batch 64 --lr 0.00141
scripts/dispatch.sh dpo --steps 600
scripts/dispatch.sh grpo --steps 400
scripts/dispatch.sh eval
scripts/dispatch.sh --list
scripts/dispatch.sh --status
scripts/dispatch.sh smoke --dry-run
```

Modes are `pipeline`, `smoke`, `bench`, `pretrain`, `sft`, `dpo`, `grpo`,
`eval`. `--watch` uses `scripts/watch_run.sh`: plain line output every 10 seconds
(not a full-screen TUI), `Ctrl+C` stops watching immediately while the run
continues, and echo/mouse reporting are disabled so wheel or arrow escape
sequences cannot leak into the terminal. Use `--interval N` or
`UENR_WATCH_INTERVAL` to change the poll period. `scripts/watch_run.sh RUN_ID`
can monitor without dispatching.

`--steps` maps to the workflow input for a single phase
(`steps_pretrain`, `steps_sft`, ...); `pipeline` accepts `--steps` as the
per-phase value. `--batch` and `--lr` populate the shared workflow inputs;
`--no-bootstrap` disables dataset bootstrap; `-f` passes any other input.

Recommended first real run:

```bash
scripts/dispatch.sh pipeline --watch
```

At the tuned defaults this is 1,600 pretrain + 1,000 SFT + 600 DPO + 400 GRPO
steps (batch 64), roughly 2.5–3 hours including data preparation.

## 6. Release cleanup

The workflow uses stable release tags:

- dataset caches: `dataset-pretrain`, `dataset-sft`, `dataset-dpo`,
  `dataset-rl`
- checkpoints: `pretrain-checkpoint`, `sft-checkpoint`, `dpo-checkpoint`,
  `grpo-checkpoint` (each also carries `tokenizer.bin`)

```bash
scripts/cleanup_releases.sh --dry-run              # show checkpoint tags
scripts/cleanup_releases.sh --checkpoints --yes    # fresh training, keep data
scripts/cleanup_releases.sh --all-pre --yes        # also discard dataset caches
```

Checkpoint cleanup is the normal "start fresh" operation: the next dispatch
builds new ones, while the dataset release caches avoid re-downloading and
re-converting ~1.9 GB. Deleting the dataset caches adds roughly 10–25 minutes
to the next run. The `v0.1.0` release is never touched. Every deletion uses
`--cleanup-tag` and requires confirmation unless `--yes` is given on a TTY.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `error: GitHub authentication is not configured` | `scripts/github_auth.sh setup` or `gh auth login` |
| Token found but API rejects it | re-create with `repo` + `workflow` scopes; `github_auth.sh status` |
| `gh: command not found` | `python3 install.py --global --yes` |
| `sudo/root is unavailable` | run as root or drop `--global` (gh falls back to `~/.local/bin`) |
| workflow dispatch 403 | token needs `repo` and `workflow` |
| release deletion refused | non-interactive stdin requires `--yes` |
| wrong repository | `UENR_GH_REPO=owner/name` or fix `origin` |

## 8. What the workflow itself uses

The workflow uses the built-in `secrets.GITHUB_TOKEN` for uploading release
assets and artifacts. The local token file is only for laptop-side
`gh workflow run`, `gh run` and `gh release` operations; it is never copied
into the repository or workflow logs.
