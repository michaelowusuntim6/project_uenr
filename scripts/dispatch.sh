#!/usr/bin/env bash
# Dispatch the UENR GitHub Actions training workflow.
#
# Usage:
#   scripts/dispatch.sh <mode> [options] [-f key=value ...]
#   scripts/dispatch.sh --list
#   scripts/dispatch.sh --status
#   scripts/dispatch.sh --watch <mode> [options]
#
# Modes: pipeline smoke bench pretrain sft dpo grpo eval
#
# Options:
#   --steps N       steps input for the selected single phase
#   --batch N       override batch_size (workflow input)
#   --lr F          override learning rate (workflow input)
#   --ref BRANCH    workflow ref (default: master, or $UENR_REF)
#   --no-bootstrap  set bootstrap_datasets=false
#   --dry-run       print the gh command without dispatching
#   --list          list recent workflow runs
#   --status        show the latest run
#   --watch         dispatch, then watch the newest run
#   --interval N    watch polling interval in seconds (default 10)
#   -h, --help      this help
# Any extra `-f key=value` is forwarded to `gh workflow run`.
# Requires an authenticated gh; see scripts/github_auth.sh setup.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=gh_env.sh
. "$HERE/gh_env.sh"

MODE=""
STEPS=""
BATCH=""
LR=""
REF=${UENR_REF:-master}
BOOTSTRAP=true
DRY=0
WATCH=0
INTERVAL=${UENR_WATCH_INTERVAL:-10}
LIST=0
STATUS=0
EXTRA=()

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --steps) STEPS=${2:?--steps needs a value}; shift 2 ;;
        --batch) BATCH=${2:?--batch needs a value}; shift 2 ;;
        --lr)    LR=${2:?--lr needs a value}; shift 2 ;;
        --ref)   REF=${2:?--ref needs a value}; shift 2 ;;
        --no-bootstrap) BOOTSTRAP=false; shift ;;
        --dry-run) DRY=1; shift ;;
        --watch) WATCH=1; shift ;;
        --interval) INTERVAL=${2:?--interval needs a value}; shift 2 ;;
        --list) LIST=1; shift ;;
        --status) STATUS=1; shift ;;
        -f) EXTRA+=(-f "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "error: unknown option $1" >&2; usage >&2; exit 2 ;;
        *) [ -z "$MODE" ] && MODE=$1 || { echo "error: extra argument $1" >&2; exit 2; }; shift ;;
    esac
done

if [ "$LIST" = 1 ]; then
    uenr_gh run list --workflow=train.yml --limit 15
    exit $?
fi
if [ "$STATUS" = 1 ]; then
    uenr_gh run list --workflow=train.yml --limit 1
    exit $?
fi
[ -n "$MODE" ] || { usage >&2; exit 2; }

case "$MODE" in
    pipeline|smoke|bench|pretrain|sft|dpo|grpo|eval) ;;
    *) echo "error: unknown mode '$MODE'" >&2; exit 2 ;;
esac

step_key=""
case "$MODE" in
    pretrain) step_key=steps_pretrain ;;
    sft)      step_key=steps_sft ;;
    dpo)      step_key=steps_dpo ;;
    grpo)     step_key=steps_grpo ;;
esac
if [ -n "$STEPS" ] && [ -z "$step_key" ]; then
    echo "error: --steps is not defined for mode '$MODE'" >&2
    echo "       use -f key=value for pipeline phase step inputs" >&2
    exit 2
fi

ARGS=(-R "$UENR_GH_REPO" --ref "$REF" -f "mode=$MODE")
[ -n "$step_key" ] && [ -n "$STEPS" ] && ARGS+=(-f "$step_key=$STEPS")
[ -n "$BATCH" ] && ARGS+=(-f "batch_size=$BATCH")
[ -n "$LR" ] && ARGS+=(-f "lr=$LR")
ARGS+=(-f "bootstrap_datasets=$BOOTSTRAP")
ARGS+=("${EXTRA[@]}")

if [ "$DRY" = 1 ]; then
    printf 'gh workflow run train.yml'; printf ' %q' "${ARGS[@]}"; printf '\n'
    exit 0
fi
uenr_load_gh_token || exit 1
BEFORE_ID=$(gh run list -R "$UENR_GH_REPO" --workflow=train.yml --limit 1 \
            --json databaseId --jq '.[0].databaseId // empty' 2>/dev/null || true)
echo "dispatching mode=$MODE ref=$REF repo=$UENR_GH_REPO"
gh workflow run train.yml "${ARGS[@]}"

if [ "$WATCH" = 1 ]; then
    RUN_ID=""
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        sleep 1
        RUN_ID=$(gh run list -R "$UENR_GH_REPO" --workflow=train.yml --limit 1 \
                 --json databaseId --jq '.[0].databaseId // empty')
        [ -n "$RUN_ID" ] && [ "$RUN_ID" != "$BEFORE_ID" ] && break
    done
    if [ -n "$RUN_ID" ] && [ "$RUN_ID" != "null" ] && [ "$RUN_ID" != "$BEFORE_ID" ]; then
        echo "watching run $RUN_ID"
        "$HERE/watch_run.sh" "$RUN_ID" --interval "$INTERVAL"
    else
        echo "warning: could not find the dispatched run; check 'scripts/dispatch.sh --list'" >&2
    fi
else
    echo "run 'scripts/dispatch.sh --list' to find it, or use --watch next time."
fi
