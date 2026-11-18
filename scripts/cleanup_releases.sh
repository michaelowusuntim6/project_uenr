#!/usr/bin/env bash
# Delete UENR checkpoint or dataset release caches. The v0.1.0 release is
# never touched. Default is checkpoint tags only; dataset caches are kept so
# the next workflow run does not re-download/convert ~1.9 GB.
#
# Usage:
#   scripts/cleanup_releases.sh [--checkpoints] [--all-pre] [--dry-run] [--yes]
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=gh_env.sh
. "$HERE/gh_env.sh"

SCOPE=checkpoints
DRY=0
YES=0
while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoints) SCOPE=checkpoints; shift ;;
        --all-pre) SCOPE=all-pre; shift ;;
        --dry-run) DRY=1; shift ;;
        --yes|-y) YES=1; shift ;;
        -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown option $1" >&2; exit 2 ;;
    esac
done

uenr_load_gh_token || exit 1
EXISTING=$(gh release list -R "$UENR_GH_REPO" --limit 200 --json tagName \
           --jq '.[].tagName')
TAGS=()
if [ "$SCOPE" = checkpoints ]; then
    for t in pretrain-checkpoint sft-checkpoint lora-checkpoint \
             train-rm-checkpoint dpo-checkpoint grpo-checkpoint ppo-checkpoint; do
        if printf '%s\n' "$EXISTING" | grep -qx "$t"; then TAGS+=("$t"); fi
    done
else
    while IFS= read -r t; do
        [ -n "$t" ] && TAGS+=("$t")
    done < <(gh release list -R "$UENR_GH_REPO" --limit 200 \
             --json tagName,isPrerelease \
             --jq '.[] | select(.isPrerelease) | .tagName')
fi

if [ ${#TAGS[@]} -eq 0 ]; then
    echo "no $SCOPE releases found"
    exit 0
fi
echo "repository: $UENR_GH_REPO"
echo "scope:      $SCOPE"
printf 'delete:     %s\n' "${TAGS[*]}"
if [ "$DRY" = 1 ]; then
    echo "dry run; nothing deleted"
    exit 0
fi
if [ "$YES" != 1 ]; then
    if [ ! -t 0 ]; then
        echo "error: refusing to delete without --yes on a non-interactive stdin" >&2
        exit 1
    fi
    printf "Delete these %d release(s)? [y/N] " "${#TAGS[@]}"
    read -r ans
    case "$ans" in y|Y|yes|YES) ;; *) echo "aborted"; exit 1 ;; esac
fi
for t in "${TAGS[@]}"; do
    echo "deleting $t"
    gh release delete "$t" -R "$UENR_GH_REPO" --cleanup-tag --yes
done
echo "deleted ${#TAGS[@]} release(s); dataset caches are untouched only in --checkpoints mode"
