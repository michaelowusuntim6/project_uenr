#!/usr/bin/env bash
# Dispatch the UENR cloud workflow with the same defaults as the workflow UI.
# Usage: scripts/dispatch.sh <mode> [extra gh -f inputs...]
#   scripts/dispatch.sh smoke
#   scripts/dispatch.sh bench
#   scripts/dispatch.sh full-core -f steps_pretrain=5000 -f batch_size=32 -f lr=0.001
# Requires an authenticated `gh` (repo scope). Never runs locally.
set -euo pipefail
MODE=${1:?mode: smoke|bench|pretrain|sft|lora|dpo|grpo|ppo|train-rm|eval|full-core|full-rl}
shift || true
REPO=${GITHUB_REPOSITORY:-michaelowusuntim6/project_uenr}
REF=${UENR_REF:-master}
ARGS=(-R "$REPO" --ref "$REF" -f "mode=$MODE")
case "$MODE" in
  smoke|bench) ARGS+=(-f bootstrap_datasets=true) ;;
esac
exec gh workflow run train.yml "${ARGS[@]}" "$@"
