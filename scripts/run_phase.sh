#!/usr/bin/env bash
# Run one UENR training phase in CI, with dataset/checkpoint checks, stable
# per-phase release checkpoints, timeouts handled by the workflow, and a
# GitHub step-summary row.  Pure shell + the C binary; no Python.
#
# Usage: scripts/run_phase.sh <phase> <steps>
# Environment: GH_TOKEN (built-in GitHub token), GITHUB_REPOSITORY,
#              GITHUB_STEP_SUMMARY (optional)
set -euo pipefail

PHASE=${1:?phase}
STEPS=${2:-20}
REPO=${GITHUB_REPOSITORY:-michaelowusuntim6/project_uenr}
CFG=config/uenr.toml
BIN=./uenr
TAG="${PHASE}-checkpoint"
CKPT="checkpoints/${PHASE}.bin"
mkdir -p checkpoints data artifacts
START=$(date +%s)

summary_row() { # phase steps seconds tok_s loss checkpoint
    [ -n "${GITHUB_STEP_SUMMARY:-}" ] || return 0
    printf '| %s | %s | %s | %s | %s | `%s` |\n' "$1" "$2" "$3" "$4" "$5" "$6" \
        >> "$GITHUB_STEP_SUMMARY"
}

fail_data() { # name tag url
    echo "ERROR: required prepared data for '$PHASE' is missing ($1)." >&2
    echo "  release tag: $2" >&2
    echo "  source:      $3" >&2
    echo "  run: ./uenr prepare --phase $PHASE (after restore-datasets)" >&2
    exit 1
}

need_file() { [ -s "$1" ] || fail_data "$1" "$2" "$3"; }

# ---------------------------------------------------------------- checkpoints
# Stable per-phase release tags. Missing is fine on the first run.
dl() { # tag asset
    [ -n "${GH_TOKEN:-}" ] || return 0
    gh release download "$1" -R "$REPO" -p "$2" -D checkpoints --clobber \
        >/dev/null 2>&1 || true
}
[ "$PHASE" = "eval" ] || dl "$TAG" "$(basename "$CKPT")"
case "$PHASE" in
  sft) dl pretrain-checkpoint pretrain.bin ;;
  eval) dl pretrain-checkpoint pretrain.bin; dl sft-checkpoint sft.bin
        dl dpo-checkpoint dpo.bin; dl grpo-checkpoint grpo.bin ;;
  dpo) dl pretrain-checkpoint pretrain.bin; dl sft-checkpoint sft.bin ;;
  grpo) dl pretrain-checkpoint pretrain.bin; dl sft-checkpoint sft.bin
        dl dpo-checkpoint dpo.bin ;;
esac

# ---------------------------------------------------------------- data checks
case "$PHASE" in
  pretrain)
      need_file data/train.bin dataset-pretrain \
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
      need_file data/val.bin dataset-pretrain "derived from the pretrain split"
      ;;
  sft)
      need_file data/sft.jsonl dataset-sft \
        "https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-valid.txt"
      need_file checkpoints/pretrain.bin pretrain-checkpoint "pretrain phase"
      ;;
  dpo)
      need_file data/pairs.jsonl dataset-dpo \
        "derived from dataset-sft (or HuggingFaceH4/ultrafeedback_binarized)"
      need_file checkpoints/pretrain.bin pretrain-checkpoint "pretrain phase"
      need_file checkpoints/sft.bin sft-checkpoint "sft phase"
      ;;
  grpo)
      need_file data/rl_prompts.jsonl dataset-rl \
        "derived from dataset-sft (or mlabonne/smoltldr)"
      need_file checkpoints/pretrain.bin pretrain-checkpoint "pretrain phase"
      need_file checkpoints/dpo.bin dpo-checkpoint "dpo phase"
      ;;
  eval)
      need_file data/val.bin dataset-pretrain "pretrain validation split"
      need_file checkpoints/pretrain.bin pretrain-checkpoint "pretrain phase"
      ;;
esac

if [ "$PHASE" = "dpo" ]; then
    # E3: the frozen reference must be the pretrained checkpoint, never SFT.
    if [ -e checkpoints/sft.bin ] && cmp -s checkpoints/pretrain.bin checkpoints/sft.bin; then
        echo "ERROR: DPO reference is byte-identical to the SFT checkpoint; refusing." >&2
        exit 1
    fi
fi

# -------------------------------------------------------------------- command
cfg_value() { awk -v k="$1" '$1==k {gsub(/"/,"",$3); print $3; exit}' "$CFG"; }
CMD=""
RESUME=""
OUT=""
case "$PHASE" in
  pretrain)
      OUT=$(cfg_value model_out)
      # Resume only the pretrain phase (optimizer/step state is meaningful).
      if [ -s checkpoints/pretrain.bin ]; then
          ln -f checkpoints/pretrain.bin checkpoints/model.bin
          RESUME="--resume"
      fi
      CMD="$BIN train --steps $STEPS $RESUME --config $CFG"
      ;;
  sft)      OUT=$(cfg_value sft_out)
            CMD="$BIN sft --steps $STEPS --load checkpoints/pretrain.bin --config $CFG" ;;
  dpo)      OUT=$(cfg_value dpo_out)
            CMD="$BIN dpo --steps $STEPS --load checkpoints/sft.bin --ref checkpoints/pretrain.bin --config $CFG" ;;
  grpo)     OUT=$(cfg_value grpo_out)
            LOAD=checkpoints/sft.bin; [ -s checkpoints/dpo.bin ] && LOAD=checkpoints/dpo.bin
            CMD="$BIN grpo --steps $STEPS --load $LOAD --ref checkpoints/pretrain.bin --config $CFG" ;;
  eval)     OUT=""
            LOAD=checkpoints/pretrain.bin
            for p in grpo dpo sft; do
                [ -s "checkpoints/$p.bin" ] && { LOAD="checkpoints/$p.bin"; break; }
            done
            CMD="$BIN eval --load $LOAD --config $CFG --batches 4 --seq 256" ;;
  *) echo "unknown phase $PHASE"; exit 2 ;;
esac

# Optional cloud overrides shared by every phase (workflow_dispatch inputs).
if [ -n "${BATCH_SIZE:-}" ]; then CMD="$CMD --batch-size $BATCH_SIZE"; fi
if [ -n "${LR:-}" ]; then CMD="$CMD --lr $LR"; fi

echo "== phase=$PHASE steps=$STEPS"
echo "== $CMD"
LOG="phase-${PHASE}.log"
set +e
eval "$CMD" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
END=$(date +%s)
SECS=$((END - START))

TOKS=$(grep -oE '[0-9]+ tok/s' "$LOG" | tail -1 | awk '{print $1}' || true)
LOSS=$(grep -oE 'loss [0-9.]+' "$LOG" | tail -1 | awk '{print $2}' || true)
[ -n "$TOKS" ] || TOKS="-"
[ -n "$LOSS" ] || LOSS="-"

if [ "$RC" -ne 0 ]; then
    summary_row "$PHASE" "$STEPS" "${SECS}s" "$TOKS" "$LOSS" "failed"
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        echo "" >> "$GITHUB_STEP_SUMMARY"
        echo "**$PHASE failed** (exit $RC). See \`$LOG\`." >> "$GITHUB_STEP_SUMMARY"
    fi
    exit "$RC"
fi

# --------------------------------------------------------------- publish ckpt
if [ "$PHASE" != "eval" ]; then
    if [ -z "${OUT:-}" ] || [ ! -s "$OUT" ]; then
        echo "ERROR: $PHASE produced no checkpoint (expected $OUT)" >&2
        summary_row "$PHASE" "$STEPS" "${SECS}s" "$TOKS" "$LOSS" "no-checkpoint"
        exit 1
    fi
    [ "$OUT" = "$CKPT" ] || cp "$OUT" "$CKPT"
    if [ -n "${GH_TOKEN:-}" ]; then
        if ! gh release view "$TAG" -R "$REPO" >/dev/null 2>&1; then
            gh release create "$TAG" -R "$REPO" --prerelease \
                --title "$PHASE checkpoint (latest)" \
                --notes "Rolling latest $PHASE checkpoint produced by CI." >/dev/null
        fi
        gh release upload "$TAG" "$CKPT" -R "$REPO" --clobber >/dev/null
    fi
    summary_row "$PHASE" "$STEPS" "${SECS}s" "$TOKS" "$LOSS" "$TAG"
else
    summary_row "eval" "$STEPS" "${SECS}s" "$TOKS" "$LOSS" "n/a"
fi

if [ "$PHASE" = "eval" ]; then
    echo "== phase eval completed in ${SECS}s (model: ${LOAD:-checkpoints/pretrain.bin})"
else
    echo "== phase $PHASE completed in ${SECS}s (checkpoint: ${CKPT})"
fi
