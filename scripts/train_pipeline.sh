#!/usr/bin/env bash
# Linear UENR training pipeline:
#   pretrain -> SFT -> DPO -> GRPO -> eval
# Each phase consumes the previous phase's checkpoint. Stopping on failure is
# intentional: a phase must not train on a stale or missing predecessor.
#
# Usage:
#   scripts/train_pipeline.sh                # cloud defaults (1600/1000/600/400)
#   scripts/train_pipeline.sh --short        # 4/4/2/2, for local verification
#   scripts/train_pipeline.sh --steps 20     # 20 steps for every phase
#   scripts/train_pipeline.sh --pretrain 20 --sft 10 --dpo 5 --grpo 5
#   scripts/train_pipeline.sh --resume       # resume pretrain from checkpoint
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
BIN=${UENR_BIN:-./uenr}
CFG=${UENR_CONFIG:-config/uenr.toml}

P=1600; S=1000; D=600; G=400
RESUME=""
while [ $# -gt 0 ]; do
    case "$1" in
        --short) P=4; S=4; D=2; G=2; shift ;;
        --steps) P=${2:?}; S=$P; D=$P; G=$P; shift 2 ;;
        --pretrain) P=${2:?}; shift 2 ;;
        --sft) S=${2:?}; shift 2 ;;
        --dpo) D=${2:?}; shift 2 ;;
        --grpo) G=${2:?}; shift 2 ;;
        --resume) RESUME="--resume"; shift ;;
        --config) CFG=${2:?}; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

need() { [ -s "$1" ] || { echo "pipeline: missing $1" >&2; exit 1; }; }
run_phase() { # label command...
    echo ""
    echo "======== $1 ========"
    shift
    "$@"
}

mkdir -p checkpoints
cfg_value() { awk -v k="$1" '$1==k {gsub(/"/,"",$3); print $3; exit}' "$CFG"; }
MODEL=$(cfg_value model_out); MODEL=${MODEL:-checkpoints/model.bin}
SFT=$(cfg_value sft_out);     SFT=${SFT:-checkpoints/sft.bin}
DPO=$(cfg_value dpo_out);     DPO=${DPO:-checkpoints/dpo.bin}
GRPO=$(cfg_value grpo_out);   GRPO=${GRPO:-checkpoints/grpo.bin}
PREF=$(cfg_value model_out);  PREF=${PREF:-checkpoints/model.bin}
run_phase "pretrain ($P steps)" "$BIN" --config "$CFG" train --steps "$P" $RESUME
need "$MODEL"
run_phase "SFT ($S steps, from pretrain)" "$BIN" --config "$CFG" sft --steps "$S" --load "$MODEL"
need "$SFT"
run_phase "DPO ($D steps, from SFT; ref=pretrain)" \
    "$BIN" --config "$CFG" dpo --steps "$D" --load "$SFT" --ref "$PREF"
need "$DPO"
run_phase "GRPO ($G steps, from DPO; ref=pretrain)" \
    "$BIN" --config "$CFG" grpo --steps "$G" --load "$DPO" --ref "$PREF"
need "$GRPO"
run_phase "eval (GRPO checkpoint)" "$BIN" --config "$CFG" eval --load "$GRPO" --batches 4 --seq 256

echo ""
echo "pipeline complete: $MODEL -> $SFT -> $DPO -> $GRPO"
