#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run() {
  local tag=$1 lr=$2 warm=$3
  local cfg=/tmp/uenr_sweep_$tag.toml
  local model=/tmp/uenr_sweep_$tag.bin
  sed -e "s|^lr = .*|lr = $lr|" \
      -e "s|^warmup_steps = .*|warmup_steps = $warm|" \
      -e "s|model_out = \"checkpoints/model.bin\"|model_out = \"$model\"|" \
      -e "s|checkpoint_every = [0-9]*|checkpoint_every = 0|" \
      -e "s|log_every = [0-9]*|log_every = 1000|" \
      config/uenr.toml > "$cfg"
  rm -f "$model"
  ./uenr --config "$cfg" train --steps 50 >/dev/null 2>&1
  local line
  line=$(./uenr --config "$cfg" eval --model "$model" --batches 16 --seq 256 2>&1 | grep '\[eval\] data')
  printf 'lr=%-8s warmup=%-4s  %s\n' "$lr" "$warm" "$line"
}
run A 5e-4 20
run B 3e-4 20
run C 1e-3 20
run D 5e-4 50
