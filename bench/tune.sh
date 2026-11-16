#!/usr/bin/env bash
# Try batch/grad_accum/seq combinations with a fixed 20-step budget.
set -euo pipefail
cd "$(dirname "$0")/.."
run() {
  local bs=$1 ga=$2 sq=$3 tag=$4
  local cfg=/tmp/uenr_tune_$tag.toml
  local model=/tmp/uenr_tune_$tag.bin
  sed -e "s|batch_size = [0-9]*|batch_size = $bs|" \
      -e "s|grad_accum = [0-9]*|grad_accum = $ga|" \
      -e "s|seq_len = [0-9]*|seq_len = $sq|" \
      -e "s|model_out = \"checkpoints/model.bin\"|model_out = \"$model\"|" \
      -e "s|checkpoint_every = [0-9]*|checkpoint_every = 0|" \
      -e "s|log_every = [0-9]*|log_every = 1|" \
      config/uenr.toml > "$cfg"
  rm -f "$model"
  local t0=$(date +%s.%N)
  local out
  out=$("./uenr" --config "$cfg" train --steps 20 2>&1 | grep 'step 20/' | tail -1)
  local t1=$(date +%s.%N)
  local tps
  tps=$(echo "$out" | sed -n 's/.*| \([0-9.]*\) tok\/s.*/\1/p')
  local loss
  loss=$(echo "$out" | sed -n 's/.*loss \([0-9.]*\).*/\1/p')
  printf 'bs=%-3s ga=%-2s seq=%-4s T=%-5s  %8s tok/s  loss=%s  wall=%.1fs\n' \
    "$bs" "$ga" "$sq" "$((bs/ga*sq))" "$tps" "$loss" "$(echo "$t1-$t0"|bc)"
}
run 8 2 256 A
run 4 1 256 B
run 8 4 256 C
run 8 2 512 D
run 16 4 256 E
run 8 2 128 F
