#!/usr/bin/env bash
# UENR measurement harness. One command per metric; same scratch config always.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=/tmp/uenr_bench.toml
MODEL=/tmp/uenr_bench_model.bin
mkdir -p /tmp/uenr_bench_ckpt

write_cfg() {
  sed -e "s|model_out = \"checkpoints/model.bin\"|model_out = \"$MODEL\"|" \
      -e "s|checkpoint_dir = \"checkpoints\"|checkpoint_dir = \"/tmp/uenr_bench_ckpt\"|" \
      -e "s|checkpoint_every = [0-9]*|checkpoint_every = 0|" \
      -e "s|log_every = [0-9]*|log_every = 1|" \
      config/uenr.toml > "$CFG"
}

case "${1:-all}" in
train)
  N=${2:-20}
  write_cfg
  rm -f "$MODEL"
  "$PWD/uenr" --config "$CFG" train --steps "$N" 2>&1 | tee /tmp/uenr_measure_train.log >/dev/null
  grep -E '^\[pretrain\] step' /tmp/uenr_measure_train.log | tail -1
  grep -E 'model: .*params' /tmp/uenr_measure_train.log | tail -1
  ;;
mem)
  write_cfg
  rm -f "$MODEL"
  /usr/bin/time -v "$PWD/uenr" --config "$CFG" train --steps 1 \
      > /tmp/uenr_measure_mem.log 2>&1 || true
  grep -E 'Maximum resident set size' /tmp/uenr_measure_mem.log
  ;;
infer)
  N=${2:-20}
  "$PWD/uenr" infer --prompt "Once upon a time" --tokens "$N" --greedy \
      2>&1 | tee /tmp/uenr_measure_infer.log | tail -1
  ;;
gemm)
  N=${2:-256}
  "$PWD/uenr" bench --size "$N" 2>&1 | tee /tmp/uenr_measure_gemm.log
  ;;
tests)
  make test 2>&1 | tee /tmp/uenr_measure_tests.log | grep -E '^==|all passed|config ok|FAIL'
  ;;
all)
  "$0" tests
  "$0" gemm 256
  "$0" infer 20
  "$0" train 20
  "$0" mem
  ;;
*)
  echo "usage: $0 {tests|gemm [n]|infer [n]|train [steps]|mem|all}" >&2
  exit 2
  ;;
esac
