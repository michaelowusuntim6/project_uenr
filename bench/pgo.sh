#!/usr/bin/env bash
# Build an instrumented binary, run a short representative workload, then
# rebuild with the profile. Keeps the normal build untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=/tmp/uenr_pgo.toml
MODEL=/tmp/uenr_pgo_model.bin
sed -e "s|model_out = \"checkpoints/model.bin\"|model_out = \"$MODEL\"|" \
    -e "s|checkpoint_every = [0-9]*|checkpoint_every = 0|" \
    -e "s|log_every = [0-9]*|log_every = 1|" \
    config/uenr.toml > "$CFG"
rm -rf prof
mkdir -p prof
make -j2 pgo-gen
rm -f "$MODEL"
./uenr-pgo-gen --config "$CFG" train --steps 5 >/dev/null
make -j2 pgo-use
echo "built ./uenr-pgo with profile data in prof/"
