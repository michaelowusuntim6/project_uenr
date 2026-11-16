#!/usr/bin/env bash
# Show/set the CPU frequency governor. Setting it needs root.
set -e
G=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo "current: $(cat "$G")"
if [ "${1:-show}" = "performance" ]; then
  echo "setting performance (needs sudo)..."
  for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee "$c" >/dev/null
  done
  echo "now: $(cat "$G")"
fi
