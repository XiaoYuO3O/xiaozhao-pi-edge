#!/usr/bin/env bash
# Sync the minimal Mode A runtime from this Mac to the Raspberry Pi, then run the Pi harness there.
#
#   scripts/sync_to_pi.sh                       # PI_HOST defaults to raspberrypi.local
#   PI_HOST=192.168.1.42 scripts/sync_to_pi.sh
#   PI_HOST='[fe80::33cf:4028:550:b54d%en5]' scripts/sync_to_pi.sh   # link-local IPv6 over Ethernet (bracketed)
#   scripts/sync_to_pi.sh --no-test             # copy only
#
# Env: PI_HOST, PI_USER (default icnclab115), PI_DIR (default ~/xiaozhao-pi-edge on the Pi).
# Syncs the current working tree (no git pull). Only these files go to the Pi; nothing else is deleted there.
set -euo pipefail

PI_HOST="${PI_HOST:-raspberrypi.local}"
PI_USER="${PI_USER:-icnclab115}"
PI_DIR="${PI_DIR:-xiaozhao-pi-edge}"
RUN_TEST=1
[[ "${1:-}" == "--no-test" ]] && RUN_TEST=0

cd "$(dirname "$0")/.."
FILES=(
  src/__init__.py
  src/config.py
  src/mpc_core.py
  src/exp1_pi_audio_check.py
  src/edge_state_machine.py
  src/iot_api_client.py
  src/run_live_audio.py
  tests/harness.py
)

echo "=== [1/2] rsync Mode A runtime -> ${PI_USER}@${PI_HOST}:${PI_DIR}/ ==="
ssh "${PI_USER}@${PI_HOST}" "mkdir -p ${PI_DIR}/audio_tmp"
rsync -av --relative "${FILES[@]}" "${PI_USER}@${PI_HOST}:${PI_DIR}/"

if [[ $RUN_TEST -eq 1 ]]; then
  echo "=== [2/2] Pi harness (--pi) ==="
  ssh "${PI_USER}@${PI_HOST}" "cd ${PI_DIR} && python3 tests/harness.py --pi"
fi
