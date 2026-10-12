#!/usr/bin/env bash
set -euo pipefail
TAANP_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TAANP_REPO_ROOT="$(cd -- "${TAANP_SCRIPT_DIR}/.." && pwd)"
cd -- "${TAANP_REPO_ROOT}"
if (( $# != 0 )); then
    echo 'Usage: bash scripts/reproduce_pretrained.sh (configure device and paths via TAANP_* variables)' >&2
    exit 2
fi
export TAANP_PYTHON="${TAANP_PYTHON:-python3}"
export TAANP_OUTPUT_ROOT="${TAANP_OUTPUT_ROOT:-${TAANP_REPO_ROOT}/outputs/pretrained/seed_42}"
# Fix the published configuration; device, Python and data/output paths remain configurable.
export TAANP_SENSOR_SEED=42 TAANP_SAMPLES=30 TAANP_DIY=taanp TAANP_WITHOUT_TAMQM=0
export TAANP_USE_STATIC=1 TAANP_USE_HIT=1 TAANP_USE_SPEED=1 TAANP_USE_POI=0 TAANP_SWANLAB=0
exec bash "${TAANP_SCRIPT_DIR}/inference_taanp.sh" \
    "${TAANP_REPO_ROOT}/check_points/original_pretrained/seed_42/best_train_checkpoint.pth"
