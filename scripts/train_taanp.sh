#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

if (( $# != 0 )); then
    echo 'Usage: bash scripts/train_taanp.sh (configure via TAANP_* environment variables)' >&2
    exit 2
fi
TAANP_LOG_ARGS=()
if [[ "${TAANP_SWANLAB}" == 1 ]]; then
    TAANP_LOG_ARGS=(-wb)
fi

echo "Training TA-ANP with sensor partition seed ${TAANP_SENSOR_SEED}."
echo "Best training checkpoint: ${TAANP_DEFAULT_CHECKPOINT}"
exec "${TAANP_PYTHON}" main.py "${TAANP_ARGS[@]}" "${TAANP_LOG_ARGS[@]}"
