#!/usr/bin/env bash
set -euo pipefail
# Resolve a supplied relative checkpoint before common.sh changes directory.
if (( $# > 1 )); then
    echo 'Usage: bash scripts/inference_taanp.sh [checkpoint.pth]' >&2
    exit 2
fi
TAANP_CHECKPOINT="${1:-}"
if [[ -n "${TAANP_CHECKPOINT}" && "${TAANP_CHECKPOINT}" != /* ]]; then
    TAANP_CHECKPOINT="${PWD}/${TAANP_CHECKPOINT}"
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
TAANP_CHECKPOINT="${TAANP_CHECKPOINT:-${TAANP_DEFAULT_CHECKPOINT}}"
if [[ ! -f "${TAANP_CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${TAANP_CHECKPOINT}" >&2
    echo 'Run train_taanp.sh first, or supply an existing checkpoint path.' >&2
    exit 1
fi

mkdir -p -- "${TAANP_OUTPUT_ROOT}/predictions/${TAANP_PROJECT}/${TAANP_EXP}" \
    "${TAANP_OUTPUT_ROOT}/results/${TAANP_PROJECT}"
echo "Running TA-ANP inference with K=${TAANP_SAMPLES}: ${TAANP_CHECKPOINT}"
exec "${TAANP_PYTHON}" main.py "${TAANP_ARGS[@]}" \
    -pmp "${TAANP_CHECKPOINT}" -io -num_sampling "${TAANP_SAMPLES}"
