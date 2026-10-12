#!/usr/bin/env bash
# Shared configuration for the published TA-ANP experiments.
# Source this file from a launcher after enabling strict Bash mode.
TAANP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TAANP_ROOT="$(cd -- "${TAANP_DIR}/.." && pwd)"
cd -- "${TAANP_ROOT}"

TAANP_PYTHON="${TAANP_PYTHON:-python3}"
TAANP_DEVICE="${TAANP_DEVICE:-cuda:0}"
TAANP_SENSOR_SEED="${TAANP_SENSOR_SEED:-42}"
TAANP_SAMPLES="${TAANP_SAMPLES:-30}"
TAANP_DATA_ROOT="${TAANP_DATA_ROOT:-${TAANP_ROOT}/MadridAllinOne/DLData}"
TAANP_OUTPUT_ROOT="${TAANP_OUTPUT_ROOT:-${TAANP_ROOT}/outputs/taanp}"
TAANP_SWANLAB="${TAANP_SWANLAB:-0}"

if [[ ! "${TAANP_SENSOR_SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo 'TAANP_SENSOR_SEED must be a non-negative integer.' >&2
    exit 2
fi
if [[ ! "${TAANP_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo 'TAANP_SAMPLES must be a positive integer.' >&2
    exit 2
fi

TAANP_DIY="${TAANP_DIY:-taanp}"
TAANP_WITHOUT_TAMQM="${TAANP_WITHOUT_TAMQM:-0}"
TAANP_ABLATION_ARGS=()
if [[ "${TAANP_WITHOUT_TAMQM}" == 1 ]]; then
    TAANP_ABLATION_ARGS=(--without_tamqm)
fi
for feature in STATIC HIT SPEED POI; do
    variable="TAANP_USE_${feature}"
    default=1
    if [[ "${feature}" == POI ]]; then default=0; fi
    value="${!variable:-${default}}"
    option="use-${feature,,}"
    option="${option//_/-}"
    case "${value}" in
        1) TAANP_ABLATION_ARGS+=("--${option}") ;;
        0) TAANP_ABLATION_ARGS+=("--no-${option}") ;;
        *) echo "${variable} must be 0 or 1." >&2; exit 2 ;;
    esac
done

TAANP_ARGS=(
    -model TAANP -dev "${TAANP_DEVICE}"
    -seed 42 -su "${TAANP_SENSOR_SEED}"
    -me 50 -log_int 2000 -tbs 16 -sbs 2 -mtb 3000 -warm 1
    -tr 0.6 -vr 0 -loss mask_beta_elbo
    -iw 4 -lbw 4 -pw 4 -st 1 -td 1 -pur 0.6
    -etype default -diy "${TAANP_DIY}" -pt 30
    -beta_scheduler linear -bse 20
    -h_dim 64 -z_dim 64 -num_heads 4 -sampler NPSampler1 -npp both
    -mn mask_mae mask_smape mask_picb mask_qice
    -dataset MadridFSDataset4NP -opp normal -p Final
    -sdp fixed_sensor_data.h5 -tfs q
    -dr "${TAANP_DATA_ROOT}"
    -bcp "${TAANP_OUTPUT_ROOT}/check_points/"
    -bpp "${TAANP_OUTPUT_ROOT}/predictions/"
    -bmp "${TAANP_OUTPUT_ROOT}/results/"
    -bip "${TAANP_OUTPUT_ROOT}/images/"
    -logp "${TAANP_OUTPUT_ROOT}/logs/"
    "${TAANP_ABLATION_ARGS[@]}"
)

# Use the same naming function as main.py so explicit ablations cannot overwrite each other.
TAANP_PROJECT="Final-MadridFSDataset4NP-0.40-4-4-4-1-${TAANP_SENSOR_SEED}-60"
TAANP_EXP="$("${TAANP_PYTHON}" args.py "${TAANP_ARGS[@]}" --print-experiment-id)"
TAANP_DEFAULT_CHECKPOINT="${TAANP_OUTPUT_ROOT}/check_points/${TAANP_PROJECT}/${TAANP_EXP}/best_train_checkpoint.pth"
