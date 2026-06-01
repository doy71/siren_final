#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-qwen3-0.6b}
DEVICE=${DEVICE:-cuda}
CONFIG=${CONFIG:-configs/exp_multilingual_siren.yaml}
BATCH_SIZE=${BATCH_SIZE:-32}
N_TRIALS=${N_TRIALS:-32}
N_FOLDS=${N_FOLDS:-3}
SEEDS=${SEEDS:-42}
POOLING_TYPES=${POOLING_TYPES:-"residual_mean mlp_mean"}
THRESHOLDS=${THRESHOLDS:-"0.6 0.8"}
METHODS=${METHODS:-"siren_original shared_only routed_shared_specific shared_plus_all_specific random_same_size_as_routed random_same_size_as_plus"}
VALIDATE_DATASETS=${VALIDATE_DATASETS:-1}
SCHEMA_CHECK_OUT=${SCHEMA_CHECK_OUT:-"outputs/multilingual_siren/_dataset_schema_check"}

if [[ "${VALIDATE_DATASETS}" == "1" ]]; then
  python scripts/validate_dataset_schemas.py \
    --config "${CONFIG}" \
    --out_dir "${SCHEMA_CHECK_OUT}" \
    --fail_on_error
fi

python train/train_multilingual_siren_selection.py \
  --config "${CONFIG}" \
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --n_trials "${N_TRIALS}" \
  --n_folds "${N_FOLDS}" \
  --seeds ${SEEDS} \
  --pooling_types ${POOLING_TYPES} \
  --thresholds ${THRESHOLDS} \
  --methods ${METHODS} \
  --skip_failed_datasets

python analysis/analyze_multilingual_siren_results.py \
  --run_dir "outputs/multilingual_siren/${MODEL}" \
  --out_dir "outputs/multilingual_siren/${MODEL}/analysis"
