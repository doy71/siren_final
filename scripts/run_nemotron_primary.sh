#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-qwen3-4b}
DEVICE=${DEVICE:-cuda}
CONFIG=${CONFIG:-configs/exp_nemotron_primary.yaml}
EXTERNAL_CONFIG=${EXTERNAL_CONFIG:-configs/exp_lumees_external_validation.yaml}
BATCH_SIZE=${BATCH_SIZE:-32}
N_TRIALS=${N_TRIALS:-32}
N_FOLDS=${N_FOLDS:-5}
SEEDS=${SEEDS:-42}

# Keep SIREN-like baseline first. Add mlp_mean only as an ablation.
POOLING_TYPES=${POOLING_TYPES:-"residual_mean"}

# Match official SIREN threshold first. Use 0.6/0.8 only as extra ablations.
THRESHOLDS=${THRESHOLDS:-"0.9"}

METHODS=${METHODS:-"siren_original shared_only"}
VALIDATE_DATASETS=${VALIDATE_DATASETS:-1}
RUN_EXTERNAL=${RUN_EXTERNAL:-0}
FORCE_REEXTRACT=${FORCE_REEXTRACT:-0}
FORCE_RETRAIN_PROBES=${FORCE_RETRAIN_PROBES:-1}

OUT_ROOT="outputs/nemotron_primary/${MODEL}"
ANALYSIS_DIR="${OUT_ROOT}/analysis"
EXTERNAL_OUT="${OUT_ROOT}/external_lumees"

if [[ "${VALIDATE_DATASETS}" == "1" ]]; then
  python scripts/validate_dataset_schemas.py \
    --config "${CONFIG}" \
    --out_dir "outputs/nemotron_primary/_dataset_schema_check" \
    --fail_on_error

  if [[ "${RUN_EXTERNAL}" == "1" ]]; then
    python scripts/validate_dataset_schemas.py \
      --config "${EXTERNAL_CONFIG}" \
      --out_dir "outputs/lumees_external_validation/_dataset_schema_check" \
      --fail_on_error
  fi
fi

EXTRA_FLAGS=()
if [[ "${FORCE_REEXTRACT}" == "1" ]]; then
  EXTRA_FLAGS+=(--force_reextract)
fi
if [[ "${FORCE_RETRAIN_PROBES}" == "1" ]]; then
  EXTRA_FLAGS+=(--force_retrain_probes)
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
  --skip_failed_datasets \
  "${EXTRA_FLAGS[@]}"

python analysis/analyze_multilingual_siren_results.py \
  --run_dir "${OUT_ROOT}" \
  --out_dir "${ANALYSIS_DIR}"

if [[ "${RUN_EXTERNAL}" == "1" ]]; then
  python analysis/evaluate_external_and_latency.py \
    --run_dir "${OUT_ROOT}" \
    --external_config "${EXTERNAL_CONFIG}" \
    --model "${MODEL}" \
    --out_dir "${EXTERNAL_OUT}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --skip_failed_datasets
fi

echo ""
echo "Done. Main analysis: ${ANALYSIS_DIR}"
if [[ "${RUN_EXTERNAL}" == "1" ]]; then
  echo "Done. Lumees external validation + latency: ${EXTERNAL_OUT}"
fi
