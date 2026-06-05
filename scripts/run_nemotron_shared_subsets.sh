#!/usr/bin/env bash
set -euo pipefail

# Runs the exact 26 shared_only subsets for both official SIREN model keys.
# Override either key when needed, e.g. LLAMA_MODEL=llama3.2-3b.
QWEN_MODEL=${QWEN_MODEL:-qwen3-4b}
LLAMA_MODEL=${LLAMA_MODEL:-llama3.1-8b}
MODELS=${MODELS:-"${QWEN_MODEL} ${LLAMA_MODEL}"}

DEVICE=${DEVICE:-cuda}
CONFIG=${CONFIG:-configs/exp_nemotron_shared_subsets.yaml}
BATCH_SIZE=${BATCH_SIZE:-32}
N_TRIALS=${N_TRIALS:-32}
N_FOLDS=${N_FOLDS:-5}
SEEDS=${SEEDS:-42}
POOLING_TYPES=${POOLING_TYPES:-"residual_mean"}
THRESHOLDS=${THRESHOLDS:-"0.9"}

VALIDATE_DATASET=${VALIDATE_DATASET:-1}
FORCE_REEXTRACT=${FORCE_REEXTRACT:-0}
FORCE_RETRAIN_PROBES=${FORCE_RETRAIN_PROBES:-0}
FORCE_RERUN_METHODS=${FORCE_RERUN_METHODS:-0}
# Optional exact subset slugs, useful for debugging/parallel chunks, e.g. "en-fr ko-ja-zh".
SHARED_SUBSET_FILTER=${SHARED_SUBSET_FILTER:-""}

if [[ "${VALIDATE_DATASET}" == "1" ]]; then
  python scripts/validate_dataset_schemas.py \
    --config "${CONFIG}" \
    --out_dir "outputs/nemotron_primary/_dataset_schema_check" \
    --fail_on_error
fi

for MODEL in ${MODELS}; do
  echo ""
  echo "============================================================"
  echo "Running 26 Nemotron shared_only subsets: ${MODEL}"
  echo "============================================================"

  EXTRA_FLAGS=()
  if [[ "${FORCE_REEXTRACT}" == "1" ]]; then
    EXTRA_FLAGS+=(--force_reextract)
  fi
  if [[ "${FORCE_RETRAIN_PROBES}" == "1" ]]; then
    EXTRA_FLAGS+=(--force_retrain_probes)
  fi
  if [[ "${FORCE_RERUN_METHODS}" == "1" ]]; then
    EXTRA_FLAGS+=(--force_rerun_methods)
  fi
  if [[ -n "${SHARED_SUBSET_FILTER}" ]]; then
    # shellcheck disable=SC2206
    SUBSET_ARGS=(${SHARED_SUBSET_FILTER})
    EXTRA_FLAGS+=(--shared_subset_filter "${SUBSET_ARGS[@]}")
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
    --methods shared_only \
    --skip_failed_datasets \
    "${EXTRA_FLAGS[@]}"

  RESOLVED_MODEL="${MODEL}"
  if [[ ! -d "outputs/nemotron_primary/${RESOLVED_MODEL}" ]] &&      [[ "${RESOLVED_MODEL}" == "llama3-8b" || "${RESOLVED_MODEL}" == "llama-3.1-8b" ]]; then
    RESOLVED_MODEL="llama3.1-8b"
  fi
  OUT_ROOT="outputs/nemotron_primary/${RESOLVED_MODEL}"
  python analysis/analyze_multilingual_siren_results.py \
    --run_dir "${OUT_ROOT}" \
    --out_dir "${OUT_ROOT}/analysis_shared_subsets"
done

echo ""
echo "Done. Results are under outputs/nemotron_primary/{model}/method=shared_only/selection_langs=..."
