#!/usr/bin/env bash
set -euo pipefail

# Uses CSSLab/SIREN official training code. Do not mix multilingual datasets here.
cd train

MODEL=${MODEL:-qwen3-4b}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-32}
C_VALUES=${C_VALUES:-"200.0 500.0 1000.0"}
THRESHOLDS=${THRESHOLDS:-"0.9"}
N_TRIALS=${N_TRIALS:-32}
N_JOBS=${N_JOBS:-1}
N_FOLDS=${N_FOLDS:-5}
VAL_RATIO=${VAL_RATIO:-0.2}
USE_GPU_DATA=${USE_GPU_DATA:-1}
REP_TYPES=${REP_TYPES:-"residual_mean"}

python train_general_siren.py \
  --model "${MODEL}" \
  --datasets toxic_chat openai_moderation aegis aegis2 wildguard safe_rlhf beavertails \
  --batch_size "${BATCH_SIZE}" \
  --c_values ${C_VALUES} \
  --pooling_types ${REP_TYPES} \
  --thresholds ${THRESHOLDS} \
  --n_trials "${N_TRIALS}" \
  --n_jobs "${N_JOBS}" \
  --n_folds "${N_FOLDS}" \
  --val_ratio "${VAL_RATIO}" \
  --use_gpu_data "${USE_GPU_DATA}" \
  --device "${DEVICE}"
