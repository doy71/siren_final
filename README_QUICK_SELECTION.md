# Quick neuron selection dry-run tool

이 스크립트는 final MLP를 학습하지 않고, representation extraction + probe training + neuron selection 수만 빠르게 확인합니다.

## 설치

repo root에서:

```bash
unzip siren_quick_selection_tool.zip
chmod +x scripts/quick_selection_probe.py
```

이전 패치(`apply_siren_dataset_selection_patch.py`)를 먼저 적용한 상태를 전제로 합니다.

## 빠른 실행 예시

```bash
python scripts/quick_selection_probe.py \
  --config configs/exp_multilingual_safety_v2.yaml \
  --model qwen3-4b \
  --thresholds 0.9 \
  --pooling_types residual_mean \
  --max_samples_per_group 300 \
  --n_c_values 1 \
  --force_retrain_probes \
  --skip_failed_datasets
```

## 출력

기본 출력 경로:

```text
outputs/quick_selection/{model}/
  quick_selection_counts.csv
  quick_selection_debug.csv
  quick_feature_budgets.csv
```

- `quick_selection_counts.csv`: layer별 selected neuron 수
- `quick_selection_debug.csv`: top100 cumulative ratio, nonzero weight 수 등 원인 분석용
- `quick_feature_budgets.csv`: siren_original/shared/routed/plus/random method별 feature dimension
