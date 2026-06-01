# Multilingual SIREN Selection Extension

이 폴더는 CSSLab/SIREN 공식 저장소의 학습 흐름을 유지하면서, neuron selection 방식만 multilingual 실험용으로 확장한 코드입니다.

## 유지한 SIREN 절차

- Frozen backbone LLM 사용
- 각 transformer layer의 residual/FFN representation 추출
- sequence-level mean pooling
- layer-wise L1-regularized linear probe 학습
- probe weight magnitude 기반 cumulative threshold neuron selection
- validation macro-F1 기반 layer performance weighting
- selected neuron aggregation 후 MLP classifier 학습
- Optuna로 MLP hidden layers / hidden dimensions / dropout / learning rate 탐색
- raw predictions, selected neurons, layer values, metrics 저장

## 바꾼 부분

원본 SIREN의 neuron selection이 `all-language/global pooled` 기준으로 하나의 neuron set을 고르는 것이라면, 이 코드는 아래 selection methods를 추가합니다.

- `siren_original`: 원본 SIREN 방식. 전체 multilingual train set에서 probe를 학습하고 threshold로 neuron 선택.
- `shared_only`: 언어별 probe에서 공통으로 선택된 neuron만 사용.
- `shared_plus_all_specific`: shared neuron + 모든 언어별 specific neuron을 전부 union으로 사용.
- `routed_shared_specific`: shared neuron은 항상 사용하고, sample의 language route에 해당하는 language-specific neuron만 활성화. 구현상 MLP input dimension을 고정하기 위해 union feature space를 만들고, route에 맞지 않는 specific feature는 0으로 masking합니다.
- `random_same_size_as_routed`: routed와 동일한 active neuron budget을 맞춘 random baseline.
- `random_same_size_as_plus`: plus와 동일한 neuron budget을 맞춘 random baseline.

## 사용 위치

CSSLab/SIREN repo root에 이 폴더의 `train/`, `analysis/`, `configs/`, `scripts/` 내용을 복사해서 실행하는 것을 전제로 합니다.

```bash
cd SIREN
cp -r /path/to/siren_multilingual_ext/* .
conda activate siren
pip install -r requirements_experiment.txt
bash scripts/run_multilingual_siren.sh
```

## 빠른 실행

```bash
python train/train_multilingual_siren_selection.py \
  --config configs/exp_multilingual_siren.yaml \
  --model qwen3-0.6b \
  --methods siren_original shared_only routed_shared_specific shared_plus_all_specific random_same_size_as_routed random_same_size_as_plus \
  --thresholds 0.6 0.8 \
  --pooling_types residual_mean mlp_mean \
  --n_trials 32 \
  --n_folds 3 \
  --device cuda
```

## 분석 실행

```bash
python analysis/analyze_multilingual_siren_results.py \
  --run_dir outputs/multilingual_siren/qwen3-0.6b \
  --out_dir outputs/multilingual_siren/qwen3-0.6b/analysis
```

## 데이터셋 구성 원칙

SIREN 원논문이 다양한 safety benchmark를 binary harmfulness로 통합한 것처럼, 여기서도 단순 hate/toxicity 문장만이 아니라 prompt harmfulness, response harmfulness, jailbreak, refusal/compliance, broader content-safety taxonomy를 포함하도록 구성했습니다. 다만 Hugging Face dataset의 schema가 자주 바뀌므로, 각 dataset spec은 `configs/exp_multilingual_siren.yaml`에서 수정 가능하게 만들었습니다.

실행 전 dataset schema가 맞는지 먼저 아래 명령으로 확인하는 것을 권장합니다.

```bash
python scripts/inspect_hf_dataset.py --dataset allenai/wildguardmix --split train --limit 2
```

## 산출물

기본 output 구조:

```text
outputs/multilingual_siren/{model}/
  normalized_dataset.jsonl
  dataset_summary.csv
  reps_cache/
  probes/
    global_probes.pkl
    lang_probes.pkl
    probe_metrics.csv
  method={method}/threshold={threshold}/pooling={pooling}/seed={seed}/
    best_model.pkl
    metrics.json
    predictions.jsonl
    layer_values.csv
    selected_neurons.json
    feature_manifest.json
    train_log.json
  analysis/
    metrics_summary.csv
    per_dataset_metrics.csv
    per_language_metrics.csv
    selection_summary.csv
    layer_distribution.csv
    plots/*.png
```
