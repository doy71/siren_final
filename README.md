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

실행 전에는 이제 `scripts/run_multilingual_siren.sh`가 자동으로 전체 enabled dataset의 schema와 normalization 가능 여부를 먼저 검사합니다. 이 단계에서 하나라도 로드/정규화가 실패하면 representation 추출을 시작하지 않고 종료합니다. 수동으로 먼저 확인하려면 아래 명령을 실행하면 됩니다.

```bash
python scripts/validate_dataset_schemas.py \
  --config configs/exp_multilingual_siren.yaml \
  --out_dir outputs/multilingual_siren/_dataset_schema_check \
  --fail_on_error
```

개별 Hugging Face dataset의 실제 config/split/column을 확인하려면 다음처럼 확인할 수 있습니다.

```bash
python scripts/inspect_hf_dataset.py --dataset lmsys/toxic-chat --name toxicchat0124 --split train --limit 2
python scripts/inspect_hf_dataset.py --dataset allenai/wildguardmix --name wildguardtrain --split train --limit 2
```

주의: `allenai/wildguardmix`는 gated dataset이라 Hugging Face에서 AI2 Responsible Use Guidelines에 동의하고 로그인/token 설정을 해야 합니다.

## 이번 수정에서 반영한 dataset schema 확인/수정

- `allenai/wildguardmix`: 실제 로딩 방식에 맞게 `hf_name: wildguardtrain`, `hf_name: wildguardtest`를 분리했습니다.
- `lmsys/toxic-chat`: 최신 권장 config인 `toxicchat0124`를 명시했습니다.
- `lumees/multilingual-safety-classification-dataset`: 실제 schema가 `prompt`, `safety_label`, `lang`이고 train/test split만 있으므로 validation 요청을 제거했습니다. `safety_label=0`은 safe, `1/2`는 binary unsafe/non-safe로 처리합니다.
- `iknow-lab/wildguardmix-test-ko`: 실제 split이 test뿐이므로 nonexistent train split 요청을 제거했습니다.
- `PKU-Alignment/PKU-SafeRLHF`: paired response schema의 per-response category/severity를 보존하도록 했습니다.
- `train/multilingual_preprocess.py`: validation/test 중 하나가 없는 dataset을 처리할 때 기존 held-out test split을 train으로 다시 섞지 않도록 split 보정 로직을 수정했습니다.

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
