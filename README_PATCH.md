# SIREN dataset + selection patch  (v2 — with bug fixes)

## 적용 방법

```bash
# CSSLab/SIREN repo root에서
cp -r train/ configs/ scripts/ /path/to/CSSLab-SIREN/
```

이 zip은 패치 스크립트를 실행하는 대신 **이미 패치가 적용된 파일**을 바로 복사하는 방식입니다.

---

## 원본 패치 대비 추가 버그 픽스

| # | 위치 | 수정 내용 |
|---|------|-----------|
| 1 | `apply_siren_dataset_selection_patch.py` – `write_file()` | `"\\n"` → `"\n"` (리터럴 백슬래시-n이 파일 끝에 쓰이던 버그) |
| 2 | `multilingual_preprocess.py` – `sample_by_group(balance_min)` | 빈 그룹: `df.iloc[0:0].copy()` → `return df` (원본 동작 복구) |
| 3 | `multilingual_preprocess.py` – `sample_by_group(balance_min)` | `groupby(by, dropna=False)` → `groupby(by)` (원본과 NaN 처리 통일) |

---

## 실험 실행

```bash
# 공식 SIREN 재현 (영어 전용)
bash scripts/run_siren_official_reproduction.sh

# 다국어 v2 실험
bash scripts/run_multilingual_safety_v2.sh

# selection debug 분석
python scripts/inspect_selection_debug.py \
  --run_dir outputs/multilingual_siren_v2/qwen3-4b \
  --out_csv  outputs/multilingual_siren_v2/qwen3-4b/selection_debug.csv
```

---

## selection 비율이 높은 이유

CSSLab/LinearProbe는 Adam + soft L1 (λ = 1/C)을 사용합니다.
sklearn의 exact L1 logistic regression과 달리 weight가 정확히 0이 되지 않아
대부분 뉴런이 아주 작은 nonzero importance를 가집니다.

`inspect_selection_debug.py` 출력의 `top100_cum_ratio` 를 확인하세요.
- 값이 0.5 미만 → weight 분포가 매우 flat → eps-cutoff 적용 검토 필요

```python
# select_salient_neurons에 절대값 하한 추가 (optional)
weights = probe.get_feature_importance()
weights = np.where(weights > 1e-5, weights, 0.0)
```

이는 SIREN 원논문과 다른 동작이므로, debug 결과 확인 후 결정하세요.

---

## local dataset JSONL 형식

```json
{"text": "...", "label": 1, "lang": "fr", "category": "...", "split": "train"}
```

- `label=1`: unsafe/harmful
- `label=0`: safe/harmless
- `split` 없으면 loader가 train/validation/test 로 자동 분할

---

## 파일 구조

```
train/
  multilingual_preprocess.py       ← sampling 붕괴 수정 + sample_by_group 추가
  train_multilingual_siren_selection.py  ← 캐시 무효화 + selection debug 추가
configs/
  exp_multilingual_safety_v2.yaml  ← v2 실험 config (strategy=cap)
scripts/
  run_multilingual_safety_v2.sh
  run_siren_official_reproduction.sh
  inspect_selection_debug.py
```
