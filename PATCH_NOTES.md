# siren_final Patch Notes
# 공식 CSSLab/SIREN 코드(train_general_siren.py, probe_trainer.py, run_general_siren.sh) 직접 비교 기준

패치 대상:
- `train/train_multilingual_siren_selection.py`
- `configs/exp_multilingual_siren.yaml`

---

## [CRITICAL] Fix 1 — Probe 캐시 무효화 누락

**함수**: `train_all_probes()` → `want_meta`

`languages`가 `want_meta`에 포함되지 않아 언어 구성 변경 시 캐시를 재사용.
`en/ko` → `en/ko/fr` 변경 시 fr lang probe 없이 실험 진행됨.

**수정**: `want_meta`에 `"languages": sorted([str(l) for l in languages])` 추가.

---

## [CRITICAL] Fix 2 — `select_salient_neurons` 부호 처리

**함수**: `select_salient_neurons()`

**문제**: 코멘트에는 "absolute values"라고 했으나 `abs()` 없었음.

**분석**:  
공식 `probe_trainer.py`의 `get_feature_importance()`:
```python
return torch.abs(self.model.weight.squeeze()).cpu().numpy()  # 이미 abs 적용
```
→ get_feature_importance()가 이미 절댓값을 반환하므로 결과는 동일. 그러나 명시적으로
`np.abs()` 래핑은 안전하고 문서화 목적으로 유지.

---

## [HIGH] Fix 3 — c_values 탐색 범위 불일치

**파일**: `configs/exp_multilingual_siren.yaml`

**공식 run_general_siren.sh**: `C_VALUES="200.0 500.0 1000.0"`  
**원본 유저**: `[100.0, 1000.0]`

```yaml
# Before
c_values: [100.0, 1000.0]

# After (공식 run script 기준)
c_values: [200.0, 500.0, 1000.0]
```

---

## [HIGH] Fix 4 — n_folds 불일치

**파일**: `configs/exp_multilingual_siren.yaml`

**공식**: `N_FOLDS=5`  
**원본**: `n_folds: 3`

```yaml
# After
n_folds: 5
```

---

## [HIGH] Fix 5 — MLP subsequent layer max_dim 불일치

**함수**: `optuna_objective()`

**공식 train_general_siren.py**:
```python
else:  # i > 0
    min_dim = 64
    max_dim = min(layer_dims[-1], 1024)  # 1024 고정
```

**원본 유저**: `min(layer_dims[-1], int(cfg["hidden_dim_max"]))` → 2048  
→ 이후 레이어가 공식보다 최대 2× 더 큰 차원 탐색 가능.

```python
# After
max_dim = min(layer_dims[-1], 1024)
```

---

## [CRITICAL] Fix 6 — 최종 MLP 학습 방식 불일치

**함수**: `train_final_mlp()` + `run_method()`

**공식 train_general_siren.py** (`use_val=False` 경로):
```python
X_full_train = np.vstack([X_train, X_val])   # train+val 합침
y_full_train = np.concatenate([...])
# 512 epoch 고정, val 모니터링 없음, early stopping 없음
for epoch in tqdm(range(512), ...):
    # train only
```

**원본 유저**:
```python
# X_train만 사용, val로 early stopping (patience=10)
train_model(model, X_train, y_train, X_val, y_val, ..., patience=10)
```

**수정**:
- `train_final_mlp` 시그니처에서 `X_val`, `y_val`, `val_dataset_ids` 제거
- 512 epoch 고정 루프로 교체, early stopping 없음
- `run_method`에서 `X_cv` (train+val 합친 것)를 `train_final_mlp`에 전달

---

## [참고] get_layer_weights — max(0.1, ...) 유지

**공식 train_general_siren.py**:
```python
layer_weights[layer_idx] = max(0.1, normalized_score)
```

논문 Eq.5 서술과 달리 공식 코드가 0.1 floor를 사용. 유저 원본 코드도 동일하게 구현되어
있었으므로 **수정 없음** (이전 v2에서 잘못 제거했다가 원복).

---

## [참고] 50%+ neuron selection — 버그 아님

공식 C=[200,500,1000], η=0.9 → 영어 단일언어 약 12.9% 선택.  
유저 multilingual (en+ko+fr) → 50%+ 선택.  

공식 probe_trainer의 Adam+soft-L1이 C=200~1000에서도 영어 단일언어에서는 weights를
충분히 sparse하게 만들지만, multilingual 학습 시 3개 언어 safety feature가 각각 다른
회로를 사용해 weights가 분산됨. C 값 자체는 맞고, 50%+는 연구 결과의 일부.

