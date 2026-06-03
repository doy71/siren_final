# siren_final Bug Fixes — Patch Notes

패치 대상 파일:
- `train/train_multilingual_siren_selection.py`
- `scripts/inspect_selection_debug.py`
- `scripts/quick_selection_probe.py`
- `configs/exp_multilingual_siren.yaml`

---

## [CRITICAL] Bug 1 — Probe 캐시 무효화 누락 (`train_all_probes`)

**파일**: `train/train_multilingual_siren_selection.py`  
**함수**: `train_all_probes()` → `want_meta` dict

`languages`가 `want_meta`에 없어서 `languages: [en, ko]` → `[en, ko, fr]`로 변경해도
캐시가 재사용됨. 프랑스어 lang probe가 없는 상태에서 `shared_min_langs=3` 조건을 만족하는
neuron이 0이 되어 조용히 틀린 실험 결과가 나옴.

**수정**: `"languages": sorted([str(l) for l in languages])` 를 `want_meta`에 추가.

**영향**: probe cache가 있는 상태에서 언어 구성을 바꾼 적 있다면 `probes/` 디렉토리를
삭제하고 재실행 필요.

---

## [HIGH] Bug 2 — `infer_hidden_size_by_layer` IndexError

**파일**: `train/train_multilingual_siren_selection.py`  
**함수**: `infer_hidden_size_by_layer()`

`representations = []`일 때 `representations[0]` → `IndexError`.
edge dataset(특정 언어 샘플 0개)에서 크래시.

**수정**: 빈 리스트 guard 추가 — `{layer: 0 for layer in range(num_layers)}` 반환.

---

## [MEDIUM+NEURON-SEL] Bug 3 — `select_salient_neurons` 부호/누적합 오류

**파일**: `train/train_multilingual_siren_selection.py`  
**함수**: `select_salient_neurons()`, `selection_debug_stats()`

주석에는 "cumulative **absolute** importance"라고 명시되어 있으나 `np.abs()`를 호출하지 않음.
Adam+softL1 probe는 음수 weight를 가질 수 있어, total이 과소 계산되고 sorted_idx 순서가
절댓값 기준이 아니게 됨.

### 추가: `eps_weight_ratio` 파라미터 (neuron selection 과다 선택 대응)

CSSLab SIREN의 softL1 probe는 weight를 정확히 0으로 만들지 않아,
cumsum이 80%에 도달하려면 전체 뉴런의 60~80%를 선택해야 하는 상황이 발생.

```yaml
# configs/exp_multilingual_siren.yaml
probe:
  eps_weight_ratio: 0.0   # 0 = 비활성화 (기본값)
                           # 1e-3 권장: max_weight * 1e-3 미만 weight를 0으로 처리
```

또는 CLI:
```bash
python scripts/quick_selection_probe.py --eps_weight_ratio 1e-3 ...
```

selection ratio > 50% 시 `[WARN-SEL]` 경고를 표준 출력으로 표시.

**수정**:
- `weights = np.abs(np.asarray(probe.get_feature_importance(), dtype=np.float64))`
- `eps_weight_ratio` 파라미터 추가 (선택적 soft-zero 컷오프)
- 선택 비율 > 50% 시 경고 출력

---

## [HIGH] Bug 4 — `GradScaler`/`autocast` 하드코딩 `"cuda"` 디바이스

**파일**: `train/train_multilingual_siren_selection.py`  
**함수**: `train_model()`, `predict_scores()`

`torch.amp.GradScaler("cuda", ...)`, `torch.amp.autocast("cuda", ...)` 가 CPU-only
환경에서 경고 또는 에러를 발생시킴.

**수정**: `"cuda"` → `device.type` 으로 동적 처리.

---

## [MEDIUM] Bug 5 — `torch.FloatTensor/LongTensor().to(device)` 이중 할당

**파일**: `train/train_multilingual_siren_selection.py`  
**함수**: `train_model()`, `predict_scores()`

CPU에 먼저 올린 뒤 GPU로 복사 → 불필요한 메모리 이동.

**수정**: `torch.tensor(X, dtype=torch.float32, device=device)` 로 직접 생성.

---

## [MEDIUM] Bug 6 — `run_method()` 대규모 feature 경고 누락

**파일**: `train/train_multilingual_siren_selection.py`  
**함수**: `run_method()`

선택된 feature가 전체 뉴런의 50% 초과 시 `[WARN-DIM]` 경고 출력.
`eps_weight_ratio` 를 config에서 읽어 selection 함수에 전달.

---

## [LOW] Bug 7 — `inspect_selection_debug.py` 경로 파싱 취약점

**파일**: `scripts/inspect_selection_debug.py`

`p.split("=")[1]` → `=`가 여러 개인 경로 컴포넌트에서 값이 잘림.

**수정**: `k, _, v = p.partition("=")` 사용.

---

## 적용 방법

```bash
# SIREN repo root 기준 (README 지시에 따라)
cp siren_final_patch/train/train_multilingual_siren_selection.py train/
cp siren_final_patch/scripts/inspect_selection_debug.py scripts/
cp siren_final_patch/scripts/quick_selection_probe.py scripts/
cp siren_final_patch/configs/exp_multilingual_siren.yaml configs/

# probe 캐시 언어 구성이 바뀐 경우 캐시 삭제
rm -rf outputs/multilingual_siren/<model>/probes/
```

## neuron selection 과다 선택 시 튜닝 가이드

1. **먼저 진단**: `quick_selection_probe.py` 실행 후 `quick_selection_debug.csv` 확인
   - `selected_ratio > 0.5` → soft-L1 weight가 평탄한 상태
   - `top100_cum_ratio < 0.5` → 상위 100개 뉴런이 중요도의 절반 미만

2. **1단계 대응**: `probe.eps_weight_ratio: 1e-3` 설정 (가장 간단)

3. **2단계 대응**: `probe.c_values: [0.1, 1.0, 10.0]` 으로 낮춰 sparser probe 훈련
   (C가 낮을수록 L1 penalty가 강해져 더 많은 weight가 0에 가까워짐)

4. **확인 방법**: 재실행 후 `WARN-SEL` 경고가 사라지면 성공
