# Nemotron 26-way `shared_only` subset experiment

This experiment changes **only the languages used to define the shared-neuron intersection**.  For every subset, the final MLP is still trained and evaluated on the same six Nemotron languages: `en, fr, ko, ja, zh, hi`.

## Exact 26 subsets

Configured in `configs/exp_nemotron_shared_subsets.yaml`:

- all 15 two-language pairs
- four three-language groups: `en-fr-hi`, `ko-ja-zh`, `en-ko-zh`, `fr-ja-hi`
- all six leave-one-language-out groups
- the full six-language group

`shared_plus_all_specific` and `routed_shared_specific` are not run.

## Run Qwen and Llama

```bash
bash scripts/run_nemotron_shared_subsets.sh
```

Defaults:

- Qwen: `qwen3-4b`
- Llama: `llama3.1-8b`
- threshold: `0.6` only
- pooling: `residual_mean` only
- one seed: `42`
- existing Nemotron representations and probes are reused when cache metadata matches
- completed subset runs are skipped automatically
- no Lumees or other external dataset is loaded

The shared-subset run script intentionally fixes these two conditions and ignores `THRESHOLDS` or `POOLING_TYPES` environment variables, preventing accidental 0.9/MLP runs. Existing `threshold=0.9` output folders may be kept; the automatic analysis command filters them out.

Run one model only:

```bash
MODELS="qwen3-4b" bash scripts/run_nemotron_shared_subsets.sh
MODELS="llama3.1-8b" bash scripts/run_nemotron_shared_subsets.sh
```

Run or debug selected subsets only:

```bash
MODELS="qwen3-4b" SHARED_SUBSET_FILTER="en-fr ko-ja-zh" bash scripts/run_nemotron_shared_subsets.sh
```

Force selected components to rerun:

```bash
FORCE_REEXTRACT=1 bash scripts/run_nemotron_shared_subsets.sh
FORCE_RETRAIN_PROBES=1 bash scripts/run_nemotron_shared_subsets.sh
FORCE_RERUN_METHODS=1 bash scripts/run_nemotron_shared_subsets.sh
```

## Output layout

```text
outputs/nemotron_primary/{model}/
  shared_only_subset_manifest.json
  method=shared_only/
    selection_langs=en-fr/
      threshold=0.6/pooling=residual_mean/seed=42/
        metrics.json
        predictions.jsonl
        selected_neurons.json
        feature_manifest.json
        best_model.pkl
  analysis_shared_subsets/
    shared_only_by_subset.csv
    shared_only_by_subset_size.csv
    shared_only_language_inclusion_effect.csv
    per_language_metrics.csv
    selection_summary.csv
    plots/
```

`save_layer_values: false` is set for this experiment because 26 runs per model would create a very large redundant table. Per-layer selected-neuron counts remain available in `feature_manifest.json` and `selection_summary.csv`.

## Check progress or resume status

```bash
python scripts/check_nemotron_shared_subset_progress.py \
  --config configs/exp_nemotron_shared_subsets.yaml \
  --root outputs/nemotron_primary \
  --models qwen3-4b llama3.1-8b \
  --out_csv outputs/nemotron_primary/shared_subset_progress.csv
```

The main run script is resumable: rerunning it skips completed subset directories and continues missing or partial runs.

For a quick smoke test before the full 52 runs:

```bash
MODELS="qwen3-4b" \
SHARED_SUBSET_FILTER="en-fr" \
N_TRIALS=1 N_FOLDS=2 \
BATCH_SIZE=32 \
bash scripts/run_nemotron_shared_subsets.sh
```
