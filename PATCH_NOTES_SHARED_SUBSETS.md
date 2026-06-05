# 26-way Nemotron `shared_only` patch

## Added

- `configs/exp_nemotron_shared_subsets.yaml`
  - generates exactly 26 subsets: 15 pairs, 4 triples, 6 leave-one-out, 1 all-language
  - uses only Nemotron
  - disables large redundant `layer_values.csv` by default
- `scripts/run_nemotron_shared_subsets.sh`
  - runs both `qwen3-4b` and `llama3.1-8b`
  - does not run Lumees/external validation
  - reuses representation/probe caches and resumes completed subset runs
- `scripts/check_nemotron_shared_subset_progress.py`
  - reports complete, partial, skipped, and missing subsets
- `README_NEMOTRON_SHARED_SUBSETS.md`

## Changed

- `train/train_multilingual_siren_selection.py`
  - supports a different `shared_only` language intersection per run
  - keeps final MLP train/evaluation on all configured languages
  - saves subset metadata in paths, metrics, predictions, selected neurons, and checkpoints
  - supports `llama3-8b` alias and model-aware extractor selection
  - adds method-level resume behavior
- `analysis/analyze_multilingual_siren_results.py`
  - reads nested subset run paths
  - creates subset-size and language-inclusion summaries/plots
  - deduplicates a legacy all-language `shared_only` run when a new explicit subset run exists
- `scripts/run_nemotron_primary.sh`
  - excludes routed/plus methods by default
  - makes external Lumees evaluation opt-in with `RUN_EXTERNAL=1`
