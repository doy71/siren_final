#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/exp_nemotron_primary.yaml}
EXTERNAL_CONFIG=${EXTERNAL_CONFIG:-configs/exp_lumees_external_validation.yaml}

python - <<'PY'
import yaml
from train.multilingual_preprocess import build_normalized_dataset

for name, path in [("Nemotron primary", "configs/exp_nemotron_primary.yaml"), ("Lumees external", "configs/exp_lumees_external_validation.yaml")]:
    print("\n" + "=" * 100)
    print(name, path)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Print raw normalized distribution before cap/drop so language availability is visible.
    config["sampling_strategy"] = "none"
    config["balance_per_dataset"] = False
    config["drop_groups_below_min_n"] = False
    config["max_samples_per_group"] = None
    config["min_samples_per_group"] = None

    dfs = build_normalized_dataset(config, skip_failed=True)
    for split, df in dfs.items():
        print("\n" + "-" * 100)
        print(f"[{split}] total n={len(df)}")
        if len(df) == 0:
            continue
        print("\nsource_dataset × lang × label")
        print(
            df.groupby(["source_dataset", "lang", "label"])
            .size()
            .reset_index(name="n")
            .sort_values(["source_dataset", "lang", "label"])
            .to_string(index=False)
        )
        print("\nlang × label")
        print(
            df.groupby(["lang", "label"])
            .size()
            .reset_index(name="n")
            .sort_values(["lang", "label"])
            .to_string(index=False)
        )
PY
