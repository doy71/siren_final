#!/usr/bin/env python
"""
Flatten selection_debug blocks from selected_neurons.json files and print a summary table.

Usage:
  python scripts/inspect_selection_debug.py \
    --run_dir outputs/multilingual_siren_v2/qwen3-4b \
    --out_csv  outputs/multilingual_siren_v2/qwen3-4b/selection_debug.csv

Key columns to check:
  selected_ratio  > 0.5  -> weights are flat (soft L1 artifact; consider eps-cutoff)
  top100_cum_ratio < 0.5 -> top-100 neurons cover less than half the total importance
  nonzero_gt_1e_6        -> number of neurons with weight above 1e-6
"""
import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd


def flatten_debug(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    run_dir = Path(path).parent
    parts = {p.split("=")[0]: p.split("=")[1] for p in run_dir.parts if "=" in p}

    for layer, stats in (data.get("selection_debug", {}).get("global") or {}).items():
        r = dict(stats)
        r.update(parts)
        r.update({"scope": "global", "lang": None, "layer_idx": int(layer), "path": str(run_dir)})
        rows.append(r)

    for lang, by_layer in (data.get("selection_debug", {}).get("language") or {}).items():
        for layer, stats in by_layer.items():
            r = dict(stats)
            r.update(parts)
            r.update({"scope": "language", "lang": lang, "layer_idx": int(layer), "path": str(run_dir)})
            rows.append(r)

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="Root output dir (e.g. outputs/…/qwen3-4b)")
    p.add_argument("--out_csv", default=None, help="Optional CSV output path")
    args = p.parse_args()

    pattern = os.path.join(args.run_dir, "method=*", "threshold=*", "pooling=*", "seed=*", "selected_neurons.json")
    rows = []
    for path in glob.glob(pattern):
        rows.extend(flatten_debug(path))

    if not rows:
        raise FileNotFoundError(
            f"No selection_debug rows found under {args.run_dir}.\n"
            "Re-run training after applying the patch to generate selection_debug data."
        )

    df = pd.DataFrame(rows)
    cols = [
        "method", "threshold", "pooling", "seed", "scope", "lang", "layer_idx",
        "hidden_size", "selected", "selected_ratio",
        "nonzero_gt_1e_6", "top10_cum_ratio", "top100_cum_ratio", "top500_cum_ratio",
        "max_weight", "mean_weight", "median_weight",
    ]
    cols = [c for c in cols if c in df.columns]
    sort_keys = [c for c in ["method", "threshold", "pooling", "scope", "lang", "layer_idx"] if c in df.columns]
    print(df.sort_values(sort_keys, na_position="first")[cols].to_string(index=False))

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
