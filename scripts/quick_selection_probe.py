#!/usr/bin/env python
"""
Quick neuron-selection dry run.

Checks how many neurons SIREN-style probe selection chooses WITHOUT training the final MLP.
Run from repo root after applying siren_dataset_selection_patch.zip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path.cwd()
sys.path.append(str(ROOT))

from train.multilingual_preprocess import build_normalized_dataset, save_dataset_artifacts
from train.train_multilingual_siren_selection import (
    active_feature_counts,
    build_feature_manifest,
    collect_selection_debug,
    dataset_signature_from_dfs,
    extract_representations,
    infer_hidden_size_by_layer,
    selected_by_language,
    selected_global,
    set_seed,
    train_all_probes,
)

DEFAULT_METHODS = [
    "siren_original",
    "shared_only",
    "routed_shared_specific",
    "shared_plus_all_specific",
    "random_same_size_as_routed",
    "random_same_size_as_plus",
]


def flatten_selection_debug(debug: Dict[str, Any], threshold: float, pooling_type: str) -> pd.DataFrame:
    rows = []
    for layer, stats in (debug.get("global") or {}).items():
        row = dict(stats)
        row.update({"threshold": threshold, "pooling_type": pooling_type, "scope": "global", "lang": None, "layer_idx": int(layer)})
        rows.append(row)
    for lang, by_layer in (debug.get("language") or {}).items():
        for layer, stats in by_layer.items():
            row = dict(stats)
            row.update({"threshold": threshold, "pooling_type": pooling_type, "scope": "language", "lang": lang, "layer_idx": int(layer)})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_selected_dict(selected: Dict[int, Sequence[int]], hidden_size_by_layer: Dict[int, int], scope: str, threshold: float, pooling_type: str, lang: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for layer, neurons in sorted(selected.items()):
        hidden = int(hidden_size_by_layer.get(int(layer), 0))
        n = int(len(neurons))
        rows.append({
            "threshold": threshold,
            "pooling_type": pooling_type,
            "scope": scope,
            "lang": lang,
            "layer_idx": int(layer),
            "hidden_size": hidden,
            "n_selected": n,
            "selected_ratio": float(n / hidden) if hidden else None,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--pooling_types", nargs="+", default=["residual_mean"])
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.9])
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--out_root", default="outputs/quick_selection")
    p.add_argument("--skip_failed_datasets", action="store_true")
    p.add_argument("--force_reextract", action="store_true")
    p.add_argument("--force_retrain_probes", action="store_true")
    p.add_argument("--max_samples_per_dataset_split", type=int, default=None)
    p.add_argument("--max_samples_per_group", type=int, default=300)
    p.add_argument("--min_samples_per_group", type=int, default=None)
    p.add_argument("--no_drop_small_groups", action="store_true")
    p.add_argument("--c_values", nargs="+", type=float, default=None)
    p.add_argument("--n_c_values", type=int, default=None, help="Use only the first N C values for faster dry runs.")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["output_root"] = args.out_root
    if args.max_samples_per_dataset_split is not None:
        config["max_samples_per_dataset_split"] = int(args.max_samples_per_dataset_split)
    if args.max_samples_per_group is not None:
        config["max_samples_per_group"] = int(args.max_samples_per_group)
        config["sampling_strategy"] = "cap"
        config["balance_per_dataset"] = False
    if args.min_samples_per_group is not None:
        config["min_samples_per_group"] = int(args.min_samples_per_group)
    if args.no_drop_small_groups:
        config["drop_groups_below_min_n"] = False

    c_values = args.c_values if args.c_values is not None else list(config["probe"]["c_values"])
    if args.n_c_values is not None:
        c_values = c_values[: int(args.n_c_values)]
    config["probe"]["c_values"] = [float(c) for c in c_values]

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    out_dir = Path(args.out_root) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_quick_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    print("[1] Build normalized dataset")
    dfs = build_normalized_dataset(config, skip_failed=args.skip_failed_datasets)
    for split, df in dfs.items():
        langs = sorted(df["lang"].unique().tolist()) if len(df) and "lang" in df else []
        labels = df["label"].value_counts().to_dict() if len(df) and "label" in df else {}
        print(f"{split:10s}: {len(df):7d} | langs={langs} | labels={labels}")

    save_dataset_artifacts(dfs, str(out_dir))
    dataset_sig = dataset_signature_from_dfs(dfs)
    with open(out_dir / "dataset_signature.json", "w", encoding="utf-8") as f:
        json.dump(dataset_sig, f, indent=2, ensure_ascii=False)

    print("[2] Extract/reuse representations")
    all_reps = extract_representations(
        args.model,
        dfs,
        str(device),
        args.batch_size,
        args.pooling_types,
        str(out_dir),
        force_reextract=args.force_reextract,
        dataset_signature=dataset_sig,
    )

    print("[3] Train global/language probes only")
    languages = [str(x) for x in config.get("languages", sorted(set(all_reps["train"]["langs"])))]
    global_probes, lang_probes = train_all_probes(
        all_reps=all_reps,
        languages=languages,
        c_values=config["probe"]["c_values"],
        pooling_types=args.pooling_types,
        device=str(device),
        out_dir=str(out_dir),
        metric=config["probe"].get("metric", "f1_macro"),
        force_retrain=args.force_retrain_probes,
    )

    print("[4] Compute selection counts and method budgets")
    num_layers = int(all_reps["train"]["num_layers"])
    selection_rows = []
    debug_frames = []
    budget_rows = []

    for pooling_type in args.pooling_types:
        hidden_size_by_layer = infer_hidden_size_by_layer(all_reps["train"]["representations"], pooling_type, num_layers)
        for threshold in args.thresholds:
            global_sel = selected_global(global_probes, pooling_type, threshold, num_layers)
            lang_sel = selected_by_language(lang_probes, languages, pooling_type, threshold, num_layers)
            selection_rows.extend(summarize_selected_dict(global_sel, hidden_size_by_layer, "global", threshold, pooling_type, None))
            for lang in languages:
                selection_rows.extend(summarize_selected_dict(lang_sel.get(lang, {}), hidden_size_by_layer, "language", threshold, pooling_type, lang))

            debug = collect_selection_debug(global_probes, lang_probes, languages, pooling_type, threshold, num_layers)
            debug_frames.append(flatten_selection_debug(debug, threshold, pooling_type))

            for method in args.methods:
                features = build_feature_manifest(
                    method=method,
                    global_selected=global_sel,
                    lang_selected=lang_sel,
                    languages=languages,
                    pooling_type=pooling_type,
                    num_layers=num_layers,
                    hidden_size_by_layer=hidden_size_by_layer,
                    shared_min_langs=int(config.get("shared_min_langs", len(languages))),
                    seed=seed,
                )
                counts = active_feature_counts(features, languages)
                budget_rows.append({
                    "method": method,
                    "threshold": threshold,
                    "pooling_type": pooling_type,
                    "union_dim": int(len(features)),
                    **{f"active_{k}": int(v) for k, v in counts.items()},
                })

    selection_df = pd.DataFrame(selection_rows)
    debug_df = pd.concat(debug_frames, ignore_index=True) if debug_frames else pd.DataFrame()
    budget_df = pd.DataFrame(budget_rows)

    selection_path = out_dir / "quick_selection_counts.csv"
    debug_path = out_dir / "quick_selection_debug.csv"
    budget_path = out_dir / "quick_feature_budgets.csv"
    selection_df.to_csv(selection_path, index=False)
    debug_df.to_csv(debug_path, index=False)
    budget_df.to_csv(budget_path, index=False)

    print("\nSaved:")
    print(f"  {selection_path}")
    print(f"  {debug_path}")
    print(f"  {budget_path}")

    print("\n=== Method feature budgets ===")
    if len(budget_df):
        print(budget_df.to_string(index=False))

    print("\n=== Global selection summary by threshold/pooling ===")
    if len(selection_df):
        global_sum = (
            selection_df[selection_df["scope"] == "global"]
            .groupby(["threshold", "pooling_type"])
            .agg(total_selected=("n_selected", "sum"), total_hidden=("hidden_size", "sum"), mean_layer_ratio=("selected_ratio", "mean"))
            .reset_index()
        )
        global_sum["total_selected_ratio"] = global_sum["total_selected"] / global_sum["total_hidden"]
        print(global_sum.to_string(index=False))

    print("\nInterpretation tip:")
    print("  If top100_cum_ratio is low and nonzero_gt_1e_6 is close to hidden_size,")
    print("  the L1 probe is diffuse; selection is large because importance is spread out.")


if __name__ == "__main__":
    main()
