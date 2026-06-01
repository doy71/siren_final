#!/usr/bin/env python
import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_auc(y, s):
    try:
        if len(np.unique(y)) < 2:
            return None
        return float(roc_auc_score(y, s))
    except Exception:
        return None


def safe_auprc(y, s):
    try:
        if len(np.unique(y)) < 2:
            return None
        return float(average_precision_score(y, s))
    except Exception:
        return None


def metrics_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    y = df["label"].astype(int).to_numpy()
    s = df["score"].astype(float).to_numpy()
    p = (s >= 0.5).astype(int)
    return {
        "n": int(len(df)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "unsafe_f1": float(f1_score(y, p, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)) if len(np.unique(y)) > 1 else None,
        "precision_unsafe": float(precision_score(y, p, pos_label=1, zero_division=0)),
        "recall_unsafe": float(recall_score(y, p, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, p)) if len(np.unique(y)) > 1 else None,
        "auroc": safe_auc(y, s),
        "auprc": safe_auprc(y, s),
    }


def load_runs(run_dir: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(run_dir, "method=*", "threshold=*", "pooling=*", "seed=*", "metrics.json")):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        row = {
            "run_path": os.path.dirname(path),
            "model": m.get("model"),
            "method": m.get("method"),
            "threshold": m.get("threshold"),
            "pooling_type": m.get("pooling_type"),
            "seed": m.get("seed"),
            "num_features_union_dim": m.get("num_features_union_dim"),
            "cv_f1": m.get("cv_f1"),
            "best_val_f1": m.get("best_val_f1"),
        }
        for k, v in (m.get("overall_test") or {}).items():
            row[f"test_{k}"] = v
        afc = m.get("active_feature_counts") or {}
        for k, v in afc.items():
            row[f"active_features_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def collect_group_metrics(run_df: pd.DataFrame, group_col: str, split: str = "test") -> pd.DataFrame:
    rows = []
    for _, r in run_df.iterrows():
        pred_path = os.path.join(r["run_path"], "predictions.jsonl")
        if not os.path.exists(pred_path):
            continue
        df = pd.read_json(pred_path, lines=True)
        df = df[df["split"] == split]
        for key, g in df.groupby(group_col):
            m = metrics_from_df(g)
            rows.append({
                "method": r["method"],
                "threshold": r["threshold"],
                "pooling_type": r["pooling_type"],
                "seed": r["seed"],
                group_col: key,
                **m,
            })
    return pd.DataFrame(rows)


def collect_selection(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in run_df.iterrows():
        manifest_path = os.path.join(r["run_path"], "feature_manifest.json")
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path, "r", encoding="utf-8") as f:
            feats = json.load(f)
        if not feats:
            continue
        fdf = pd.DataFrame(feats)
        for (layer, kind, route), g in fdf.groupby(["layer", "kind", "route_lang"], dropna=False):
            rows.append({
                "method": r["method"],
                "threshold": r["threshold"],
                "pooling_type": r["pooling_type"],
                "seed": r["seed"],
                "layer_idx": layer,
                "kind": kind,
                "route_lang": route if pd.notna(route) else "shared_or_global",
                "n_features": len(g),
            })
    return pd.DataFrame(rows)


def collect_layer_values(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in run_df.iterrows():
        path = os.path.join(r["run_path"], "layer_values.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            # layer_values.csv historically did not carry threshold (and may
            # lack method/pooling_type for older runs); attach from run_df so
            # the downstream groupby on these keys never raises KeyError.
            df["method"] = r["method"]
            df["threshold"] = r["threshold"]
            df["pooling_type"] = r["pooling_type"]
            df["seed"] = r["seed"]
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_table(df: pd.DataFrame, path: str) -> None:
    if len(df) > 0:
        df.to_csv(path, index=False)


def line_plot(df: pd.DataFrame, x: str, y: str, hue: str, path: str, title: str, ylabel: Optional[str] = None) -> None:
    if len(df) == 0:
        return
    plt.figure(figsize=(10, 6))
    for key, g in df.groupby(hue):
        agg = g.groupby(x)[y].mean().reset_index().sort_values(x)
        plt.plot(agg[x], agg[y], marker="o", label=str(key))
    plt.xlabel(x)
    plt.ylabel(ylabel or y)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def bar_plot(df: pd.DataFrame, x: str, y: str, path: str, title: str, group: Optional[str] = None) -> None:
    if len(df) == 0:
        return
    plt.figure(figsize=(12, 6))
    if group:
        labels = sorted(df[x].unique())
        groups = sorted(df[group].unique())
        width = 0.8 / max(1, len(groups))
        positions = np.arange(len(labels))
        for i, gr in enumerate(groups):
            vals = []
            for lab in labels:
                vals.append(df[(df[x] == lab) & (df[group] == gr)][y].mean())
            plt.bar(positions + i * width, vals, width=width, label=str(gr))
        plt.xticks(positions + width * (len(groups)-1) / 2, labels, rotation=30, ha="right")
        plt.legend()
    else:
        agg = df.groupby(x)[y].mean().sort_values(ascending=False)
        plt.bar(agg.index.astype(str), agg.values)
        plt.xticks(rotation=30, ha="right")
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    runs = load_runs(args.run_dir)
    if len(runs) == 0:
        raise FileNotFoundError(f"No metrics.json found under {args.run_dir}")
    runs = runs.sort_values(["pooling_type", "threshold", "method", "seed"])
    save_table(runs, os.path.join(args.out_dir, "metrics_summary.csv"))

    # Mean/std summary across seeds.
    metric_cols = [c for c in runs.columns if c.startswith("test_") and pd.api.types.is_numeric_dtype(runs[c])]
    agg = runs.groupby(["method", "threshold", "pooling_type"])[metric_cols + ["num_features_union_dim"]].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([str(x) for x in col if x != ""]).rstrip("_") for col in agg.columns]
    save_table(agg, os.path.join(args.out_dir, "metrics_mean_std.csv"))

    lang_metrics = collect_group_metrics(runs, "lang")
    ds_metrics = collect_group_metrics(runs, "source_dataset")
    save_table(lang_metrics, os.path.join(args.out_dir, "per_language_metrics.csv"))
    save_table(ds_metrics, os.path.join(args.out_dir, "per_dataset_metrics.csv"))

    selection = collect_selection(runs)
    save_table(selection, os.path.join(args.out_dir, "selection_summary.csv"))

    layer_values = collect_layer_values(runs)
    if len(layer_values) > 0:
        layer_dist = (
            layer_values.groupby(["method", "threshold", "pooling_type", "lang", "layer_idx"])
            .agg(n_selected_neurons=("n_selected_neurons", "mean"), abs_sum=("abs_sum", "mean"), l2=("l2", "mean"))
            .reset_index()
        )
        save_table(layer_dist, os.path.join(args.out_dir, "layer_distribution.csv"))
    else:
        layer_dist = pd.DataFrame()

    # Plots: keep metrics intentionally non-redundant.
    bar_plot(runs, "method", "test_macro_f1", os.path.join(plot_dir, "overall_macro_f1_by_method.png"), "Overall Macro-F1 by method", group="threshold")
    bar_plot(runs, "method", "test_auroc", os.path.join(plot_dir, "overall_auroc_by_method.png"), "Overall AUROC by method", group="threshold")
    bar_plot(runs, "method", "test_auprc", os.path.join(plot_dir, "overall_auprc_by_method.png"), "Overall AUPRC by method", group="threshold")
    bar_plot(runs, "method", "num_features_union_dim", os.path.join(plot_dir, "feature_dim_by_method.png"), "Union feature dimension by method", group="threshold")

    if len(lang_metrics) > 0:
        bar_plot(lang_metrics, "method", "macro_f1", os.path.join(plot_dir, "language_macro_f1_by_method.png"), "Per-language Macro-F1 by method", group="lang")
        bar_plot(lang_metrics, "method", "recall_unsafe", os.path.join(plot_dir, "language_unsafe_recall_by_method.png"), "Per-language unsafe recall by method", group="lang")

    if len(layer_dist) > 0:
        for method in sorted(layer_dist["method"].unique()):
            sub = layer_dist[layer_dist["method"] == method]
            line_plot(sub, "layer_idx", "n_selected_neurons", "lang", os.path.join(plot_dir, f"layer_selected_neurons_{method}.png"), f"Layer distribution: {method}", "mean selected/active neurons")
            line_plot(sub, "layer_idx", "abs_sum", "lang", os.path.join(plot_dir, f"layer_abs_sum_{method}.png"), f"Layer abs activation sum: {method}", "mean abs_sum")

    print(f"Saved analysis to {args.out_dir}")
    print("Top runs by test_macro_f1:")
    cols = ["method", "threshold", "pooling_type", "seed", "test_macro_f1", "test_auroc", "test_auprc", "num_features_union_dim"]
    print(runs.sort_values("test_macro_f1", ascending=False)[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
