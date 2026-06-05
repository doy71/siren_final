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


def _path_metadata(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for part in os.path.normpath(path).split(os.sep):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def _selection_fields(m: Dict[str, Any], path: str) -> Dict[str, Any]:
    path_meta = _path_metadata(path)
    selection_langs = m.get("selection_langs") or path_meta.get("selection_langs") or ""
    selection_languages = m.get("selection_languages") or ([x for x in str(selection_langs).split("-") if x])
    # Legacy shared_only runs used all configured language probes but did not
    # store subset metadata. Infer those languages from active_feature_counts.
    if not selection_languages and m.get("method") == "shared_only":
        selection_languages = [str(k) for k in (m.get("active_feature_counts") or {}).keys() if str(k) != "union_dim"]
        selection_langs = "-".join(selection_languages)
    excluded = m.get("excluded_languages") or []
    return {
        "selection_langs": str(selection_langs),
        "selection_languages": ",".join(str(x) for x in selection_languages),
        "n_selection_languages": int(m.get("n_selection_languages") or len(selection_languages) or 0),
        "excluded_languages": ",".join(str(x) for x in excluded),
    }


def load_runs(run_dir: str) -> pd.DataFrame:
    rows = []
    paths = glob.glob(os.path.join(run_dir, "method=*", "**", "metrics.json"), recursive=True)
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        row = {
            "run_path": os.path.dirname(path),
            "explicit_subset_path": any(part.startswith("selection_langs=") for part in os.path.normpath(path).split(os.sep)),
            "model": m.get("model"),
            "method": m.get("method"),
            **_selection_fields(m, path),
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
                "selection_langs": r.get("selection_langs", ""),
                "n_selection_languages": r.get("n_selection_languages", 0),
                "excluded_languages": r.get("excluded_languages", ""),
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
                "selection_langs": r.get("selection_langs", ""),
                "n_selection_languages": r.get("n_selection_languages", 0),
                "excluded_languages": r.get("excluded_languages", ""),
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
            df["selection_langs"] = r.get("selection_langs", "")
            df["n_selection_languages"] = r.get("n_selection_languages", 0)
            df["excluded_languages"] = r.get("excluded_languages", "")
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


def subset_size_summary(shared_runs: pd.DataFrame) -> pd.DataFrame:
    if len(shared_runs) == 0:
        return pd.DataFrame()
    value_cols = [
        c for c in [
            "test_macro_f1", "test_unsafe_f1", "test_balanced_accuracy", "test_mcc",
            "test_auroc", "test_auprc", "num_features_union_dim",
        ] if c in shared_runs.columns
    ]
    grouped = shared_runs.groupby(["model", "n_selection_languages", "threshold", "pooling_type"])[value_cols].agg(["mean", "std", "min", "max", "count"]).reset_index()
    grouped.columns = ["_".join([str(x) for x in col if x != ""]).rstrip("_") for col in grouped.columns]
    return grouped


def language_inclusion_effects(shared_runs: pd.DataFrame) -> pd.DataFrame:
    if len(shared_runs) == 0:
        return pd.DataFrame()
    all_languages = sorted({lang for slug in shared_runs["selection_langs"].astype(str) for lang in slug.split("-") if lang})
    rows = []
    for (model, n_langs, threshold, pooling), base in shared_runs.groupby(["model", "n_selection_languages", "threshold", "pooling_type"]):
        for lang in all_languages:
            included_mask = base["selection_langs"].astype(str).apply(lambda x: lang in x.split("-"))
            included = base[included_mask]
            excluded = base[~included_mask]
            if len(included) == 0 or len(excluded) == 0:
                continue
            row = {
                "model": model,
                "n_selection_languages": n_langs,
                "threshold": threshold,
                "pooling_type": pooling,
                "language": lang,
                "n_included_runs": int(len(included)),
                "n_excluded_runs": int(len(excluded)),
            }
            for col in ["test_macro_f1", "test_unsafe_f1", "num_features_union_dim"]:
                if col in base.columns:
                    inc = float(included[col].mean())
                    exc = float(excluded[col].mean())
                    row[f"{col}_included_mean"] = inc
                    row[f"{col}_excluded_mean"] = exc
                    row[f"{col}_included_minus_excluded"] = inc - exc
            rows.append(row)
    return pd.DataFrame(rows)


def subset_scatter(df: pd.DataFrame, path: str) -> None:
    if len(df) == 0 or "test_macro_f1" not in df or "num_features_union_dim" not in df:
        return
    plt.figure(figsize=(10, 7))
    for n_langs, g in df.groupby("n_selection_languages"):
        plt.scatter(g["num_features_union_dim"], g["test_macro_f1"], label=f"{int(n_langs)} languages")
    for _, row in df.iterrows():
        plt.annotate(str(row["selection_langs"]), (row["num_features_union_dim"], row["test_macro_f1"]), fontsize=6, alpha=0.75)
    plt.xlabel("Shared feature dimension")
    plt.ylabel("Test macro-F1")
    plt.title("Shared-only subset: performance vs feature dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def subset_size_plot(df: pd.DataFrame, y: str, path: str, title: str) -> None:
    if len(df) == 0 or y not in df:
        return
    agg = df.groupby("n_selection_languages")[y].agg(["mean", "std"]).reset_index().sort_values("n_selection_languages")
    plt.figure(figsize=(8, 5))
    plt.errorbar(agg["n_selection_languages"], agg["mean"], yerr=agg["std"].fillna(0.0), marker="o", capsize=4)
    plt.xlabel("Number of languages used for shared intersection")
    plt.ylabel(y)
    plt.title(title)
    plt.xticks(sorted(df["n_selection_languages"].unique()))
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
    # Prefer the new explicit subset-path run over a legacy all-language
    # shared_only run when both describe the same experiment key.
    dedupe_keys = ["model", "method", "selection_langs", "threshold", "pooling_type", "seed"]
    runs = runs.sort_values(dedupe_keys + ["explicit_subset_path"]).drop_duplicates(dedupe_keys, keep="last")
    runs = runs.sort_values(["pooling_type", "threshold", "method", "n_selection_languages", "selection_langs", "seed"])
    save_table(runs, os.path.join(args.out_dir, "metrics_summary.csv"))

    metric_cols = [c for c in runs.columns if c.startswith("test_") and pd.api.types.is_numeric_dtype(runs[c])]
    group_keys = ["method", "selection_langs", "n_selection_languages", "threshold", "pooling_type"]
    agg = runs.groupby(group_keys)[metric_cols + ["num_features_union_dim"]].agg(["mean", "std"]).reset_index()
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
            layer_values.groupby(["method", "selection_langs", "n_selection_languages", "threshold", "pooling_type", "lang", "layer_idx"])
            .agg(n_selected_neurons=("n_selected_neurons", "mean"), abs_sum=("abs_sum", "mean"), l2=("l2", "mean"))
            .reset_index()
        )
        save_table(layer_dist, os.path.join(args.out_dir, "layer_distribution.csv"))
    else:
        layer_dist = pd.DataFrame()

    shared_runs = runs[runs["method"] == "shared_only"].copy()
    save_table(shared_runs, os.path.join(args.out_dir, "shared_only_by_subset.csv"))
    size_summary = subset_size_summary(shared_runs)
    save_table(size_summary, os.path.join(args.out_dir, "shared_only_by_subset_size.csv"))
    inclusion = language_inclusion_effects(shared_runs)
    save_table(inclusion, os.path.join(args.out_dir, "shared_only_language_inclusion_effect.csv"))

    # General plots retained for compatibility with previous experiments.
    bar_plot(runs, "method", "test_macro_f1", os.path.join(plot_dir, "overall_macro_f1_by_method.png"), "Overall Macro-F1 by method", group="threshold")
    bar_plot(runs, "method", "num_features_union_dim", os.path.join(plot_dir, "feature_dim_by_method.png"), "Union feature dimension by method", group="threshold")

    # Subset-specific plots answer the current experiment questions directly.
    subset_size_plot(shared_runs, "test_macro_f1", os.path.join(plot_dir, "shared_macro_f1_by_subset_size.png"), "Shared-only macro-F1 by subset size")
    subset_size_plot(shared_runs, "num_features_union_dim", os.path.join(plot_dir, "shared_feature_dim_by_subset_size.png"), "Shared feature dimension by subset size")
    subset_scatter(shared_runs, os.path.join(plot_dir, "shared_macro_f1_vs_feature_dim.png"))
    if len(shared_runs) > 0:
        bar_plot(shared_runs, "selection_langs", "test_macro_f1", os.path.join(plot_dir, "shared_macro_f1_by_subset.png"), "Shared-only macro-F1 by language subset")
        bar_plot(shared_runs, "selection_langs", "num_features_union_dim", os.path.join(plot_dir, "shared_feature_dim_by_subset.png"), "Shared feature dimension by language subset")

    print(f"Saved analysis to {args.out_dir}")
    print("Top runs by test_macro_f1:")
    cols = ["method", "selection_langs", "n_selection_languages", "threshold", "pooling_type", "seed", "test_macro_f1", "test_auroc", "num_features_union_dim"]
    cols = [c for c in cols if c in runs.columns]
    print(runs.sort_values("test_macro_f1", ascending=False)[cols].head(26).to_string(index=False))


if __name__ == "__main__":
    main()
