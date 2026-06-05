#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)


def parse_meta_from_path(path: Path):
    meta = {}
    for part in path.parts:
        if "=" in part:
            k, v = part.split("=", 1)
            meta[k] = v
    return meta


def read_jsonl(path: Path):
    meta = parse_meta_from_path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # path metadata fallback
            for k, v in meta.items():
                obj.setdefault(k, v)

            # normalize common names
            if "pooling" in obj and "pooling_type" not in obj:
                obj["pooling_type"] = obj["pooling"]
            if "dataset" in obj and "source_dataset" not in obj:
                obj["source_dataset"] = obj["dataset"]

            rows.append(obj)
    return rows


def safe_auroc(y_true, score):
    try:
        if score is None:
            return np.nan
        if len(set(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, score)
    except Exception:
        return np.nan


def summarize_group(g: pd.DataFrame):
    y = g["label"].astype(int).to_numpy()
    p = g["pred"].astype(int).to_numpy()

    score = None
    if "score" in g.columns:
        score = pd.to_numeric(g["score"], errors="coerce").to_numpy()
        if np.isnan(score).all():
            score = None

    labels = [0, 1]
    cm = confusion_matrix(y, p, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    out = {
        "n": len(g),
        "n_safe": int((y == 0).sum()),
        "n_unsafe": int((y == 1).sum()),
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "safe_f1": f1_score(y, p, labels=[0], average="macro", zero_division=0),
        "unsafe_f1": f1_score(y, p, labels=[1], average="macro", zero_division=0),
        "precision_safe": precision_score(y, p, pos_label=0, zero_division=0),
        "recall_safe": recall_score(y, p, pos_label=0, zero_division=0),
        "precision_unsafe": precision_score(y, p, pos_label=1, zero_division=0),
        "recall_unsafe": recall_score(y, p, pos_label=1, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "mcc": matthews_corrcoef(y, p) if len(set(y)) > 1 and len(set(p)) > 1 else np.nan,
        "auroc": safe_auroc(y, score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # latency가 raw에 있으면 자동 집계
    latency_cols = [c for c in g.columns if c.lower() in {
        "latency", "latency_ms", "elapsed", "elapsed_ms", "inference_time", "inference_time_ms"
    }]
    for c in latency_cols:
        vals = pd.to_numeric(g[c], errors="coerce")
        out[f"{c}_mean"] = vals.mean()
        out[f"{c}_median"] = vals.median()
        out[f"{c}_p95"] = vals.quantile(0.95)

    # selected neuron 수가 raw에 있으면 자동 집계
    neuron_cols = [c for c in g.columns if c in {
        "n_selected_neurons", "num_selected_neurons", "selected_neurons"
    }]
    for c in neuron_cols:
        vals = pd.to_numeric(g[c], errors="coerce")
        out[f"{c}_mean"] = vals.mean()
        out[f"{c}_median"] = vals.median()

    return pd.Series(out)


def make_summary(df, group_cols):
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return summarize_group(df).to_frame().T
    return (
        df.groupby(group_cols, dropna=False)
        .apply(summarize_group)
        .reset_index()
    )


def percent_table(df, index_cols, column_col="source_dataset", value_col="macro_f1"):
    needed = list(index_cols) + [column_col, value_col]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()

    table = df.pivot_table(
        index=index_cols,
        columns=column_col,
        values=value_col,
        aggfunc="mean",
    )

    table["Avg."] = table.mean(axis=1)
    table = table.reset_index()

    # 논문 표처럼 0~1이 아니라 0~100 점수로 표시
    metric_cols = [c for c in table.columns if c not in index_cols]
    for c in metric_cols:
        table[c] = table[c] * 100

    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="qwen output root directory")
    ap.add_argument("--out", type=str, required=True, help="summary output directory")
    ap.add_argument("--split", type=str, default="test", help="default: test")
    ap.add_argument("--include-random", action="store_true", help="include random methods if present")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(root.rglob("predictions.jsonl"))
    if not files:
        files = sorted(root.rglob("*.jsonl"))

    if not files:
        raise FileNotFoundError(f"No jsonl files found under {root}")

    rows = []
    for f in files:
        rows.extend(read_jsonl(f))

    df = pd.DataFrame(rows)

    required = {"label", "pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # test split 우선. test가 없으면 전체 사용.
    if "split" in df.columns and args.split:
        test_df = df[df["split"].astype(str) == args.split].copy()
        if len(test_df) > 0:
            df = test_df

    # random 제외 기본값
    if not args.include_random and "method" in df.columns:
        df = df[~df["method"].astype(str).str.contains("random", case=False, na=False)].copy()

    # 기본 metadata 없으면 채우기
    for c in ["method", "threshold", "pooling_type", "seed", "source_dataset", "lang"]:
        if c not in df.columns:
            df[c] = "NA"

    # raw 병합본 저장
    df.to_csv(out / "merged_predictions.csv", index=False)

    base_group = ["method", "threshold", "pooling_type", "seed"]

    overall = make_summary(df, base_group)
    overall.to_csv(out / "overall_summary.csv", index=False)

    by_dataset = make_summary(df, base_group + ["source_dataset"])
    by_dataset.to_csv(out / "by_dataset_summary.csv", index=False)

    by_lang = make_summary(df, base_group + ["lang"])
    by_lang.to_csv(out / "by_lang_summary.csv", index=False)

    by_dataset_lang = make_summary(df, base_group + ["source_dataset", "lang"])
    by_dataset_lang.to_csv(out / "by_dataset_lang_summary.csv", index=False)

    # SIREN Table 1 스타일: dataset별 Macro F1 + Avg.
    paper_table = percent_table(
        by_dataset,
        index_cols=base_group,
        column_col="source_dataset",
        value_col="macro_f1",
    )
    paper_table.to_csv(out / "paper_like_macro_f1_table.csv", index=False)

    # unsafe F1 버전도 같이 저장
    unsafe_table = percent_table(
        by_dataset,
        index_cols=base_group,
        column_col="source_dataset",
        value_col="unsafe_f1",
    )
    unsafe_table.to_csv(out / "paper_like_unsafe_f1_table.csv", index=False)

    print(f"[OK] loaded files: {len(files)}")
    print(f"[OK] rows used: {len(df)}")
    print(f"[OK] saved to: {out}")
    print("\nTop overall by macro_f1:")
    print(
        overall.sort_values("macro_f1", ascending=False)
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()