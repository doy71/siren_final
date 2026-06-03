#!/usr/bin/env python
"""Evaluate trained multilingual SIREN runs on an external dataset and measure latency.

Designed for the Nemotron-primary experiment:
  - train/evaluate on Nemotron with train/train_multilingual_siren_selection.py
  - externally validate each saved run on Lumees with this script
  - measure post-backbone latency per method: feature aggregation + MLP classifier

Latency note:
  This measures latency after backbone representation extraction. Backbone forward time is
  shared across all selection methods and usually dominates absolute runtime, so this script
  is intended to compare selection-method overheads fairly.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from train.multilingual_preprocess import build_normalized_dataset, save_dataset_artifacts
from train.train_multilingual_siren_selection import (
    AdaptiveMLPClassifier,
    FeatureSpec,
    aggregate_features,
    compute_metrics,
    dataset_signature_from_dfs,
    extract_representations,
    per_group_metrics,
    predict_scores,
)


def load_feature_manifest(path: str) -> List[FeatureSpec]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return [FeatureSpec(**r) for r in rows]


def build_model_from_checkpoint(ckpt: Dict[str, Any], device: torch.device) -> AdaptiveMLPClassifier:
    params = ckpt["best_params"]
    input_dim = int(ckpt["feature_dim"])
    n_layers = int(params["n_layers"])
    dims = [int(params[f"hidden_dim_layer{i}"]) for i in range(n_layers)]
    drops = [float(params[f"dropout_layer{i}"]) for i in range(n_layers)]
    model = AdaptiveMLPClassifier(input_dim, dims, drops).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ckpt["model_state_dict"].items()})
    model.eval()
    return model


def measure_classifier_latency(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    repeats: int,
    warmup: int,
) -> Dict[str, float]:
    if len(X) == 0:
        return {"classifier_total_s_mean": 0.0, "classifier_ms_per_sample_mean": 0.0}
    n = len(X)

    def one_pass() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = predict_scores(model, X, device, batch_size=batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(max(0, warmup)):
        one_pass()
    vals = [one_pass() for _ in range(max(1, repeats))]
    return {
        "classifier_total_s_mean": float(np.mean(vals)),
        "classifier_total_s_std": float(np.std(vals)),
        "classifier_ms_per_sample_mean": float(np.mean(vals) * 1000.0 / n),
        "classifier_ms_per_sample_std": float(np.std(vals) * 1000.0 / n),
    }


def run_dir_metadata(run_dir: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for part in Path(run_dir).parts:
        if "=" in part:
            k, v = part.split("=", 1)
            meta[k] = v
    if "threshold" in meta:
        try:
            meta["threshold"] = float(meta["threshold"])
        except Exception:
            pass
    if "seed" in meta:
        try:
            meta["seed"] = int(meta["seed"])
        except Exception:
            pass
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Primary experiment run dir, e.g. outputs/nemotron_primary/qwen3-4b")
    p.add_argument("--external_config", required=True, help="External validation dataset config, e.g. configs/exp_lumees_external_validation.yaml")
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for representation extraction")
    p.add_argument("--classifier_batch_size", type=int, default=2048)
    p.add_argument("--latency_repeats", type=int, default=5)
    p.add_argument("--latency_warmup", type=int, default=2)
    p.add_argument("--latency_max_samples", type=int, default=5000)
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--force_reextract", action="store_true")
    p.add_argument("--skip_failed_datasets", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.external_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Use a stable external cache under the external-analysis directory.
    ext_cache_dir = out_dir / "external_reps" / args.model
    ext_cache_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Build external validation dataset")
    dfs = build_normalized_dataset(config, skip_failed=args.skip_failed_datasets)
    for split, df in dfs.items():
        print(f"{split:10s}: {len(df):7d} | langs={sorted(df['lang'].unique().tolist()) if len(df) else []}")
    save_dataset_artifacts(dfs, str(ext_cache_dir))

    # Determine pooling types actually needed from run directories.
    run_metric_paths = glob.glob(os.path.join(args.run_dir, "method=*", "threshold=*", "pooling=*", "seed=*", "metrics.json"))
    if not run_metric_paths:
        raise FileNotFoundError(f"No trained runs found under {args.run_dir}")
    pooling_types = sorted({run_dir_metadata(os.path.dirname(p))["pooling"] for p in run_metric_paths})

    print(f"[2] Extract/reuse external representations for pooling={pooling_types}")
    dataset_sig = dataset_signature_from_dfs(dfs)
    with open(ext_cache_dir / "dataset_signature.json", "w", encoding="utf-8") as f:
        json.dump(dataset_sig, f, indent=2, ensure_ascii=False)
    all_reps = extract_representations(
        args.model,
        dfs,
        str(device),
        args.batch_size,
        pooling_types,
        str(ext_cache_dir),
        force_reextract=args.force_reextract,
        dataset_signature=dataset_sig,
    )

    target = all_reps[args.split]
    if len(target["labels"]) == 0:
        raise ValueError(f"External split {args.split!r} is empty")

    metrics_rows = []
    latency_rows = []
    pred_frames = []

    print("[3] Evaluate saved runs on external data and measure post-backbone latency")
    for metrics_path in sorted(run_metric_paths):
        rdir = os.path.dirname(metrics_path)
        meta = run_dir_metadata(rdir)
        method = meta.get("method")
        threshold = meta.get("threshold")
        pooling_type = meta.get("pooling")
        seed = meta.get("seed")

        ckpt_path = os.path.join(rdir, "best_model.pkl")
        manifest_path = os.path.join(rdir, "feature_manifest.json")
        if not os.path.exists(ckpt_path) or not os.path.exists(manifest_path):
            print(f"[WARN] skip incomplete run: {rdir}")
            continue

        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        model = build_model_from_checkpoint(ckpt, device)
        features = load_feature_manifest(manifest_path)
        layer_weights = {int(k): float(v) for k, v in ckpt.get("layer_weights", {}).items()}
        route_fallback = config.get("route_unknown_lang_to")

        # Feature aggregation latency is part of post-backbone method overhead.
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        X_ext = aggregate_features(target["representations"], target["langs"], features, layer_weights, method, route_fallback)
        if device.type == "cuda":
            torch.cuda.synchronize()
        aggregation_s = time.perf_counter() - t0

        pred, score = predict_scores(model, X_ext, device, batch_size=args.classifier_batch_size)
        pred_df = pd.DataFrame({
            "split": args.split,
            "id": target["source_ids"],
            "source_dataset": target["source_datasets"],
            "lang": target["langs"],
            "category": target["categories"],
            "label": target["labels"].astype(int),
            "pred": pred.astype(int),
            "score": score.astype(float),
            "method": method,
            "threshold": threshold,
            "pooling_type": pooling_type,
            "seed": seed,
        })
        pred_frames.append(pred_df)

        overall = compute_metrics(pred_df["label"].to_numpy(), pred_df["score"].to_numpy())
        row = {
            "method": method,
            "threshold": threshold,
            "pooling_type": pooling_type,
            "seed": seed,
            "external_split": args.split,
            "feature_dim": int(X_ext.shape[1]),
            **{f"external_{k}": v for k, v in overall.items()},
        }
        metrics_rows.append(row)

        lang_metrics = per_group_metrics(pred_df, "lang")
        lang_metrics["method"] = method
        lang_metrics["threshold"] = threshold
        lang_metrics["pooling_type"] = pooling_type
        lang_metrics["seed"] = seed
        lang_metrics.to_csv(out_dir / f"external_per_language_{method}_{threshold}_{pooling_type}_seed{seed}.csv", index=False)

        # Latency can be measured on a subset to avoid long repeated passes.
        if len(X_ext) > args.latency_max_samples:
            rng = np.random.default_rng(int(seed or 42))
            idx = rng.choice(len(X_ext), size=args.latency_max_samples, replace=False)
            X_lat = X_ext[idx]
        else:
            X_lat = X_ext
        lat = measure_classifier_latency(model, X_lat, device, args.classifier_batch_size, args.latency_repeats, args.latency_warmup)
        latency_rows.append({
            "method": method,
            "threshold": threshold,
            "pooling_type": pooling_type,
            "seed": seed,
            "feature_dim": int(X_ext.shape[1]),
            "n_external_samples": int(len(X_ext)),
            "n_latency_samples": int(len(X_lat)),
            "aggregation_total_s": float(aggregation_s),
            "aggregation_ms_per_sample": float(aggregation_s * 1000.0 / len(X_ext)),
            **lat,
            "post_backbone_ms_per_sample_mean": float(aggregation_s * 1000.0 / len(X_ext) + lat["classifier_ms_per_sample_mean"]),
        })

        print(f"  {method:28s} threshold={threshold} pooling={pooling_type} F1={overall['macro_f1']:.4f} latency={lat['classifier_ms_per_sample_mean']:.4f} ms/sample")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["pooling_type", "threshold", "method", "seed"])
    latency_df = pd.DataFrame(latency_rows).sort_values(["pooling_type", "threshold", "method", "seed"])
    preds_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()

    metrics_df.to_csv(out_dir / "external_metrics_summary.csv", index=False)
    latency_df.to_csv(out_dir / "latency_summary.csv", index=False)
    if len(preds_df):
        preds_df.to_json(out_dir / "external_predictions.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"\nSaved external metrics: {out_dir / 'external_metrics_summary.csv'}")
    print(f"Saved latency summary: {out_dir / 'latency_summary.csv'}")
    if len(preds_df):
        print(f"Saved external predictions: {out_dir / 'external_predictions.jsonl'}")


if __name__ == "__main__":
    main()
