#!/usr/bin/env python
"""Report completion status for the configured Nemotron shared_only subsets."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import yaml


MODEL_ALIASES = {"llama3-8b": "llama3.1-8b", "llama-3.1-8b": "llama3.1-8b"}


def normalize_subset(subset: Sequence[str], languages: Sequence[str]) -> List[str]:
    requested = [str(x) for x in subset]
    unknown = sorted(set(requested) - set(languages))
    if unknown:
        raise ValueError(f"Unknown languages in subset: {unknown}")
    return [lang for lang in languages if lang in set(requested)]


def resolve_subsets(config: Dict[str, Any]) -> List[List[str]]:
    languages = [str(x) for x in config["languages"]]
    plan = config.get("shared_only_subset_plan") or {}
    explicit = config.get("shared_only_subsets")
    raw: List[Sequence[str]] = []
    if explicit:
        raw.extend(explicit)
    else:
        if plan.get("all_pairs"):
            raw.extend(itertools.combinations(languages, 2))
        raw.extend(plan.get("triples") or [])
        if plan.get("leave_one_out"):
            raw.extend([[x for x in languages if x != excluded] for excluded in languages])
        if plan.get("include_all"):
            raw.append(languages)
    out, seen = [], set()
    for subset in raw:
        normalized = normalize_subset(subset, languages)
        key = tuple(normalized)
        if key not in seen:
            seen.add(key)
            out.append(normalized)
    expected = plan.get("expected_count")
    if expected is not None and len(out) != int(expected):
        raise ValueError(f"Expected {expected} subsets, got {len(out)}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/exp_nemotron_shared_subsets.yaml")
    p.add_argument("--root", default="outputs/nemotron_primary")
    p.add_argument("--models", nargs="+", default=["qwen3-4b", "llama3.1-8b"])
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--pooling", default="residual_mean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", default=None)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    subsets = resolve_subsets(config)
    required = ["metrics.json", "best_model.pkl", "feature_manifest.json", "predictions.jsonl", "selected_neurons.json"]

    rows = []
    for requested_model in args.models:
        model = MODEL_ALIASES.get(requested_model, requested_model)
        for subset in subsets:
            slug = "-".join(subset)
            run_dir = (
                Path(args.root) / model / "method=shared_only" / f"selection_langs={slug}"
                / f"threshold={args.threshold}" / f"pooling={args.pooling}" / f"seed={args.seed}"
            )
            missing = [name for name in required if not (run_dir / name).exists()]
            if not missing:
                status = "complete"
            elif (run_dir / "skipped.json").exists():
                status = "skipped"
            elif run_dir.exists():
                status = "partial"
            else:
                status = "missing"
            rows.append({
                "model": model,
                "selection_langs": slug,
                "n_selection_languages": len(subset),
                "status": status,
                "missing_files": ",".join(missing),
                "run_dir": str(run_dir),
            })

    df = pd.DataFrame(rows)
    for model, group in df.groupby("model"):
        counts = group["status"].value_counts().to_dict()
        print(f"{model}: {counts} / expected={len(subsets)}")
        remaining = group[group["status"].isin(["missing", "partial"])]["selection_langs"].tolist()
        if remaining:
            print("  remaining:", " ".join(remaining))
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
