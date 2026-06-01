#!/usr/bin/env python
"""Validate every enabled dataset in configs/exp_multilingual_siren.yaml before an expensive SIREN run.

This script checks three things:
1. The configured Hugging Face/local dataset and split can be loaded.
2. The configured text/label/lang candidates match the actual schema.
3. The full requested split can be normalized into binary safe/unsafe rows.

It writes both JSON and CSV summaries so failed schemas can be fixed before
representation extraction, probe fitting, and MLP training begin.
"""

import argparse
import json
import os
import re
import sys
import traceback
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml
from datasets import Dataset, load_dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.multilingual_preprocess import (  # noqa: E402
    first_present,
    format_text,
    infer_label,
    infer_lang,
    infer_category,
    infer_paired_category,
    iter_rows_from_local,
    normalize_paired_saferlhf_row,
    normalize_row,
    norm_lang,
)


def template_fields(template: Optional[str]) -> List[str]:
    if not template:
        return []
    return re.findall(r"{([^{}]+)}", template)


def has_any(columns: Iterable[str], candidates: Iterable[str]) -> bool:
    cols = set(columns)
    return any(c in cols for c in candidates or [])


def load_hf_split(spec: Dict[str, Any], split: str, cache_dir: Optional[str]) -> Dataset:
    hf_id = spec["hf_id"]
    hf_name = spec.get("hf_name")
    if hf_name:
        return load_dataset(hf_id, hf_name, split=split, cache_dir=cache_dir)
    return load_dataset(hf_id, split=split, cache_dir=cache_dir)


def get_columns_for_local(spec: Dict[str, Any], split: Optional[str]) -> List[str]:
    for row in iter_rows_from_local(spec, split):
        return list(row.keys())
    return []


def schema_checks(spec: Dict[str, Any], columns: List[str]) -> List[str]:
    warnings = []
    cols = set(columns)

    if spec.get("paired_response_mode"):
        if not has_any(cols, spec.get("text_candidates", ["prompt", "text"])):
            warnings.append("no prompt/text candidate found for paired_response_mode")
        if not has_any(cols, spec.get("response0_candidates", ["response_0"])):
            warnings.append("no response0 candidate found")
        if not has_any(cols, spec.get("response1_candidates", ["response_1"])):
            warnings.append("no response1 candidate found")
        if not has_any(cols, spec.get("label0_candidates", ["is_response_0_safe"])):
            warnings.append("no label0 candidate found")
        if not has_any(cols, spec.get("label1_candidates", ["is_response_1_safe"])):
            warnings.append("no label1 candidate found")
    else:
        text_ok = has_any(cols, spec.get("text_candidates", ["text", "prompt", "query", "instruction"]))
        # A template can refer directly to raw columns. Response fields are optional, but at least one
        # text-like field or one template field should be present.
        if spec.get("text_template"):
            t_fields = template_fields(spec.get("text_template"))
            text_ok = text_ok or any(f in cols for f in t_fields if f not in {"response"})
        if not text_ok:
            warnings.append("no text/template candidate found")
        if not has_any(cols, spec.get("label_candidates", [])):
            warnings.append("no label candidate found")

    if not spec.get("lang") and not has_any(cols, spec.get("lang_candidates", ["lang", "language", "locale"])):
        warnings.append("no language candidate found and no fixed lang is configured")
    return warnings


def scan_rows(spec: Dict[str, Any], rows: Iterable[Dict[str, Any]], split_name: str, max_rows: Optional[int]) -> Dict[str, Any]:
    raw_n = 0
    norm_n = 0
    missing_text = 0
    missing_label = 0
    missing_lang = 0
    filtered_lang = 0
    label_counts = Counter()
    lang_counts = Counter()
    category_counts = Counter()
    examples = []

    keep_langs = spec.get("keep_langs")
    keep = {norm_lang(x) for x in keep_langs} if keep_langs else None

    for row_id, row in enumerate(rows):
        if max_rows is not None and raw_n >= max_rows:
            break
        raw_n += 1

        if spec.get("paired_response_mode"):
            normed_list = normalize_paired_saferlhf_row(row, spec, spec["name"], split_name, row_id)
            # Count coarse missing signals for diagnostics.
            _, prompt = first_present(row, spec.get("text_candidates", ["prompt", "text"]))
            if prompt is None:
                missing_text += 1
            for idx in [0, 1]:
                _, resp = first_present(row, spec.get(f"response{idx}_candidates", [f"response_{idx}"]))
                _, lab = first_present(row, spec.get(f"label{idx}_candidates", [f"is_response_{idx}_safe"]))
                if resp is None:
                    missing_text += 1
                if lab is None:
                    missing_label += 1
        else:
            text = format_text(row, spec)
            label = infer_label(row, spec)
            lang = infer_lang(row, spec)
            if text is None:
                missing_text += 1
            if label is None:
                missing_label += 1
            if lang is None:
                missing_lang += 1
            if keep and lang is not None and lang not in keep:
                filtered_lang += 1
            nr = normalize_row(row, spec, spec["name"], split_name, row_id)
            normed_list = [nr] if nr is not None else []

        for nr in normed_list:
            if nr is None:
                continue
            norm_n += 1
            label_counts[int(nr["label"])] += 1
            lang_counts[str(nr["lang"])] += 1
            cat = nr.get("category") or "<none>"
            category_counts[str(cat)] += 1
            if len(examples) < 2:
                examples.append({
                    "text_preview": str(nr["text"])[:200],
                    "label": int(nr["label"]),
                    "lang": str(nr["lang"]),
                    "category": nr.get("category"),
                })

    return {
        "raw_rows_scanned": raw_n,
        "normalized_rows": norm_n,
        "missing_text_signals": missing_text,
        "missing_label_signals": missing_label,
        "missing_lang_signals": missing_lang,
        "filtered_by_lang": filtered_lang,
        "label_counts": dict(sorted(label_counts.items())),
        "lang_counts": dict(sorted(lang_counts.items())),
        "top_categories": dict(category_counts.most_common(10)),
        "examples": examples,
    }


def validate_dataset(spec: Dict[str, Any], cache_dir: Optional[str], max_rows: Optional[int]) -> Dict[str, Any]:
    result = {
        "name": spec["name"],
        "source": spec.get("source", "hf"),
        "hf_id": spec.get("hf_id"),
        "hf_name": spec.get("hf_name"),
        "path": spec.get("path"),
        "ok": True,
        "errors": [],
        "warnings": [],
        "splits": [],
    }

    split_map = spec.get("split_map", {"train": "train", "validation": "validation", "test": "test"})
    requested = [(target, source) for target, source in split_map.items() if source is not None]
    if not requested:
        result["ok"] = False
        result["errors"].append("no non-null split is configured")
        return result

    total_norm = 0
    total_labels = Counter()
    total_langs = Counter()

    for target_split, source_split in requested:
        split_result = {
            "target_split": target_split,
            "source_split": source_split,
            "columns": [],
            "num_rows_reported": None,
            "schema_warnings": [],
        }
        try:
            if spec.get("source", "hf") == "hf":
                ds = load_hf_split(spec, source_split, cache_dir)
                columns = list(ds.column_names)
                split_result["columns"] = columns
                split_result["num_rows_reported"] = len(ds) if hasattr(ds, "__len__") else None
                split_result["schema_warnings"] = schema_checks(spec, columns)
                scan = scan_rows(spec, (dict(r) for r in ds), target_split, max_rows)
            else:
                rows = list(iter_rows_from_local(spec, source_split))
                columns = list(rows[0].keys()) if rows else []
                split_result["columns"] = columns
                split_result["num_rows_reported"] = len(rows)
                split_result["schema_warnings"] = schema_checks(spec, columns)
                scan = scan_rows(spec, rows, target_split, max_rows)

            split_result.update(scan)
            total_norm += int(scan["normalized_rows"])
            total_labels.update({int(k): int(v) for k, v in scan["label_counts"].items()})
            total_langs.update({str(k): int(v) for k, v in scan["lang_counts"].items()})

            if split_result["schema_warnings"]:
                result["warnings"].extend([f"{target_split}/{source_split}: {w}" for w in split_result["schema_warnings"]])
            if scan["normalized_rows"] == 0:
                result["errors"].append(f"{target_split}/{source_split}: normalized 0 rows")
        except Exception as e:
            result["ok"] = False
            result["errors"].append(f"{target_split}/{source_split}: {type(e).__name__}: {e}")
            split_result["exception"] = traceback.format_exc()
        result["splits"].append(split_result)

    if total_norm == 0:
        result["ok"] = False
        result["errors"].append("dataset normalized 0 rows across all configured splits")
    if total_norm > 0 and len(total_labels) < 2:
        result["warnings"].append(f"only one binary label found after normalization: {dict(total_labels)}")
    if total_norm > 0 and len(total_langs) == 0:
        result["warnings"].append("no languages found after normalization")
    if result["errors"]:
        result["ok"] = False
    return result


def flatten_for_csv(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        for s in r.get("splits", []):
            rows.append({
                "dataset": r["name"],
                "source": r.get("source"),
                "hf_id": r.get("hf_id"),
                "hf_name": r.get("hf_name"),
                "target_split": s.get("target_split"),
                "source_split": s.get("source_split"),
                "ok": r.get("ok"),
                "num_rows_reported": s.get("num_rows_reported"),
                "raw_rows_scanned": s.get("raw_rows_scanned"),
                "normalized_rows": s.get("normalized_rows"),
                "label_counts": json.dumps(s.get("label_counts", {}), ensure_ascii=False, sort_keys=True),
                "lang_counts": json.dumps(s.get("lang_counts", {}), ensure_ascii=False, sort_keys=True),
                "columns": ",".join(s.get("columns", [])),
                "schema_warnings": " | ".join(s.get("schema_warnings", [])),
                "dataset_errors": " | ".join(r.get("errors", [])),
                "dataset_warnings": " | ".join(r.get("warnings", [])),
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", default="outputs/multilingual_siren/_dataset_schema_check")
    p.add_argument("--max_rows_per_split", type=int, default=None,
                   help="Debug only. By default, scans all rows in every configured split.")
    p.add_argument("--fail_on_error", action="store_true")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cache_dir = config.get("cache_dir")

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for spec in config.get("datasets", []):
        if not spec.get("enabled", True):
            continue
        print(f"\n[CHECK] {spec['name']} ({spec.get('hf_id') or spec.get('path')})")
        res = validate_dataset(spec, cache_dir, args.max_rows_per_split)
        results.append(res)
        status = "OK" if res["ok"] else "FAIL"
        print(f"  status={status}")
        for split in res.get("splits", []):
            print(
                f"  {split.get('target_split')}<-{split.get('source_split')}: "
                f"rows={split.get('num_rows_reported')} normalized={split.get('normalized_rows')} "
                f"labels={split.get('label_counts')} langs={split.get('lang_counts')}"
            )
        for w in res.get("warnings", []):
            print(f"  [WARN] {w}")
        for e in res.get("errors", []):
            print(f"  [ERROR] {e}")

    json_path = os.path.join(args.out_dir, "dataset_schema_report.json")
    csv_path = os.path.join(args.out_dir, "dataset_schema_report.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    flatten_for_csv(results).to_csv(csv_path, index=False)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")

    failed = [r["name"] for r in results if not r.get("ok")]
    if failed:
        print(f"\nFailed datasets: {failed}")
        if args.fail_on_error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
