import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split


LANG_ALIASES = {
    "english": "en", "eng": "en", "en-us": "en", "en_usa": "en",
    "korean": "ko", "kor": "ko", "kr": "ko",
    "french": "fr", "fra": "fr", "fre": "fr", "fr-fr": "fr",
}


def norm_lang(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower().replace("_", "-")
    return LANG_ALIASES.get(s, s[:2] if len(s) >= 2 else s)


def as_bool_label(value: Any, unsafe_values: Iterable[Any], safe_values: Iterable[Any]) -> Optional[int]:
    """Return 1 for unsafe/harmful, 0 for safe, None if unknown."""
    if value is None:
        return None
    unsafe_set = {str(v).strip().lower() for v in unsafe_values}
    safe_set = {str(v).strip().lower() for v in safe_values}
    s = str(value).strip().lower()
    if s in unsafe_set:
        return 1
    if s in safe_set:
        return 0
    # handle numeric severity/risk strings robustly
    try:
        f = float(s)
        if str(int(f)) in unsafe_set:
            return 1
        if str(int(f)) in safe_set:
            return 0
    except Exception:
        pass
    return None


def first_present(row: Dict[str, Any], candidates: Iterable[str]) -> Tuple[Optional[str], Any]:
    for c in candidates or []:
        if c in row and row[c] is not None:
            val = row[c]
            if isinstance(val, str) and val.strip() == "":
                continue
            return c, val
    return None, None


def format_text(row: Dict[str, Any], spec: Dict[str, Any]) -> Optional[str]:
    template = spec.get("text_template")
    text_candidates = spec.get("text_candidates", ["text", "prompt", "query", "instruction"])
    response_candidates = spec.get("response_candidates", ["response", "answer", "output", "completion"])
    _, text = first_present(row, text_candidates)
    _, response = first_present(row, response_candidates)

    if template:
        # Provide both canonical placeholders and raw row keys.
        fmt = dict(row)
        fmt.setdefault("text", text or "")
        fmt.setdefault("prompt", text or "")
        fmt.setdefault("response", response or "")
        try:
            rendered = template.format(**fmt)
        except Exception:
            rendered = "\n\n".join([str(x) for x in [text, response] if x is not None])
    else:
        rendered = "\n\n".join([str(x) for x in [text, response] if x is not None])
    rendered = rendered.strip()
    return rendered if rendered else None


def infer_label(row: Dict[str, Any], spec: Dict[str, Any]) -> Optional[int]:
    _, value = first_present(row, spec.get("label_candidates", []))
    return as_bool_label(value, spec.get("unsafe_values", []), spec.get("safe_values", []))


def infer_lang(row: Dict[str, Any], spec: Dict[str, Any]) -> Optional[str]:
    if spec.get("lang"):
        return norm_lang(spec["lang"])
    _, value = first_present(row, spec.get("lang_candidates", ["lang", "language", "locale"]))
    return norm_lang(value)


def stringify_category(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dict):
        active = [str(k) for k, v in value.items() if bool(v)]
        if active:
            return ";".join(active)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        return ";".join(map(str, value))
    return str(value)


def infer_category(row: Dict[str, Any], spec: Dict[str, Any]) -> Optional[str]:
    _, value = first_present(row, spec.get("category_candidates", []))
    return stringify_category(value)


def infer_paired_category(row: Dict[str, Any], spec: Dict[str, Any], idx: int) -> Optional[str]:
    candidates = spec.get(f"response{idx}_category_candidates", [])
    _, value = first_present(row, candidates)
    if value is None:
        return infer_category(row, spec)
    return stringify_category(value)


def iter_rows_from_hf(spec: Dict[str, Any], split: str, cache_dir: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    hf_id = spec["hf_id"]
    hf_name = spec.get("hf_name")
    if hf_name:
        ds = load_dataset(hf_id, hf_name, split=split, cache_dir=cache_dir)
    else:
        ds = load_dataset(hf_id, split=split, cache_dir=cache_dir)
    for r in ds:
        yield dict(r)


def iter_rows_from_local(spec: Dict[str, Any], split: Optional[str]) -> Iterable[Dict[str, Any]]:
    path = spec["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if split is None or row.get("split") == split:
                        yield row
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if split in data:
                data = data[split]
            elif "data" in data:
                data = data["data"]
            else:
                data = list(data.values())
        for row in data:
            if split is None or row.get("split") in (None, split):
                yield row
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split]
        for row in df.to_dict(orient="records"):
            yield row
    else:
        raise ValueError(f"Unsupported local file type: {path}")


def normalize_paired_saferlhf_row(row: Dict[str, Any], spec: Dict[str, Any], source_name: str, split_name: str, row_id: int) -> List[Dict[str, Any]]:
    _, prompt = first_present(row, spec.get("text_candidates", ["prompt", "text"]))
    out = []
    for idx in [0, 1]:
        _, resp = first_present(row, spec.get(f"response{idx}_candidates", [f"response_{idx}", f"answer_{idx}"]))
        _, lab = first_present(row, spec.get(f"label{idx}_candidates", [f"is_response_{idx}_safe"]))
        label = as_bool_label(lab, spec.get("unsafe_values", []), spec.get("safe_values", []))
        if prompt is None or resp is None or label is None:
            continue
        lang = infer_lang(row, spec)
        if lang is None:
            continue
        keep_langs = spec.get("keep_langs")
        if keep_langs and lang not in {norm_lang(x) for x in keep_langs}:
            continue
        text = f"{prompt}\n\n{resp}".strip()
        out.append({
            "text": text,
            "label": int(label),
            "lang": lang,
            "source_dataset": source_name,
            "source_split": split_name,
            "source_id": f"{row_id}_{idx}",
            "category": infer_paired_category(row, spec, idx),
        })
    return out


def normalize_row(row: Dict[str, Any], spec: Dict[str, Any], source_name: str, split_name: str, row_id: int) -> Optional[Dict[str, Any]]:
    text = format_text(row, spec)
    label = infer_label(row, spec)
    lang = infer_lang(row, spec)
    if text is None or label is None or lang is None:
        return None
    keep_langs = spec.get("keep_langs")
    if keep_langs and lang not in {norm_lang(x) for x in keep_langs}:
        return None
    return {
        "text": text,
        "label": int(label),
        "lang": lang,
        "source_dataset": source_name,
        "source_split": split_name,
        "source_id": str(row.get("id", row.get("uid", row_id))),
        "category": infer_category(row, spec),
    }


def stratify_arg(df: pd.DataFrame):
    if len(df) == 0 or "label" not in df or "lang" not in df:
        return None
    strat = df["label"].astype(str) + "_" + df["lang"].astype(str)
    return strat if strat.value_counts().min() >= 2 else None


def stratified_train_test_split(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) == 0:
        return df.copy(), df.copy()
    if len(df) < 2 or test_size <= 0.0:
        return df.reset_index(drop=True), df.iloc[0:0].copy().reset_index(drop=True)
    test_size = min(max(float(test_size), 1.0 / len(df)), 0.5 if len(df) < 10 else 0.9)
    a, b = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify_arg(df))
    return a.reset_index(drop=True), b.reset_index(drop=True)


def split_dataframe(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> Dict[str, pd.DataFrame]:
    if len(df) == 0:
        return {"train": df, "validation": df, "test": df}
    train_val, test = stratified_train_test_split(df, test_ratio, seed)
    val_size = val_ratio / max(1e-9, 1.0 - test_ratio)
    train, val = stratified_train_test_split(train_val, val_size, seed)
    return {"train": train.reset_index(drop=True), "validation": val.reset_index(drop=True), "test": test.reset_index(drop=True)}


def complete_missing_splits(per_dataset_rows: Dict[str, List[Dict[str, Any]]], val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    """Create missing validation/test splits without unnecessarily reshuffling existing held-out splits.

    Previous versions merged train/validation/test and re-split whenever either
    validation or test was missing. That could leak an explicit HF test split back
    into training. This function only splits the available train split when
    possible. If a dataset provides only validation/test, it falls back to a
    deterministic train/validation/test split over the available rows.
    """
    n_train = len(per_dataset_rows.get("train", []))
    n_val = len(per_dataset_rows.get("validation", []))
    n_test = len(per_dataset_rows.get("test", []))
    if n_train == 0:
        base_rows = []
        for s in ["train", "validation", "test"]:
            base_rows.extend(per_dataset_rows.get(s, []))
        if not base_rows:
            return per_dataset_rows
        df = pd.DataFrame(base_rows).drop_duplicates(subset=["text", "label", "lang"])
        split_dfs = split_dataframe(df, val_ratio, test_ratio, seed)
        return {k: v.to_dict(orient="records") for k, v in split_dfs.items()}

    if n_val > 0 and n_test > 0:
        return per_dataset_rows

    train_df = pd.DataFrame(per_dataset_rows["train"]).drop_duplicates(subset=["text", "label", "lang"])
    if n_val == 0 and n_test == 0:
        split_dfs = split_dataframe(train_df, val_ratio, test_ratio, seed)
        per_dataset_rows["train"] = split_dfs["train"].to_dict(orient="records")
        per_dataset_rows["validation"] = split_dfs["validation"].to_dict(orient="records")
        per_dataset_rows["test"] = split_dfs["test"].to_dict(orient="records")
        return per_dataset_rows

    if n_val == 0:
        train_df, val_df = stratified_train_test_split(train_df, val_ratio, seed)
        per_dataset_rows["train"] = train_df.to_dict(orient="records")
        per_dataset_rows["validation"] = val_df.to_dict(orient="records")
        return per_dataset_rows

    if n_test == 0:
        train_df, test_df = stratified_train_test_split(train_df, test_ratio, seed)
        per_dataset_rows["train"] = train_df.to_dict(orient="records")
        per_dataset_rows["test"] = test_df.to_dict(orient="records")
        return per_dataset_rows

    return per_dataset_rows


def maybe_balance(df: pd.DataFrame, seed: int, by: List[str]) -> pd.DataFrame:
    if len(df) == 0:
        return df
    rng = np.random.default_rng(seed)
    groups = []
    for _, g in df.groupby(by):
        groups.append(g)
    min_n = min(len(g) for g in groups)
    if min_n <= 0:
        return df
    sampled = [g.sample(n=min_n, random_state=int(rng.integers(0, 1_000_000))) for g in groups]
    return pd.concat(sampled, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_normalized_dataset(config: Dict[str, Any], skip_failed: bool = True) -> Dict[str, pd.DataFrame]:
    all_by_split = {"train": [], "validation": [], "test": []}
    seed = int(config.get("seed", 42))
    cache_dir = config.get("cache_dir")
    val_ratio = float(config.get("val_ratio_if_no_validation", 0.2))
    test_ratio = float(config.get("test_ratio_if_no_test", 0.1))
    max_per_split = config.get("max_samples_per_dataset_split")

    for spec in config.get("datasets", []):
        if not spec.get("enabled", True):
            continue
        source_name = spec["name"]
        split_map = spec.get("split_map", {"train": "train", "validation": "validation", "test": "test"})
        source = spec.get("source", "hf")
        per_dataset_rows = {"train": [], "validation": [], "test": []}
        try:
            for target_split in ["train", "validation", "test"]:
                source_split = split_map.get(target_split)
                if source_split is None:
                    continue
                rows_iter = iter_rows_from_hf(spec, source_split, cache_dir) if source == "hf" else iter_rows_from_local(spec, source_split)
                for row_id, row in enumerate(rows_iter):
                    if spec.get("paired_response_mode"):
                        normed_list = normalize_paired_saferlhf_row(row, spec, source_name, target_split, row_id)
                    else:
                        nr = normalize_row(row, spec, source_name, target_split, row_id)
                        normed_list = [nr] if nr is not None else []
                    per_dataset_rows[target_split].extend(normed_list)
                    if max_per_split and len(per_dataset_rows[target_split]) >= int(max_per_split):
                        break
        except Exception as e:
            if skip_failed:
                print(f"[WARN] failed loading {source_name}: {type(e).__name__}: {e}")
                continue
            raise

        # Create missing validation/test splits. Keep explicit held-out splits intact whenever possible.
        if len(per_dataset_rows["validation"]) == 0 or len(per_dataset_rows["test"]) == 0 or len(per_dataset_rows["train"]) == 0:
            per_dataset_rows = complete_missing_splits(per_dataset_rows, val_ratio, test_ratio, seed)

        for split, rows in per_dataset_rows.items():
            all_by_split[split].extend(rows)
            print(f"Loaded {source_name:28s} {split:10s}: {len(rows):7d}")

    dfs = {}
    for split, rows in all_by_split.items():
        df = pd.DataFrame(rows)
        if len(df) == 0:
            dfs[split] = df
            continue
        df = df.dropna(subset=["text", "label", "lang"]).drop_duplicates(subset=["text", "label", "lang", "source_dataset"])
        languages = config.get("languages")
        if languages:
            keep = {norm_lang(x) for x in languages}
            df = df[df["lang"].isin(keep)]
        if config.get("balance_per_dataset", False):
            # Preserve SIREN's per-dataset averaging idea by avoiding one dataset/class/language dominating.
            by_cols = ["source_dataset", "lang", "label"]
            valid_groups = df.groupby(by_cols).size()
            if len(valid_groups) > 0 and valid_groups.min() > 0:
                df = maybe_balance(df, seed, by_cols)
        dfs[split] = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return dfs


def save_dataset_artifacts(dfs: Dict[str, pd.DataFrame], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for split, df in dfs.items():
        if len(df) == 0:
            continue
        tmp = df.copy()
        tmp["split"] = split
        all_rows.append(tmp)
    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_df.to_json(os.path.join(out_dir, "normalized_dataset.jsonl"), orient="records", lines=True, force_ascii=False)
        summary = (
            all_df.groupby(["split", "source_dataset", "lang", "label"])
            .size().reset_index(name="n")
            .sort_values(["split", "source_dataset", "lang", "label"])
        )
        summary.to_csv(os.path.join(out_dir, "dataset_summary.csv"), index=False)
