#!/usr/bin/env python
import argparse
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from tqdm import tqdm

# Run from CSSLab/SIREN repo root.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from train.multilingual_preprocess import build_normalized_dataset, save_dataset_artifacts
from train.probe_trainer import LinearProbe, extract_layer_features, train_and_evaluate_probe
from utils.config import MODEL_CONFIGS

try:
    import utils.model_hooks as model_hooks
except Exception as e:
    model_hooks = None


# -----------------------------
# Reproducibility / metrics
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None


def safe_auprc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(average_precision_score(y_true, scores))
    except Exception:
        return None


def compute_metrics(y_true: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> Dict[str, Optional[float]]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    y_pred = (scores >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "unsafe_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None,
        "precision_unsafe": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_unsafe": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None,
        "auroc": safe_auc(y_true, scores),
        "auprc": safe_auprc(y_true, scores),
    }
    return out


def per_group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col):
        m = compute_metrics(g["label"].to_numpy(), g["score"].to_numpy())
        m[group_col] = key
        rows.append(m)
    return pd.DataFrame(rows)


def compute_per_dataset_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, dataset_ids: np.ndarray) -> float:
    vals = []
    for ds in np.unique(dataset_ids):
        mask = dataset_ids == ds
        vals.append(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0))
    return float(np.mean(vals)) if vals else 0.0


# -----------------------------
# Model / representation extraction
# -----------------------------

def get_extractor_class():
    if model_hooks is None:
        raise RuntimeError("Cannot import utils.model_hooks from official SIREN repo.")
    # Prefer the official class used in train_general_siren.py.
    if hasattr(model_hooks, "Qwen3RepresentationExtractor"):
        return getattr(model_hooks, "Qwen3RepresentationExtractor")
    # Fallback: any class with RepresentationExtractor suffix.
    for name in dir(model_hooks):
        if name.endswith("RepresentationExtractor"):
            return getattr(model_hooks, name)
    raise RuntimeError("No RepresentationExtractor class found in utils.model_hooks.")


def build_extractor(model_name: str, device: str, batch_size: int, rep_types: Sequence[str]):
    cfg = MODEL_CONFIGS[model_name]
    extractor_cls = get_extractor_class()
    model_path = cfg.get("model_path") or cfg.get("hf_id") or cfg.get("path")
    if model_path is None:
        raise KeyError(f"MODEL_CONFIGS[{model_name!r}] must contain model_path/hf_id/path")
    return extractor_cls(model_path, device=device, batch_size=batch_size, rep_types=rep_types)


def available_pooling_types(all_reps: Dict[str, Dict[str, Any]]) -> set:
    """Pooling-type keys actually present in a (possibly cached) reps dict."""
    for split in all_reps.values():
        reps = split.get("representations") or []
        if reps:
            sample = reps[0]
            if sample:
                any_layer = next(iter(sample.values()))
                return set(any_layer.keys())
    return set()


def extract_representations(
    model_name: str,
    dfs: Dict[str, pd.DataFrame],
    device: str,
    batch_size: int,
    pooling_types: Sequence[str],
    out_dir: str,
    force_reextract: bool = False,
) -> Dict[str, Dict[str, Any]]:
    cache_path = os.path.join(out_dir, "reps_cache", f"{model_name}_representations.pkl")
    if os.path.exists(cache_path) and not force_reextract:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        avail = available_pooling_types(cached)
        if set(pooling_types).issubset(avail):
            print(f"[CACHE] loading representations: {cache_path} (pooling={sorted(avail)})")
            return cached
        # Cache was built with a different pooling/rep set; reusing it would
        # silently miss keys (e.g. KeyError 'mlp_mean'). Re-extract instead.
        print(f"[CACHE] stale representations cache: have {sorted(avail)}, "
              f"need {sorted(pooling_types)}; re-extracting.")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cfg = MODEL_CONFIGS[model_name]
    num_layers = int(cfg["num_layers"])
    extractor = build_extractor(model_name, device, batch_size, pooling_types)
    extractor.register_hooks()

    all_reps = {}
    for split, df in dfs.items():
        print(f"\n[REPR] Extracting {split}: {len(df)} samples")
        texts = df["text"].tolist()
        reps = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"extract {split}"):
            batch_texts = texts[i:i + batch_size]
            with torch.no_grad():
                batch_reps = extractor.extract_batch(batch_texts)
            reps.extend(batch_reps)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_reps[split] = {
            "representations": reps,
            "labels": df["label"].astype(int).to_numpy(),
            "langs": df["lang"].astype(str).to_numpy(),
            "source_datasets": df["source_dataset"].astype(str).to_numpy(),
            "source_ids": df["source_id"].astype(str).to_numpy(),
            "categories": df.get("category", pd.Series([None] * len(df))).astype(str).to_numpy(),
            "num_layers": num_layers,
        }
    extractor.remove_hooks()
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with open(cache_path, "wb") as f:
        pickle.dump(all_reps, f)
    return all_reps


# -----------------------------
# Probes and neuron selection
# -----------------------------

def probe_key(layer_idx: int, pooling_type: str) -> str:
    return f"layer{layer_idx}_{pooling_type}"


def split_pooling(pooling_type: str) -> Tuple[str, str]:
    rep_type = pooling_type.split("_")[0]
    pooling = "_".join(pooling_type.split("_")[1:])
    return rep_type, pooling


def train_probes_for_subset(
    train_reps: List[Any], train_labels: np.ndarray,
    val_reps: List[Any], val_labels: np.ndarray, val_dataset_ids: np.ndarray,
    test_reps: List[Any], test_labels: np.ndarray, test_dataset_ids: np.ndarray,
    num_layers: int, c_values: Sequence[float], pooling_types: Sequence[str], device: str,
    metric: str = "f1_macro",
) -> Dict[str, Dict[str, Any]]:
    best_probes = {}
    for pooling_type in pooling_types:
        rep_type, pooling = split_pooling(pooling_type)
        print(f"\n[PROBE] pooling={pooling_type}, n_train={len(train_labels)}, n_val={len(val_labels)}")
        for layer_idx in range(num_layers):
            if len(np.unique(train_labels)) < 2 or len(np.unique(val_labels)) < 2:
                print(f"  layer {layer_idx:02d}: skipped; only one class in train/val")
                continue
            try:
                probe, train_f1, val_f1, best_C = train_and_evaluate_probe(
                    train_reps, train_labels, val_reps, val_labels, val_dataset_ids,
                    layer_idx, rep_type, pooling, c_values, device, metric=metric,
                )
                test_X = extract_layer_features(test_reps, layer_idx, rep_type, pooling)
                test_f1 = probe.evaluate(test_X, test_labels, test_dataset_ids, metric=metric)
                key = probe_key(layer_idx, pooling_type)
                best_probes[key] = {
                    "layer": layer_idx,
                    "rep_type": rep_type,
                    "pooling": pooling,
                    "pooling_type": pooling_type,
                    "best_C": float(best_C),
                    "train_f1": float(train_f1),
                    "val_f1": float(val_f1),
                    "test_f1": float(test_f1),
                    "probe": probe,
                }
                print(f"  layer {layer_idx:02d}: train={train_f1:.4f} val={val_f1:.4f} test={test_f1:.4f} C={best_C}")
            except Exception as e:
                print(f"  layer {layer_idx:02d}: failed {type(e).__name__}: {e}")
    return best_probes


def train_all_probes(
    all_reps: Dict[str, Dict[str, Any]], languages: Sequence[str], c_values: Sequence[float],
    pooling_types: Sequence[str], device: str, out_dir: str, metric: str = "f1_macro",
    force_retrain: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    probe_dir = os.path.join(out_dir, "probes")
    os.makedirs(probe_dir, exist_ok=True)
    global_path = os.path.join(probe_dir, "global_probes.pkl")
    lang_path = os.path.join(probe_dir, "lang_probes.pkl")
    meta_path = os.path.join(probe_dir, "probe_cache_meta.json")
    want_meta = {
        "pooling_types": sorted(pooling_types),
        "c_values": [float(c) for c in c_values],
        "metric": metric,
    }
    if os.path.exists(global_path) and os.path.exists(lang_path) and not force_retrain:
        have_meta = None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                have_meta = json.load(f)
        if have_meta == want_meta:
            print(f"[CACHE] loading probes from {probe_dir}")
            with open(global_path, "rb") as f:
                global_probes = pickle.load(f)
            with open(lang_path, "rb") as f:
                lang_probes = pickle.load(f)
            return global_probes, lang_probes
        # Probes were trained with a different pooling/C/metric setup; reusing
        # them would silently mix configs or miss pooling keys. Retrain.
        print(f"[CACHE] stale probe cache (have={have_meta}, need={want_meta}); retraining.")

    train = all_reps["train"]
    val = all_reps["validation"]
    test = all_reps["test"]
    dataset_vocab = {d: i for i, d in enumerate(sorted(set(train["source_datasets"]) | set(val["source_datasets"]) | set(test["source_datasets"]))) }
    train_ds_ids = np.array([dataset_vocab[x] for x in train["source_datasets"]])
    val_ds_ids = np.array([dataset_vocab[x] for x in val["source_datasets"]])
    test_ds_ids = np.array([dataset_vocab[x] for x in test["source_datasets"]])

    print("\n=== Training global SIREN probes ===")
    global_probes = train_probes_for_subset(
        train["representations"], train["labels"],
        val["representations"], val["labels"], val_ds_ids,
        test["representations"], test["labels"], test_ds_ids,
        train["num_layers"], c_values, pooling_types, device, metric,
    )

    lang_probes = {}
    for lang in languages:
        tr_mask = train["langs"] == lang
        va_mask = val["langs"] == lang
        te_mask = test["langs"] == lang
        if tr_mask.sum() < 20 or va_mask.sum() < 10:
            print(f"[WARN] skipping language probe {lang}: too few samples")
            continue
        print(f"\n=== Training language-specific probes: {lang} ===")
        lang_probes[lang] = train_probes_for_subset(
            [r for r, m in zip(train["representations"], tr_mask) if m], train["labels"][tr_mask],
            [r for r, m in zip(val["representations"], va_mask) if m], val["labels"][va_mask], val_ds_ids[va_mask],
            [r for r, m in zip(test["representations"], te_mask) if m], test["labels"][te_mask], test_ds_ids[te_mask],
            train["num_layers"], c_values, pooling_types, device, metric,
        )

    with open(global_path, "wb") as f:
        pickle.dump(global_probes, f)
    with open(lang_path, "wb") as f:
        pickle.dump(lang_probes, f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(want_meta, f, indent=2)

    # CSV without model objects.
    rows = []
    for scope, probes in [("global", global_probes)] + [(f"lang:{k}", v) for k, v in lang_probes.items()]:
        for k, v in probes.items():
            rows.append({kk: vv for kk, vv in v.items() if kk != "probe"} | {"scope": scope, "key": k})
    pd.DataFrame(rows).to_csv(os.path.join(probe_dir, "probe_metrics.csv"), index=False)
    return global_probes, lang_probes


def select_salient_neurons(probe: LinearProbe, threshold: float) -> List[int]:
    weights = probe.get_feature_importance()
    total = float(np.sum(weights))
    if total <= 0:
        return []
    sorted_idx = np.argsort(weights)[::-1]
    selected = []
    csum = 0.0
    for idx in sorted_idx:
        selected.append(int(idx))
        csum += float(weights[idx])
        if csum >= threshold * total:
            break
    return selected


def get_layer_weights(global_probes: Dict[str, Any], pooling_type: str, num_layers: int) -> Dict[int, float]:
    scores = {}
    for layer_idx in range(num_layers):
        key = probe_key(layer_idx, pooling_type)
        if key in global_probes:
            scores[layer_idx] = float(global_probes[key]["val_f1"])
    if not scores:
        return {}
    max_s, min_s = max(scores.values()), min(scores.values())
    denom = (max_s - min_s) if max_s > min_s else 1.0
    return {layer: max(0.1, (score - min_s) / denom) for layer, score in scores.items()}


def selected_by_language(
    lang_probes: Dict[str, Dict[str, Any]], languages: Sequence[str], pooling_type: str,
    threshold: float, num_layers: int,
) -> Dict[str, Dict[int, List[int]]]:
    out = {lang: {} for lang in languages}
    for lang in languages:
        probes = lang_probes.get(lang, {})
        for layer_idx in range(num_layers):
            key = probe_key(layer_idx, pooling_type)
            if key in probes:
                out[lang][layer_idx] = select_salient_neurons(probes[key]["probe"], threshold)
            else:
                out[lang][layer_idx] = []
    return out


def selected_global(global_probes: Dict[str, Any], pooling_type: str, threshold: float, num_layers: int) -> Dict[int, List[int]]:
    out = {}
    for layer_idx in range(num_layers):
        key = probe_key(layer_idx, pooling_type)
        out[layer_idx] = select_salient_neurons(global_probes[key]["probe"], threshold) if key in global_probes else []
    return out


@dataclass(frozen=True)
class FeatureSpec:
    layer: int
    neuron: int
    pooling_type: str
    kind: str                 # global/shared/specific/random
    route_lang: Optional[str] # None = always active

    def as_dict(self):
        return {
            "layer": self.layer,
            "neuron": self.neuron,
            "pooling_type": self.pooling_type,
            "kind": self.kind,
            "route_lang": self.route_lang,
        }


def build_feature_manifest(
    method: str,
    global_selected: Dict[int, List[int]],
    lang_selected: Dict[str, Dict[int, List[int]]],
    languages: Sequence[str],
    pooling_type: str,
    num_layers: int,
    hidden_size_by_layer: Dict[int, int],
    shared_min_langs: int,
    seed: int,
) -> List[FeatureSpec]:
    rng = np.random.default_rng(seed)
    features: List[FeatureSpec] = []

    if method == "siren_original":
        for layer in range(num_layers):
            for n in sorted(set(global_selected.get(layer, []))):
                features.append(FeatureSpec(layer, n, pooling_type, "global", None))
        return features

    # Compute shared and lang-specific from language probes.
    shared_by_layer: Dict[int, set] = {}
    specific_by_layer_lang: Dict[Tuple[int, str], set] = {}
    for layer in range(num_layers):
        counts = {}
        for lang in languages:
            for n in set(lang_selected.get(lang, {}).get(layer, [])):
                counts[n] = counts.get(n, 0) + 1
        shared = {n for n, c in counts.items() if c >= shared_min_langs}
        shared_by_layer[layer] = shared
        for lang in languages:
            specific_by_layer_lang[(layer, lang)] = set(lang_selected.get(lang, {}).get(layer, [])) - shared

    if method == "shared_only":
        for layer in range(num_layers):
            for n in sorted(shared_by_layer[layer]):
                features.append(FeatureSpec(layer, n, pooling_type, "shared", None))
        return features

    if method == "shared_plus_all_specific":
        for layer in range(num_layers):
            union = set(shared_by_layer[layer])
            for lang in languages:
                union |= specific_by_layer_lang[(layer, lang)]
            for n in sorted(union):
                kind = "shared" if n in shared_by_layer[layer] else "specific_union"
                features.append(FeatureSpec(layer, n, pooling_type, kind, None))
        return features

    if method == "routed_shared_specific":
        for layer in range(num_layers):
            for n in sorted(shared_by_layer[layer]):
                features.append(FeatureSpec(layer, n, pooling_type, "shared", None))
            for lang in languages:
                for n in sorted(specific_by_layer_lang[(layer, lang)]):
                    features.append(FeatureSpec(layer, n, pooling_type, "specific", lang))
        return features

    if method in {"random_same_size_as_plus", "random_same_size_as_routed"}:
        # Match layer-wise feature budgets of plus/routed to check whether gains are just neuron-count effects.
        for layer in range(num_layers):
            h = hidden_size_by_layer[layer]
            if method == "random_same_size_as_plus":
                target = len(set(shared_by_layer[layer]).union(*[specific_by_layer_lang[(layer, l)] for l in languages]))
                chosen = rng.choice(h, size=min(target, h), replace=False).tolist() if target > 0 else []
                for n in sorted(chosen):
                    features.append(FeatureSpec(layer, int(n), pooling_type, "random_plus_budget", None))
            else:
                shared_target = len(shared_by_layer[layer])
                shared_rand = set(rng.choice(h, size=min(shared_target, h), replace=False).tolist()) if shared_target > 0 else set()
                for n in sorted(shared_rand):
                    features.append(FeatureSpec(layer, int(n), pooling_type, "random_shared", None))
                for lang in languages:
                    target = len(specific_by_layer_lang[(layer, lang)])
                    available = list(set(range(h)) - shared_rand)
                    if target > 0 and available:
                        chosen = rng.choice(available, size=min(target, len(available)), replace=False).tolist()
                        for n in sorted(chosen):
                            features.append(FeatureSpec(layer, int(n), pooling_type, "random_specific", lang))
        return features

    raise ValueError(f"Unknown method: {method}")


def infer_hidden_size_by_layer(representations: List[Any], pooling_type: str, num_layers: int) -> Dict[int, int]:
    sample = representations[0]
    return {layer: int(len(sample[layer][pooling_type])) for layer in range(num_layers)}


# -----------------------------
# Feature aggregation / layer values
# -----------------------------

def resolve_route_lang(lang: str, routable_langs: set, route_fallback_lang: Optional[str]) -> str:
    """Map a sample's language to the routing key used for specific features.

    Implements the previously-unused `route_unknown_lang_to` config: a sample
    whose language has no language-specific features (e.g. its probe was
    skipped for too few samples, or it is outside the configured languages)
    is routed to the fallback language so routed methods still activate a
    coherent set of specific neurons instead of silently zeroing them all.
    """
    if not routable_langs or lang in routable_langs:
        return lang
    if route_fallback_lang is not None and route_fallback_lang in routable_langs:
        return route_fallback_lang
    return lang


def aggregate_features(
    representations: List[Any], langs: np.ndarray, features: List[FeatureSpec],
    layer_weights: Dict[int, float], method: str,
    route_fallback_lang: Optional[str] = None,
) -> np.ndarray:
    routable_langs = {fs.route_lang for fs in features if fs.route_lang is not None}
    X = np.zeros((len(representations), len(features)), dtype=np.float32)
    for i, rep in enumerate(tqdm(representations, desc=f"aggregate {method}", leave=False)):
        eff_lang = resolve_route_lang(langs[i], routable_langs, route_fallback_lang)
        for j, fs in enumerate(features):
            if fs.route_lang is not None and fs.route_lang != eff_lang:
                continue
            X[i, j] = float(rep[fs.layer][fs.pooling_type][fs.neuron]) * float(layer_weights.get(fs.layer, 1.0))
    return X


def compute_layer_values(
    representations: List[Any], labels: np.ndarray, langs: np.ndarray, datasets: np.ndarray,
    source_ids: np.ndarray, features: List[FeatureSpec], layer_weights: Dict[int, float], method: str,
    threshold: float, route_fallback_lang: Optional[str] = None,
) -> pd.DataFrame:
    routable_langs = {fs.route_lang for fs in features if fs.route_lang is not None}
    by_layer: Dict[int, List[FeatureSpec]] = {}
    for fs in features:
        by_layer.setdefault(fs.layer, []).append(fs)
    rows = []
    for i, rep in enumerate(tqdm(representations, desc="layer values", leave=False)):
        lang = langs[i]
        eff_lang = resolve_route_lang(lang, routable_langs, route_fallback_lang)
        for layer, fss in by_layer.items():
            vals = []
            for fs in fss:
                if fs.route_lang is not None and fs.route_lang != eff_lang:
                    continue
                vals.append(float(rep[layer][fs.pooling_type][fs.neuron]))
            arr = np.asarray(vals, dtype=np.float32)
            if arr.size == 0:
                raw_signed = raw_abs = raw_l2 = raw_mean_abs = 0.0
            else:
                raw_signed = float(arr.sum())
                raw_abs = float(np.abs(arr).sum())
                raw_l2 = float(np.linalg.norm(arr))
                raw_mean_abs = float(np.abs(arr).mean())
            w = float(layer_weights.get(layer, 1.0))
            rows.append({
                "id": str(source_ids[i]),
                "source_dataset": str(datasets[i]),
                "lang": str(lang),
                "label": int(labels[i]),
                "layer_idx": int(layer),
                "pooling_type": fss[0].pooling_type if fss else None,
                "method": method,
                "threshold": threshold,
                "layer_weight": w,
                "n_selected_neurons": int(arr.size),
                "raw_signed_sum": raw_signed,
                "raw_abs_sum": raw_abs,
                "raw_l2": raw_l2,
                "raw_mean_abs": raw_mean_abs,
                "signed_sum": raw_signed * w,
                "abs_sum": raw_abs * w,
                "l2": raw_l2 * w,
                "mean_abs": raw_mean_abs * w,
            })
    return pd.DataFrame(rows)


# -----------------------------
# MLP classifier: copied/adapted from official train_general_siren.py
# -----------------------------

class AdaptiveMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, layer_dims: Sequence[int], dropout_rates: Sequence[float], num_classes: int = 2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim, dropout in zip(layer_dims, dropout_rates):
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        final = nn.Linear(prev_dim, num_classes)
        nn.init.kaiming_normal_(final.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def predict_scores(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, scores = [], []
    use_cuda_amp = device.type == "cuda"
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.FloatTensor(X[i:i + batch_size]).to(device)
            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                logits = model(bx)
                prob = torch.softmax(logits, dim=1)[:, 1]
            scores.extend(prob.detach().cpu().numpy().tolist())
            preds.extend((prob.detach().cpu().numpy() >= 0.5).astype(int).tolist())
    return np.asarray(preds), np.asarray(scores)


def train_model(
    model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, val_dataset_ids: np.ndarray,
    lr: float, batch_size: int, epochs: int, device: torch.device,
    patience: int = 10, trial: Optional[optuna.Trial] = None, show_progress: bool = False,
    weight_decay: float = 1e-4,
) -> float:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    use_cuda_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    best_val_f1, best_state, patience_counter = 0.0, None, 0
    iterator = tqdm(range(epochs), desc="train mlp") if show_progress else range(epochs)
    for epoch in iterator:
        model.train()
        indices = torch.randperm(len(X_train))
        for start in range(0, len(X_train), batch_size):
            idx = indices[start:start + batch_size]
            bx = torch.FloatTensor(X_train[idx]).to(device)
            by = torch.LongTensor(y_train[idx]).to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                loss = criterion(model(bx), by)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        val_pred, _ = predict_scores(model, X_val, device, batch_size=2048)
        val_f1 = compute_per_dataset_macro_f1(y_val, val_pred, val_dataset_ids)
        if show_progress:
            iterator.set_postfix({"val_f1": f"{val_f1:.4f}", "best": f"{best_val_f1:.4f}"})
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if trial is not None:
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if patience_counter >= patience:
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_val_f1


def optuna_objective(
    trial: optuna.Trial, X: np.ndarray, y: np.ndarray, dataset_ids: np.ndarray,
    device: torch.device, cfg: Dict[str, Any], seed: int,
) -> float:
    input_dim = X.shape[1]
    n_layers = trial.suggest_int("n_layers", int(cfg["hidden_layers_min"]), int(cfg["hidden_layers_max"]))
    lr = trial.suggest_float("lr", float(cfg["lr_min"]), float(cfg["lr_max"]), log=True)
    layer_dims, dropouts = [], []
    for i in range(n_layers):
        if i == 0:
            min_dim = min(input_dim, 256)
            max_dim = min(input_dim * 2, int(cfg["hidden_dim_max"]))
            min_dim = max(64, min_dim)
            max_dim = max(min_dim, max_dim)
        else:
            min_dim = int(cfg["hidden_dim_min"])
            max_dim = min(layer_dims[-1], int(cfg["hidden_dim_max"]))
            max_dim = max(min_dim, max_dim)
        step = 64 if max_dim - min_dim >= 64 else 1
        h = trial.suggest_int(f"hidden_dim_layer{i}", min_dim, max_dim, step=step)
        d = trial.suggest_float(f"dropout_layer{i}", float(cfg["dropout_min"]), float(cfg["dropout_max"]))
        layer_dims.append(h)
        dropouts.append(d)

    kfold = KFold(n_splits=int(cfg["n_folds"]), shuffle=True, random_state=seed)
    vals = []
    for train_idx, val_idx in kfold.split(X):
        model = AdaptiveMLPClassifier(input_dim, layer_dims, dropouts).to(device)
        v = train_model(
            model, X[train_idx], y[train_idx], X[val_idx], y[val_idx], dataset_ids[val_idx],
            lr=lr, batch_size=int(cfg["batch_size"]), epochs=100, device=device,
            patience=int(cfg.get("early_stopping_patience", 10)), trial=None,
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
        )
        vals.append(v)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return float(np.mean(vals))


def search_mlp_params(X_train: np.ndarray, y_train: np.ndarray, dataset_ids: np.ndarray, device: torch.device, cfg: Dict[str, Any], seed: int) -> Tuple[Dict[str, Any], float]:
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )
    study.optimize(lambda t: optuna_objective(t, X_train, y_train, dataset_ids, device, cfg, seed),
                   n_trials=int(cfg["n_trials"]), n_jobs=1, show_progress_bar=True)
    best = max(study.trials, key=lambda t: t.value if t.value is not None else -1)
    if not best.params or best.value is None:
        # All trials failed/pruned: fall back to a minimal valid config so the
        # final model can still be trained instead of crashing on float(None).
        fallback = {
            "n_layers": int(cfg["hidden_layers_min"]),
            "lr": float(cfg["lr_min"]),
        }
        for i in range(fallback["n_layers"]):
            fallback[f"hidden_dim_layer{i}"] = int(cfg["hidden_dim_min"])
            fallback[f"dropout_layer{i}"] = float(cfg["dropout_min"])
        print("[WARN] no successful Optuna trial; using fallback MLP params.")
        return fallback, 0.0
    return dict(best.params), float(best.value)


def train_final_mlp(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, val_dataset_ids: np.ndarray,
    params: Dict[str, Any], device: torch.device, cfg: Dict[str, Any], cv_f1: float,
) -> Tuple[nn.Module, float]:
    input_dim = X_train.shape[1]
    n_layers = int(params["n_layers"])
    dims = [int(params[f"hidden_dim_layer{i}"]) for i in range(n_layers)]
    drops = [float(params[f"dropout_layer{i}"]) for i in range(n_layers)]
    model = AdaptiveMLPClassifier(input_dim, dims, drops).to(device)
    best_val = train_model(
        model, X_train, y_train, X_val, y_val, val_dataset_ids,
        lr=float(params["lr"]), batch_size=int(cfg["batch_size"]), epochs=int(cfg.get("epochs_final", 512)),
        device=device, patience=int(cfg.get("early_stopping_patience", 10)), show_progress=True,
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    return model, best_val if best_val > 0 else cv_f1


def dataset_ids_from_sources(sources: np.ndarray, vocab: Dict[str, int]) -> np.ndarray:
    return np.asarray([vocab[x] for x in sources])


def active_feature_counts(features: List[FeatureSpec], languages: Sequence[str]) -> Dict[str, int]:
    out = {}
    for lang in languages:
        out[lang] = sum(1 for fs in features if fs.route_lang is None or fs.route_lang == lang)
    out["union_dim"] = len(features)
    return out


# -----------------------------
# Main method runner
# -----------------------------

def run_method(
    method: str, threshold: float, pooling_type: str, seed: int,
    all_reps: Dict[str, Dict[str, Any]], global_probes: Dict[str, Any], lang_probes: Dict[str, Dict[str, Any]],
    config: Dict[str, Any], model_name: str, out_dir: str, device: torch.device,
) -> None:
    set_seed(seed)
    languages = [str(x) for x in config.get("languages", sorted(set(all_reps["train"]["langs"])))]
    num_layers = int(all_reps["train"]["num_layers"])
    layer_weights = get_layer_weights(global_probes, pooling_type, num_layers)
    global_sel = selected_global(global_probes, pooling_type, threshold, num_layers)
    lang_sel = selected_by_language(lang_probes, languages, pooling_type, threshold, num_layers)
    hidden_size_by_layer = infer_hidden_size_by_layer(all_reps["train"]["representations"], pooling_type, num_layers)
    features = build_feature_manifest(
        method, global_sel, lang_sel, languages, pooling_type, num_layers, hidden_size_by_layer,
        int(config.get("shared_min_langs", len(languages))), seed,
    )
    if len(features) == 0:
        print(f"[WARN] no features selected for {method} threshold={threshold} pooling={pooling_type}; skip")
        return

    run_dir = os.path.join(out_dir, f"method={method}", f"threshold={threshold}", f"pooling={pooling_type}", f"seed={seed}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n[RUN] {method} threshold={threshold} pooling={pooling_type} seed={seed} dim={len(features)}")
    route_fallback = config.get("route_unknown_lang_to")
    X_train = aggregate_features(all_reps["train"]["representations"], all_reps["train"]["langs"], features, layer_weights, method, route_fallback)
    X_val = aggregate_features(all_reps["validation"]["representations"], all_reps["validation"]["langs"], features, layer_weights, method, route_fallback)
    X_test = aggregate_features(all_reps["test"]["representations"], all_reps["test"]["langs"], features, layer_weights, method, route_fallback)

    source_vocab = {d: i for i, d in enumerate(sorted(set(all_reps["train"]["source_datasets"]) | set(all_reps["validation"]["source_datasets"]) | set(all_reps["test"]["source_datasets"]))) }
    train_ds_ids = dataset_ids_from_sources(all_reps["train"]["source_datasets"], source_vocab)
    val_ds_ids = dataset_ids_from_sources(all_reps["validation"]["source_datasets"], source_vocab)

    # SIREN-style Optuna CV. Search on train+val, final model still monitored on validation for stable early stopping.
    X_cv = np.vstack([X_train, X_val])
    y_cv = np.concatenate([all_reps["train"]["labels"], all_reps["validation"]["labels"]])
    ds_cv = np.concatenate([train_ds_ids, val_ds_ids])
    mlp_cfg = dict(config["mlp"])
    mlp_cfg["n_folds"] = int(config["mlp"].get("n_folds", 3))
    params, cv_f1 = search_mlp_params(X_cv, y_cv, ds_cv, device, mlp_cfg, seed)
    model, best_val_f1 = train_final_mlp(
        X_train, all_reps["train"]["labels"], X_val, all_reps["validation"]["labels"], val_ds_ids,
        params, device, mlp_cfg, cv_f1,
    )

    pred_test, score_test = predict_scores(model, X_test, device)
    pred_train, score_train = predict_scores(model, X_train, device)
    pred_val, score_val = predict_scores(model, X_val, device)

    def pred_df(split: str, preds: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
        r = all_reps[split]
        return pd.DataFrame({
            "split": split,
            "id": r["source_ids"],
            "source_dataset": r["source_datasets"],
            "lang": r["langs"],
            "category": r["categories"],
            "label": r["labels"].astype(int),
            "pred": preds.astype(int),
            "score": scores.astype(float),
            "method": method,
            "threshold": threshold,
            "pooling_type": pooling_type,
            "seed": seed,
        })

    all_pred_df = pd.concat([
        pred_df("train", pred_train, score_train),
        pred_df("validation", pred_val, score_val),
        pred_df("test", pred_test, score_test),
    ], ignore_index=True)
    all_pred_df.to_json(os.path.join(run_dir, "predictions.jsonl"), orient="records", lines=True, force_ascii=False)

    test_df = all_pred_df[all_pred_df["split"] == "test"].copy()
    metrics = {
        "model": model_name,
        "method": method,
        "threshold": threshold,
        "pooling_type": pooling_type,
        "seed": seed,
        "num_features_union_dim": len(features),
        "active_feature_counts": active_feature_counts(features, languages),
        "best_params": params,
        "cv_f1": cv_f1,
        "best_val_f1": best_val_f1,
        "overall_test": compute_metrics(test_df["label"].to_numpy(), test_df["score"].to_numpy()),
        "per_language_test": per_group_metrics(test_df, "lang").to_dict(orient="records"),
        "per_dataset_test": per_group_metrics(test_df, "source_dataset").to_dict(orient="records"),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Save model and feature metadata.
    with open(os.path.join(run_dir, "best_model.pkl"), "wb") as f:
        pickle.dump({
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "model_class": "AdaptiveMLPClassifier",
            "feature_dim": len(features),
            "best_params": params,
            "method": method,
            "threshold": threshold,
            "pooling_type": pooling_type,
            "layer_weights": layer_weights,
            "languages": languages,
        }, f)

    with open(os.path.join(run_dir, "feature_manifest.json"), "w", encoding="utf-8") as f:
        json.dump([fs.as_dict() for fs in features], f, indent=2, ensure_ascii=False)

    selected_summary = {
        "global_selected": {str(k): v for k, v in global_sel.items()},
        "lang_selected": {lang: {str(k): v for k, v in layers.items()} for lang, layers in lang_sel.items()},
        "layer_weights": {str(k): float(v) for k, v in layer_weights.items()},
        "shared_min_langs": int(config.get("shared_min_langs", len(languages))),
        "active_feature_counts": active_feature_counts(features, languages),
    }
    with open(os.path.join(run_dir, "selected_neurons.json"), "w", encoding="utf-8") as f:
        json.dump(selected_summary, f, indent=2, ensure_ascii=False)

    lv = compute_layer_values(
        all_reps["test"]["representations"], all_reps["test"]["labels"], all_reps["test"]["langs"],
        all_reps["test"]["source_datasets"], all_reps["test"]["source_ids"], features, layer_weights, method,
        threshold, route_fallback,
    )
    lv.to_csv(os.path.join(run_dir, "layer_values.csv"), index=False)

    print(f"[DONE] {run_dir}")
    print(json.dumps(metrics["overall_test"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--methods", nargs="+", default=["siren_original", "shared_only", "routed_shared_specific", "shared_plus_all_specific", "random_same_size_as_routed", "random_same_size_as_plus"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.6, 0.8])
    parser.add_argument("--pooling_types", nargs="+", default=["residual_mean"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=None)
    parser.add_argument("--skip_failed_datasets", action="store_true")
    parser.add_argument("--force_reextract", action="store_true")
    parser.add_argument("--force_retrain_probes", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.n_trials is not None:
        config["mlp"]["n_trials"] = args.n_trials
    if args.n_folds is not None:
        config["mlp"]["n_folds"] = args.n_folds
    seeds = args.seeds if args.seeds is not None else [int(config.get("seed", 42))]

    set_seed(seeds[0])
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    out_dir = os.path.join(config.get("output_root", "outputs/multilingual_siren"), args.model)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    print("\n[1] Build normalized dataset")
    dfs = build_normalized_dataset(config, skip_failed=args.skip_failed_datasets)
    for split, df in dfs.items():
        print(f"{split:10s}: {len(df):7d} | langs={sorted(df['lang'].unique().tolist()) if len(df) else []}")
    save_dataset_artifacts(dfs, out_dir)

    print("\n[2] Extract SIREN representations")
    all_reps = extract_representations(args.model, dfs, str(device), args.batch_size, args.pooling_types, out_dir, force_reextract=args.force_reextract)

    print("\n[3] Train global and language-specific SIREN probes")
    global_probes, lang_probes = train_all_probes(
        all_reps, config.get("languages", []), config["probe"]["c_values"], args.pooling_types,
        str(device), out_dir, metric=config["probe"].get("metric", "f1_macro"), force_retrain=args.force_retrain_probes,
    )

    print("\n[4] Run selection methods")
    for seed in seeds:
        for pooling_type in args.pooling_types:
            for threshold in args.thresholds:
                for method in args.methods:
                    run_method(method, threshold, pooling_type, seed, all_reps, global_probes, lang_probes, config, args.model, out_dir, device)


if __name__ == "__main__":
    main()
