"""Multi-label mood classification metrics and the cross-model comparison table.

load_all_metrics() scans models/*/metrics.json — adding CRNN later only requires
writing a metrics.json in the same shape; no code here needs to change.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def compute_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    acc = float((preds == labels).all(axis=1).mean())
    try:
        auc = roc_auc_score(labels, probs, average="macro")
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": round(acc, 4),
        "f1_micro": round(f1_micro, 4),
        "f1_macro": round(f1_macro, 4),
        "roc_auc": round(float(auc), 4),
    }


def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """results = {model_display_name: {"accuracy":..., "f1_micro":..., "f1_macro":..., "roc_auc":...}}."""
    if not results:
        return pd.DataFrame(columns=["Accuracy", "F1(micro)", "F1(macro)", "ROC-AUC"])

    df = pd.DataFrame(results).T
    df = df.rename(
        columns={
            "accuracy": "Accuracy",
            "f1_micro": "F1(micro)",
            "f1_macro": "F1(macro)",
            "roc_auc": "ROC-AUC",
        }
    )
    return df[["Accuracy", "F1(micro)", "F1(macro)", "ROC-AUC"]]


def load_all_metrics(models_dir: str = "models") -> dict[str, dict]:
    """Scans models/*/metrics.json. Each file stores a "history" list (one entry per
    epoch); the latest epoch's accuracy/F1/ROC-AUC is used for the comparison table."""
    results: dict[str, dict] = {}
    for metrics_path in sorted(glob.glob(os.path.join(models_dir, "*", "metrics.json"))):
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        model_dir_name = os.path.basename(os.path.dirname(metrics_path))
        display_name = data.get("display_name", model_dir_name)
        history = data.get("history")
        latest = history[-1] if history else data
        results[display_name] = {
            "accuracy": latest.get("accuracy"),
            "f1_micro": latest.get("f1_micro"),
            "f1_macro": latest.get("f1_macro"),
            "roc_auc": latest.get("roc_auc"),
        }
    return results
