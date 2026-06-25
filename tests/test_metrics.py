"""Tests for src.evaluation.metrics — multi-label metric computation."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import build_comparison_table, compute_metrics


def test_compute_metrics_perfect_predictions():
    labels = np.array([[1, 0], [0, 1], [1, 1]])
    logits = np.where(labels == 1, 10.0, -10.0)
    metrics = compute_metrics(logits, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_micro"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_build_comparison_table_columns():
    results = {"cnn": {"accuracy": 0.5, "f1_micro": 0.4, "f1_macro": 0.3, "roc_auc": 0.6}}
    df = build_comparison_table(results)
    assert list(df.columns) == ["Accuracy", "F1(micro)", "F1(macro)", "ROC-AUC"]
    assert df.loc["cnn", "Accuracy"] == 0.5


def test_build_comparison_table_empty():
    df = build_comparison_table({})
    assert list(df.columns) == ["Accuracy", "F1(micro)", "F1(macro)", "ROC-AUC"]
    assert len(df) == 0
