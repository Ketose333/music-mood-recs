"""Train the MoodCNN classifier on precomputed mel-spectrograms.

Multi-label mood classification (5 tags) trained with BCEWithLogitsLoss.
Saves the model checkpoint, label tags, and metrics (accuracy/F1/ROC-AUC)
to ``models/cnn/`` so the Streamlit app and recommendation step can load
them.

Usage:
  python scripts/train_cnn.py --epochs 15 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.dataset import MelspecDataset
from src.data.jamendo import build_subset
from src.models.cnn import CNNConfig, MoodCNN, count_parameters


def _metrics_from_logits(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
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


@torch.no_grad()
def evaluate(model: MoodCNN, loader: DataLoader, device: torch.device, criterion) -> dict:
    model.eval()
    all_logits, all_labels, total_loss = [], [], 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    metrics = _metrics_from_logits(logits, labels)
    metrics["loss"] = round(total_loss / len(labels), 4)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="artifacts/melspec_manifest.csv")
    parser.add_argument("--subset-meta", default="artifacts/subset_meta.csv")
    parser.add_argument("--model-out", default="models/cnn")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    subset = build_subset(top_n=5, dest_dir="data/jamendo")
    tags = subset["tags"]
    n_classes = len(tags)
    print(f"Tags ({n_classes}): {tags}")

    train_ds = MelspecDataset(args.manifest, args.subset_meta, tags, split="train", n_mels=args.n_mels)
    val_ds = MelspecDataset(args.manifest, args.subset_meta, tags, split="validation", n_mels=args.n_mels)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    if len(train_ds) == 0:
        print("No training samples found. Run scripts/download_audio.py and scripts/extract_melspecs.py first.")
        return 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    cfg = CNNConfig(n_mels=args.n_mels, n_classes=n_classes, embedding_dim=args.embedding_dim)
    model = MoodCNN(cfg).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.model_out, exist_ok=True)
    history: list[dict] = []
    best_val_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        val_metrics = evaluate(model, val_loader, device, criterion)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_f1_micro={val_metrics['f1_micro']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), **val_metrics})
        if val_metrics["f1_micro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_micro"]
            torch.save(model.state_dict(), os.path.join(args.model_out, "model.pt"))
            print(f"  -> saved best model (val_f1_micro={best_val_f1:.4f})")

    with open(os.path.join(args.model_out, "tags.json"), "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.model_out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"n_mels": cfg.n_mels, "n_classes": cfg.n_classes, "embedding_dim": cfg.embedding_dim}, f, indent=2)
    metrics = {"best_val_f1_micro": best_val_f1, "history": history, "tags": tags}
    with open(os.path.join(args.model_out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Best val F1(micro)={best_val_f1:.4f}. Artifacts in {args.model_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
