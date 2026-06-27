"""Generate the training-curve figure for the report PPT from models/cnn/metrics.json.

Produces:
  - artifacts/report_figures/fig_training_curves.png — train/val loss + val F1(micro/macro) curves

Usage:
  python scripts/plot_training_curves.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "artifacts/report_figures"
FIGSIZE = (7.59, 4.43)  # report slide image cap: 759x443px at dpi=100
DPI = 100


def main() -> None:
    with open("models/cnn/metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    history = metrics["history"]
    epochs = [h["epoch"] for h in history]

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    ax1.plot(epochs, [h["train_loss"] for h in history], label="train loss", color="#5B8DEF")
    ax1.plot(epochs, [h["loss"] for h in history], label="val loss", color="#E0656A")
    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()

    ax2.plot(epochs, [h["f1_micro"] for h in history], label="val F1(micro)", color="#4C9F70")
    ax2.plot(epochs, [h["f1_macro"] for h in history], label="val F1(macro)", color="#C99A4B")
    ax2.set_title("Validation F1")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("F1")
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_training_curves.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print("saved", out_path)


if __name__ == "__main__":
    main()
