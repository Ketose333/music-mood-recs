"""Generate EDA figures for the report PPT.

Produces:
  - artifacts/fig_tag_distribution.png — train tag counts bar chart
  - artifacts/fig_duration_hist.png — track duration histogram
  - artifacts/fig_melspec_example.png — example log-mel spectrogram (if audio available)

Usage:
  python scripts/compute_eda.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.load_jamendo import build_subset


def main() -> None:
    os.makedirs("artifacts", exist_ok=True)
    subset = build_subset(top_n=5, dest_dir="data/jamendo")
    tags = subset["tags"]
    train = subset["train"]

    # 1. Tag distribution
    tag_cols = [f"tag_{t}" for t in tags]
    counts = train[tag_cols].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax, color="#4C9F70")
    ax.set_title("Train split - mood/theme tag distribution")
    ax.set_ylabel("track count")
    ax.set_xlabel("tag")
    plt.tight_layout()
    fig.savefig("artifacts/fig_tag_distribution.png", dpi=150)
    plt.close(fig)
    print("saved artifacts/fig_tag_distribution.png")

    # 2. Duration histogram
    durations = train["DURATION"].astype(float)
    fig, ax = plt.subplots(figsize=(8, 4))
    durations.hist(bins=40, ax=ax, color="#5B8DEF", edgecolor="white")
    ax.set_title("Track duration distribution (s)")
    ax.set_xlabel("duration (seconds)")
    ax.set_ylabel("track count")
    plt.tight_layout()
    fig.savefig("artifacts/fig_duration_hist.png", dpi=150)
    plt.close(fig)
    print("saved artifacts/fig_duration_hist.png")

    # 3. Mel-spectrogram example (if any .npy exists)
    mel_path = None
    for root, _, names in os.walk("artifacts/melspecs"):
        for n in names:
            if n.endswith(".npy"):
                mel_path = os.path.join(root, n)
                break
        if mel_path:
            break
    if mel_path:
        mel = np.load(mel_path)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
        ax.set_title("Log-mel spectrogram example (30s segment)")
        ax.set_xlabel("time frames")
        ax.set_ylabel("mel bins")
        plt.tight_layout()
        fig.savefig("artifacts/fig_melspec_example.png", dpi=150)
        plt.close(fig)
        print("saved artifacts/fig_melspec_example.png")
    else:
        print("no .npy mel-spectrogram found, skipping fig_melspec_example.png")

    # 4. Summary stats
    stats = {
        "total_train": len(train),
        "total_val": len(subset["validation"]),
        "total_test": len(subset["test"]),
        "tags": tags,
        "tag_counts": {t: int(counts.get(f"tag_{t}", 0)) for t in tags},
        "duration_mean": round(float(durations.mean()), 1),
        "duration_median": round(float(durations.median()), 1),
    }
    import json
    os.makedirs("models/eda", exist_ok=True)
    with open("models/eda/stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("saved models/eda/stats.json")
    print(f"summary: {stats['total_train']} train, {stats['total_val']} val, {stats['total_test']} test")


if __name__ == "__main__":
    main()
