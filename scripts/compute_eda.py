"""Generate EDA figures for the report PPT.

Produces:
  - artifacts/report_figures/fig_tag_distribution.png — train tag counts bar chart
  - artifacts/report_figures/fig_duration_hist.png — track duration histogram
  - artifacts/report_figures/fig_melspec_example.png — example log-mel spectrogram (if audio available)

Usage:
  python -m scripts.compute_eda
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.load_jamendo import build_subset

OUT_DIR = "artifacts/report_figures"
FIGSIZE = (7.59, 4.43)  # report slide image cap: 759x443px at dpi=100
DPI = 100
# data/audio + artifacts/melspecs are hosted on this HF Hub dataset repo, not
# tracked in git (see app.py's _resolve()) — fall back to it below when no
# local melspecs/ folder exists.
HF_ASSETS_REPO = "Ketose333/music-mood-recs-assets"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    subset = build_subset(top_n=5, dest_dir="data/jamendo")
    tags = subset["tags"]
    train = subset["train"]

    # 1. Tag distribution
    tag_cols = [f"tag_{t}" for t in tags]
    counts = train[tag_cols].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    counts.plot.bar(ax=ax, color="#4C9F70")
    ax.set_title("Train split - mood/theme tag distribution")
    ax.set_ylabel("track count")
    ax.set_xlabel("tag")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_tag_distribution.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print("saved", out_path)

    # 2. Duration histogram
    durations = train["DURATION"].astype(float)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    durations.hist(bins=40, ax=ax, color="#5B8DEF", edgecolor="white")
    ax.set_title("Track duration distribution (s)")
    ax.set_xlabel("duration (seconds)")
    ax.set_ylabel("track count")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_duration_hist.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print("saved", out_path)

    # 3. Mel-spectrogram example: prefer a local .npy under artifacts/melspecs/,
    # otherwise fetch just one example file from the HF Hub assets repo (the
    # full melspecs/ folder lives there now — no need to pull all of it).
    mel_path = None
    for root, _, names in os.walk("artifacts/melspecs"):
        for n in names:
            if n.endswith(".npy"):
                mel_path = os.path.join(root, n)
                break
        if mel_path:
            break
    if mel_path is None and os.path.exists("artifacts/melspec_manifest.csv"):
        manifest = pd.read_csv("artifacts/melspec_manifest.csv")
        if len(manifest):
            from huggingface_hub import hf_hub_download
            rel = manifest.iloc[0]["npy_path"].replace(os.sep, "/")
            mel_path = hf_hub_download(repo_id=HF_ASSETS_REPO, repo_type="dataset", filename=rel)
            print(f"fetched example mel-spectrogram from HF Hub: {rel}")
    if mel_path:
        mel = np.load(mel_path)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
        ax.set_title("Log-mel spectrogram example (30s segment)")
        ax.set_xlabel("time frames")
        ax.set_ylabel("mel bins")
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, "fig_melspec_example.png")
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
        print("saved", out_path)
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
