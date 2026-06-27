"""Generate the model-prediction example figures for the report PPT
(slide "모델 예측 - 무드 분류" / "모델 예측 - 코사인 유사도 Top-5 추천").

Uses the real trained model (models/cnn/model.pt) and the real embeddings
already computed for the 30-TAR subset. Only artifacts/embeddings.npy needs
fetching from the HF Hub assets repo (small, ~tracks x 64 floats) — no raw
audio or mel-spectrogram download needed, since predict_mood_probs/
top_k_similar both operate directly on precomputed embeddings.

Produces:
  - artifacts/report_figures/fig_mood_probs_example.png
  - artifacts/report_figures/fig_top5_similarity_example.png

Usage:
  python -m scripts.plot_prediction_examples
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.models.cnn import CNNConfig, MoodCNN
from src.recommend.similar import predict_mood_probs, top_k_similar

HF_ASSETS_REPO = "Ketose333/music-mood-recs-assets"
OUT_DIR = "artifacts/report_figures"
FIGSIZE = (7.59, 4.43)  # report slide image cap: 759x443px at dpi=100
DPI = 100
QUERY_IDX = 0


def _resolve(rel_path: str) -> str:
    """Same lazy-download pattern as app.py's _resolve()."""
    if os.path.exists(rel_path):
        return rel_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_ASSETS_REPO, repo_type="dataset", filename=rel_path.replace(os.sep, "/"))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    with open("models/cnn/tags.json", encoding="utf-8") as f:
        tags = json.load(f)
    with open("models/cnn/config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    model = MoodCNN(CNNConfig(**cfg))
    model.load_state_dict(torch.load("models/cnn/model.pt", map_location="cpu"))
    model.eval()

    manifest = pd.read_csv(_resolve("artifacts/melspec_manifest.csv"))
    embeddings = np.load(_resolve("artifacts/embeddings.npy"))
    track_ids = manifest["TRACK_ID"].tolist()
    track_id = track_ids[QUERY_IDX]

    # 1. Mood probability bar chart for the example track
    probs = predict_mood_probs(model, embeddings[QUERY_IDX : QUERY_IDX + 1])[0]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(tags, probs, color="#4C9F70")
    ax.set_ylim(0, 1)
    ax.set_title(f"{track_id} - mood probability (sigmoid)")
    ax.set_ylabel("probability")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_mood_probs_example.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print("saved", out_path)

    # 2. Top-5 cosine-similarity recommendation bar chart for the same track.
    # This embedding space's baseline similarity is high (mean ~0.72 across
    # all pairs), so the top-5 scores all sit in a narrow band near 1.0 — a
    # fixed 0-1 x-axis would render 5 visually identical bars. Zoom to the
    # actual value range (with small padding) and label each bar so the
    # ranking is still legible.
    idxs, sims = top_k_similar(QUERY_IDX, embeddings, k=5)
    labels = [track_ids[i] for i in idxs]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.barh(labels[::-1], sims[::-1], color="#5B8DEF")
    lo = min(sims) - 0.01
    ax.set_xlim(max(0.0, lo), 1.0)
    for bar, sim in zip(bars, sims[::-1]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {sim:.3f}", va="center", fontsize=9)
    ax.set_title(f"{track_id} - top-5 similar tracks (cosine similarity)")
    ax.set_xlabel("cosine similarity")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_top5_similarity_example.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print("saved", out_path)


if __name__ == "__main__":
    main()
