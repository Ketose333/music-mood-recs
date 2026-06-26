"""Precompute MoodCNN embeddings for every track in the melspec manifest.

Streamlit Community Cloud guarantees only 1GB RAM. Loading all ~2,247
mel-spectrograms (128 x 1292 float32, ~0.63MB each => ~1.4GB) into memory to
run a forward pass at app startup blows that budget and OOM-kills the app.

This script does that forward pass once, offline, and saves just the
resulting embeddings (2,247 x embedding_dim floats, a few hundred KB) so
app.py only needs to load a tiny precomputed array at runtime - never the
full mel-spectrogram set.

Usage:
  python -m scripts.precompute_embeddings
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from src.models.cnn import CNNConfig, MoodCNN
from src.recommend.similar import extract_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default="models/cnn")
    parser.add_argument("--manifest", default="artifacts/melspec_manifest.csv")
    parser.add_argument("--out", default="artifacts/embeddings.npy")
    args = parser.parse_args()

    with open(os.path.join(args.model_dir, "config.json"), encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = CNNConfig(n_mels=cfg_dict["n_mels"], n_classes=cfg_dict["n_classes"], embedding_dim=cfg_dict["embedding_dim"])
    model = MoodCNN(cfg)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pt"), map_location="cpu"))
    model.eval()

    manifest = pd.read_csv(args.manifest)
    mels = np.stack([np.load(p).astype(np.float32) for p in manifest["npy_path"]])
    embeddings = extract_embeddings(model, mels, batch_size=32, device="cpu")

    np.save(args.out, embeddings)
    print(f"Saved {embeddings.shape} embeddings -> {args.out} (manifest row order, {len(manifest)} tracks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
