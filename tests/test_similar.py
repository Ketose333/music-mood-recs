"""Tests for src.recommend.similar — embedding extraction and top-k ranking."""

from __future__ import annotations

import numpy as np
import torch

from src.models.cnn import CNNConfig, MoodCNN
from src.recommend.similar import extract_embeddings, top_k_similar


def test_extract_embeddings_shape():
    cfg = CNNConfig(n_mels=64, embedding_dim=32)
    model = MoodCNN(cfg)
    mels = np.random.RandomState(0).randn(10, cfg.n_mels, 300).astype(np.float32)
    emb = extract_embeddings(model, mels, batch_size=4)
    assert emb.shape == (10, cfg.embedding_dim)
    assert emb.dtype == np.float32


def test_top_k_similar_excludes_query_and_returns_sorted_desc():
    # Use a controlled embedding matrix so the ranking is predictable.
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.95, 0.05],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    idx, sims = top_k_similar(0, embeddings, k=3)
    # query=0 is [1,0]; most similar are 3 ([0.95,0.05]) then 1 ([0.9,0.1])
    assert idx[0] == 3
    assert idx[1] == 1
    assert 0 not in idx
    assert np.all(np.diff(sims) <= 0)


def test_top_k_similar_respects_k_and_exclude_query():
    embeddings = np.random.RandomState(1).randn(20, 8).astype(np.float32)
    idx, sims = top_k_similar(5, embeddings, k=5)
    assert len(idx) == 5
    assert 5 not in idx
    assert len(sims) == 5
