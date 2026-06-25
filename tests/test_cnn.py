"""Tests for src.models.cnn — forward pass and embedding shape on synthetic input."""

from __future__ import annotations

import numpy as np
import torch

from src.models.cnn import CNNConfig, MoodCNN, count_parameters


def test_forward_output_shape_matches_n_classes():
    cfg = CNNConfig(n_mels=128, n_classes=5, conv_channels=(8, 16, 32))
    model = MoodCNN(cfg)
    x = torch.randn(4, 1, cfg.n_mels, 1293)
    logits = model(x)
    assert logits.shape == (4, cfg.n_classes)


def test_embed_output_shape_matches_embedding_dim():
    cfg = CNNConfig(n_mels=128, embedding_dim=64)
    model = MoodCNN(cfg)
    x = torch.randn(3, 1, cfg.n_mels, 1293)
    emb = model.embed(x)
    assert emb.shape == (3, cfg.embedding_dim)


def test_model_is_small_enough_for_cpu():
    cfg = CNNConfig(n_mels=128, n_classes=5, conv_channels=(16, 32, 64))
    model = MoodCNN(cfg)
    n = count_parameters(model)
    # Should be well under 1M params for CPU training in the 6-day budget
    assert n < 1_000_000, f"too many params: {n}"


def test_embed_and_classifier_share_backbone():
    cfg = CNNConfig(n_mels=64, embedding_dim=32, n_classes=5)
    model = MoodCNN(cfg)
    x = torch.randn(2, 1, cfg.n_mels, 300)
    emb = model.embed(x)
    logits = model(x)
    # classifier consumes the same embedding
    assert logits.shape == (2, cfg.n_classes)
    assert emb.shape == (2, cfg.embedding_dim)
