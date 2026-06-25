"""Simple CNN mood classifier on 30s log-mel spectrograms.

Architecture: 3 conv blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool) followed
by a global average pool, a small embedding head (used for both classification
and cosine-similarity recommendation), and a linear classifier. Kept small on
purpose for CPU training within the 6-day deadline.

Input: (batch, 1, n_mels, n_frames) float32 — log-mel spectrogram.
Output: logits (batch, n_classes).

The embedding vector (output of ``embed``) is what the recommendation step
reuses via cosine similarity (PRD §7, PR-003).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CNNConfig:
    n_mels: int = 128
    n_classes: int = 5
    embedding_dim: int = 64
    conv_channels: tuple[int, int, int] = (16, 32, 64)
    kernel_size: int = 3
    dropout: float = 0.3


class MoodCNN(nn.Module):
    def __init__(self, cfg: CNNConfig | None = None):
        super().__init__()
        self.cfg = cfg or CNNConfig()
        c1, c2, c3 = self.cfg.conv_channels

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, self.cfg.kernel_size, padding=self.cfg.kernel_size // 2),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c1, c2, self.cfg.kernel_size, padding=self.cfg.kernel_size // 2),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c2, c3, self.cfg.kernel_size, padding=self.cfg.kernel_size // 2),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, self.cfg.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
        )
        self.classifier = nn.Linear(self.cfg.embedding_dim, self.cfg.n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the embedding vector (batch, embedding_dim) used for recommendation."""
        h = self.features(x)
        h = self.pool(h)
        return self.embed_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        return self.classifier(z)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
