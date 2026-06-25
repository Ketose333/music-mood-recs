"""Embedding extraction and cosine-similarity recommendation.

Reuses the ``MoodCNN.embed`` head (PRD §7, PR-003) to produce a fixed-size
embedding per track, then ranks tracks by cosine similarity to a query track.
No separate recommendation model is trained — the classification embedding
is reused directly.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.models.cnn import MoodCNN


@torch.no_grad()
def extract_embeddings(
    model: MoodCNN,
    mels: np.ndarray,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Compute embeddings for an array of mel-spectrograms.

    ``mels``: (N, n_mels, n_frames) float32. Returns (N, embedding_dim) float32.
    """
    model.eval()
    model.to(device)
    out: list[np.ndarray] = []
    for i in range(0, len(mels), batch_size):
        batch = mels[i : i + batch_size]
        x = torch.from_numpy(batch).unsqueeze(1).to(device)
        z = model.embed(x)
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


def top_k_similar(
    query_idx: int,
    embeddings: np.ndarray,
    k: int = 5,
    exclude_query: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, similarities) of the top-k most similar tracks.

    Cosine similarity is computed between the query embedding and all others.
    If ``exclude_query`` is True, the query itself is removed from the results.
    """
    query = embeddings[query_idx : query_idx + 1]
    sims = cosine_similarity(query, embeddings).ravel()
    order = np.argsort(-sims)
    if exclude_query:
        order = order[order != query_idx]
    top = order[:k]
    return top, sims[top]
