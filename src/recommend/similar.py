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


def top_k_similar_to_vector(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Same ranking as ``top_k_similar`` but for a query embedding that is not
    already a row of ``embeddings`` (e.g. a user-uploaded track)."""
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(query, embeddings).ravel()
    order = np.argsort(-sims)
    top = order[:k]
    return top, sims[top]


@torch.no_grad()
def predict_mood_probs(model: MoodCNN, embeddings: np.ndarray) -> np.ndarray:
    """Run the trained classifier head on precomputed embeddings to get each
    track's mood probabilities without re-loading any mel-spectrogram.

    ``model.embed`` already produced ``embeddings``, so this only replays the
    final linear layer (``model.classifier``) — cheap enough to run over the
    whole library at app start-up.
    """
    model.eval()
    z = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    logits = model.classifier(z)
    return torch.sigmoid(logits).numpy()


# Korean keyword -> mood tag lookup for the free-text mood search. This is a
# small heuristic bridge (not a trained NLP model): the project's only
# trained model classifies *audio*, so a typed feeling has to be mapped onto
# the same 5 tags before it can be used to rank the library by
# ``predict_mood_probs``.
MOOD_KEYWORDS: dict[str, list[str]] = {
    "happy": ["행복", "기쁘", "기뻐", "즐겁", "신나", "설레", "웃음", "웃기", "들뜨", "유쾌"],
    "energetic": ["에너지", "파워", "운동", "헬스", "달리", "파이팅", "격렬", "흥분", "활기", "신남"],
    "relaxing": ["편안", "휴식", "잠", "졸려", "힐링", "차분", "여유", "평화", "느긋", "안정"],
    "film": ["영화", "웅장", "드라마틱", "서사", "영상", "스토리", "장엄", "긴장감", "몰입"],
    "dark": ["우울", "슬픔", "슬프", "어둡", "외롭", "쓸쓸", "그리움", "눈물", "힘들", "지치", "불안"],
}


def infer_mood_from_text(text: str, tags: list[str]) -> tuple[str | None, dict[str, int]]:
    """Score ``text`` against each tag's keyword list and return the
    best-matching tag (``None`` if no keyword hits at all) plus the raw
    per-tag hit counts, so the caller can show why a tag was picked.
    """
    counts = {tag: 0 for tag in tags}
    for tag in tags:
        for kw in MOOD_KEYWORDS.get(tag, []):
            if kw in text:
                counts[tag] += 1
    best_tag = max(counts, key=counts.get) if counts else None
    if best_tag is not None and counts[best_tag] == 0:
        best_tag = None
    return best_tag, counts
