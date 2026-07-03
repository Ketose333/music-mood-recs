"""Tests for src/recommend/preview_rank.py — iTunes preview embedding and
DL re-ranking of real-track recommendations. All HTTP is monkeypatched; the
"downloaded" preview is a generated wav so the melspec->CNN path runs for
real without network access."""

import io

import numpy as np
import pytest
import requests
import soundfile as sf
import torch

from src.models.cnn import CNNConfig, MoodCNN
from src.recommend import preview_rank as pr


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = MoodCNN(CNNConfig())
    m.eval()
    return m


def _wav_bytes(seconds: float = 1.0, sr: int = 22050) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


class TestEmbedPreview:
    def test_empty_url_returns_none(self, model):
        assert pr.embed_preview("", model) is None

    def test_download_failure_returns_none(self, model, monkeypatch):
        def _boom(url, timeout=None):
            raise requests.ConnectionError("no network")

        monkeypatch.setattr(pr.requests, "get", _boom)
        assert pr.embed_preview("https://example.com/p.m4a", model) is None

    def test_undecodable_audio_returns_none(self, model, monkeypatch):
        monkeypatch.setattr(
            pr.requests, "get", lambda url, timeout=None: _FakeResponse(b"not audio at all")
        )
        assert pr.embed_preview("https://example.com/p.m4a", model) is None

    def test_valid_preview_returns_embedding(self, model, monkeypatch):
        wav = _wav_bytes()
        monkeypatch.setattr(pr.requests, "get", lambda url, timeout=None: _FakeResponse(wav))
        emb = pr.embed_preview("https://example.com/preview.wav", model)
        assert emb is not None
        assert emb.shape == (model.cfg.embedding_dim,)
        assert emb.dtype == np.float32


class TestPreviewSimilarities:
    def test_identical_embedding_gives_similarity_one(self):
        q = np.array([1.0, 0.0, 2.0], dtype=np.float32)
        sims = pr.preview_similarities(q, [q.copy()])
        assert sims[0] == pytest.approx(1.0)

    def test_none_entries_stay_none_and_positions_align(self):
        q = np.array([1.0, 0.0], dtype=np.float32)
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        sims = pr.preview_similarities(q, [None, a, None, b])
        assert sims[0] is None and sims[2] is None
        assert sims[1] == pytest.approx(1.0)
        assert sims[3] == pytest.approx(0.0)


class TestRerankWithSimilarity:
    def test_sorts_scored_desc_and_appends_unscored_in_order(self):
        items = ["low", "no-preview-1", "high", "no-preview-2"]
        sims = [0.1, None, 0.9, None]
        ranked = pr.rerank_with_similarity(items, sims)
        assert [item for item, _ in ranked] == ["high", "low", "no-preview-1", "no-preview-2"]
        assert [s for _, s in ranked] == [0.9, 0.1, None, None]

    def test_all_none_keeps_original_order(self):
        items = ["a", "b", "c"]
        ranked = pr.rerank_with_similarity(items, [None, None, None])
        assert [item for item, _ in ranked] == items
