"""DL re-ranking of real-track recommendations via iTunes preview clips.

The LLM proposes released songs and iTunes verifies they exist, but until now
the *order* of the real-track Top-5 had nothing to do with the trained model.
This module closes that gap: each verified track's official 30-second iTunes
preview clip (the ``previewUrl`` Apple serves publicly for exactly this kind
of sampling) is run through the same melspec -> MoodCNN.embed path as every
library track, and the resulting embedding is compared to the query track's
embedding by cosine similarity — so the DL model ranks the real-world
candidates too, not just the library.

Copyright posture is unchanged from music_search.py: the preview is fetched
transiently for feature extraction only — written to a temp file solely
because librosa needs a path, deleted immediately after, never stored and
never played in the app.

Every failure mode (no preview URL, download error, undecodable audio) yields
``None`` instead of raising, and ``rerank_with_similarity`` keeps such tracks
in their original position after the scored ones — graceful degradation, same
as the LLM fallback chain.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import urllib.parse
from typing import TypeVar

import numpy as np
import requests
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.models.cnn import MoodCNN
from src.preprocessing.melspec import MelspecConfig, extract_melspec

PREVIEW_TIMEOUT = 15
FFMPEG_TIMEOUT = 60

T = TypeVar("T")


def _decode_with_ffmpeg(src_path: str) -> str | None:
    """Transcode an audio file to a temp wav; returns the wav path or None.

    iTunes previews are AAC/m4a, which neither libsndfile nor audioread can
    decode without a system ffmpeg. imageio-ffmpeg ships a static ffmpeg
    binary through pip (no apt/brew needed — works on Streamlit Cloud and
    local Windows alike), so we shell out to it only when librosa's own
    loaders have already failed. The caller deletes the returned wav.
    """
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = get_ffmpeg_exe()
    except Exception:
        return None
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    try:
        subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", src_path, "-ac", "1", "-ar", "22050", wav_path],
            check=True,
            capture_output=True,
            timeout=FFMPEG_TIMEOUT,
        )
    except Exception:
        os.remove(wav_path)
        return None
    return wav_path


def melspec_from_preview(preview_url: str, n_mels: int = 128) -> np.ndarray | None:
    """Download one iTunes preview clip and return its log-mel spectrogram
    (the same array shape ``extract_melspec`` produces for a library track),
    or None on any failure (no URL, network error, undecodable audio).

    Split out from ``embed_preview`` so callers that also need the trained
    classifier's mood probabilities (not just the embedding) — e.g. the
    "search a released song" input mode — can run one download+decode and
    feed the resulting mel into ``model.embed``/``model.classifier`` both,
    exactly like ``app.py``'s uploaded-audio path already does.
    """
    if not preview_url:
        return None
    try:
        resp = requests.get(preview_url, timeout=PREVIEW_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    suffix = os.path.splitext(urllib.parse.urlparse(preview_url).path)[1] or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name
    try:
        try:
            return extract_melspec(tmp_path, MelspecConfig(n_mels=n_mels))
        except Exception:
            # librosa can't decode AAC/m4a directly — transcode via the
            # pip-bundled ffmpeg and retry once; still failing → give up.
            wav_path = _decode_with_ffmpeg(tmp_path)
            if wav_path is None:
                return None
            try:
                return extract_melspec(wav_path, MelspecConfig(n_mels=n_mels))
            except Exception:
                return None
            finally:
                os.remove(wav_path)
    finally:
        os.remove(tmp_path)


@torch.no_grad()
def embed_preview(preview_url: str, model: MoodCNN, n_mels: int = 128) -> np.ndarray | None:
    """Download one preview clip and return its CNN embedding, or None.

    The clip (~30s AAC/m4a) goes through the exact pipeline used for library
    tracks (``extract_melspec`` pads/trims to the fixed 30s segment), so the
    embedding lives in the same space as ``artifacts/embeddings.npy`` and is
    directly comparable by cosine similarity.
    """
    mel = melspec_from_preview(preview_url, n_mels=n_mels)
    if mel is None:
        return None
    model.eval()
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
    z = model.embed(x)
    return z[0].numpy()


def preview_similarities(
    query_embedding: np.ndarray, preview_embeddings: list[np.ndarray | None]
) -> list[float | None]:
    """Cosine similarity of each preview embedding to the query embedding,
    positionally aligned with the input; None entries stay None."""
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    sims: list[float | None] = []
    for emb in preview_embeddings:
        if emb is None:
            sims.append(None)
            continue
        vec = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        sims.append(float(cosine_similarity(query, vec)[0, 0]))
    return sims


def rerank_with_similarity(items: list[T], sims: list[float | None]) -> list[tuple[T, float | None]]:
    """Sort ``items`` by similarity (desc). Unscored items (sim None) keep
    their original relative order and follow the scored ones, so a missing
    preview never buries a track the LLM ranked high on its own."""
    scored = [(item, s) for item, s in zip(items, sims) if s is not None]
    unscored = [(item, s) for item, s in zip(items, sims) if s is None]
    scored.sort(key=lambda pair: -pair[1])
    return scored + unscored
