"""Mel-spectrogram preprocessing for MTG-Jamendo mood classification.

Each track is trimmed/padded to a fixed 30-second segment and converted to a
log-mel spectrogram (n_mels=128, sr=22050, hop_length=512 → ~1293 frames for
30s). Output is a 2D numpy array (n_mels, n_frames). The segment offset is
configurable (start or center) to keep inference deterministic.

Per PRD §17.1: 곡당 30초 세그먼트 1개, log-mel, CPU 추론/학습 시간 절감.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np


@dataclass(frozen=True)
class MelspecConfig:
    sr: int = 22050
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    segment_seconds: float = 30.0
    offset_mode: str = "start"  # "start" or "center"
    top_db: float = 80.0

    @property
    def segment_samples(self) -> int:
        return int(self.sr * self.segment_seconds)

    @property
    def expected_frames(self) -> int:
        # librosa default center=True pads n_fft//2 on each side, so
        # n_frames = 1 + n_samples // hop_length.
        return 1 + self.segment_samples // self.hop_length


def load_segment(
    audio_path: str, cfg: MelspecConfig, duration_cap: Optional[float] = None
) -> np.ndarray:
    """Load a fixed 30-second mono segment as a 1D float32 numpy array."""
    total_duration = duration_cap
    if total_duration is None:
        try:
            total_duration = float(librosa.get_duration(path=audio_path))
        except Exception:
            total_duration = cfg.segment_seconds
    if cfg.offset_mode == "center" and total_duration > cfg.segment_seconds:
        offset = (total_duration - cfg.segment_seconds) / 2.0
    else:
        offset = 0.0
    y, _ = librosa.load(
        audio_path,
        sr=cfg.sr,
        mono=True,
        offset=offset,
        duration=cfg.segment_seconds,
    )
    target = cfg.segment_samples
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    elif len(y) > target:
        y = y[:target]
    return y.astype(np.float32)


def compute_melspec(y: np.ndarray, cfg: MelspecConfig) -> np.ndarray:
    """Compute a log-mel spectrogram (n_mels, n_frames) from a 1D waveform."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=cfg.top_db)
    return log_mel.astype(np.float32)


def extract_melspec(
    audio_path: str, cfg: Optional[MelspecConfig] = None
) -> np.ndarray:
    """Load segment + compute log-mel spectrogram in one call."""
    cfg = cfg or MelspecConfig()
    y = load_segment(audio_path, cfg)
    return compute_melspec(y, cfg)


def save_melspec(
    audio_path: str, cache_path: str, cfg: Optional[MelspecConfig] = None
) -> np.ndarray:
    """Compute and cache a mel-spectrogram to ``cache_path`` (.npy). Returns the array."""
    cfg = cfg or MelspecConfig()
    mel = extract_melspec(audio_path, cfg)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, mel)
    return mel


def load_or_compute_melspec(
    audio_path: str, cache_path: str, cfg: Optional[MelspecConfig] = None
) -> np.ndarray:
    """Load cached .npy if present, otherwise compute and save."""
    if os.path.exists(cache_path):
        return np.load(cache_path)
    return save_melspec(audio_path, cache_path, cfg)


def extract_subset_melspecs(
    meta_csv: str, audio_dir: str, out_dir: str, cfg: Optional[MelspecConfig] = None
):
    """Batch-extract melspecs for every track in a subset metadata CSV.

    Reads TRACK_ID/PATH/DURATION/split columns from ``meta_csv``, skips tracks
    whose audio file is missing under ``audio_dir``, and caches each
    melspec under ``out_dir`` (already-cached .npy files are reused). Returns
    ``(manifest_df, missing_count)``; the caller writes the manifest CSV.
    """
    import pandas as pd

    cfg = cfg or MelspecConfig()
    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(meta_csv)

    rows: list[dict] = []
    missing = 0
    for _, row in meta.iterrows():
        rel = row["PATH"]
        audio_path = os.path.join(audio_dir, rel)
        if not os.path.exists(audio_path):
            missing += 1
            continue
        cache_path = os.path.join(out_dir, rel.replace("/", os.sep).removesuffix(".mp3") + ".npy")
        mel = load_or_compute_melspec(audio_path, cache_path, cfg)
        rows.append(
            {
                "TRACK_ID": row["TRACK_ID"],
                "PATH": rel,
                "DURATION": row["DURATION"],
                "split": row["split"],
                "npy_path": cache_path,
                "n_mels": mel.shape[0],
                "n_frames": mel.shape[1],
            }
        )
    return pd.DataFrame(rows), missing


def validate_melspec(mel: np.ndarray, cfg: Optional[MelspecConfig] = None) -> bool:
    """Check shape and dtype match the config (used in tests / inference guards)."""
    cfg = cfg or MelspecConfig()
    if mel.ndim != 2 or mel.dtype != np.float32:
        return False
    n_mels, n_frames = mel.shape
    if n_mels != cfg.n_mels:
        return False
    # Frame count may differ slightly depending on librosa version / padding,
    # so we accept the expected value or expected±1.
    return abs(n_frames - cfg.expected_frames) <= 1
