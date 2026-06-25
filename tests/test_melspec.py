"""Tests for src.preprocessing.melspec — uses synthesized WAV files (no network)."""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from src.preprocessing.melspec import (
    MelspecConfig,
    compute_melspec,
    extract_melspec,
    extract_subset_melspecs,
    load_or_compute_melspec,
    load_segment,
    validate_melspec,
)


def _write_wav(path: str, seconds: float, sr: int = 22050, freq: float = 440.0) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, y, sr)


def test_load_segment_pads_short_audio(tmp_path):
    cfg = MelspecConfig(sr=22050, segment_seconds=1.0)
    wav = tmp_path / "short.wav"
    _write_wav(str(wav), seconds=0.3, sr=cfg.sr)
    y = load_segment(str(wav), cfg)
    assert y.shape == (cfg.segment_samples,)
    assert y.dtype == np.float32
    # Tail is zero-padded
    assert np.allclose(y[int(cfg.sr * 0.3):], 0.0)


def test_load_segment_trims_long_audio(tmp_path):
    cfg = MelspecConfig(sr=22050, segment_seconds=1.0)
    wav = tmp_path / "long.wav"
    _write_wav(str(wav), seconds=2.0, sr=cfg.sr)
    y = load_segment(str(wav), cfg)
    assert y.shape == (cfg.segment_samples,)


def test_compute_melspec_shape_and_dtype():
    cfg = MelspecConfig(sr=22050, segment_seconds=1.0, n_mels=64)
    y = np.random.RandomState(0).randn(cfg.segment_samples).astype(np.float32) * 0.01
    mel = compute_melspec(y, cfg)
    assert mel.ndim == 2
    assert mel.shape[0] == cfg.n_mels
    assert mel.dtype == np.float32


def test_extract_melspec_end_to_end(tmp_path):
    cfg = MelspecConfig(sr=22050, segment_seconds=1.0, n_mels=64)
    wav = tmp_path / "tone.wav"
    _write_wav(str(wav), seconds=1.5, sr=cfg.sr)
    mel = extract_melspec(str(wav), cfg)
    assert mel.shape[0] == cfg.n_mels
    assert validate_melspec(mel, cfg)


def test_load_or_compute_melspec_caches(tmp_path):
    cfg = MelspecConfig(sr=22050, segment_seconds=1.0, n_mels=64)
    wav = tmp_path / "tone.wav"
    _write_wav(str(wav), seconds=1.5, sr=cfg.sr)
    cache = tmp_path / "cache" / "tone.npy"
    first = load_or_compute_melspec(str(wav), str(cache), cfg)
    assert os.path.exists(str(cache))
    second = load_or_compute_melspec(str(wav), str(cache), cfg)
    np.testing.assert_array_equal(first, second)


def test_validate_melspec_rejects_wrong_shape():
    cfg = MelspecConfig(n_mels=128)
    bad = np.zeros((64, 100), dtype=np.float32)
    assert not validate_melspec(bad, cfg)


def test_extract_subset_melspecs_skips_missing_and_builds_manifest(tmp_path):
    cfg = MelspecConfig(sr=22050, segment_seconds=1.0, n_mels=64)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(str(audio_dir / "present.wav"), seconds=1.2, sr=cfg.sr)

    meta_csv = tmp_path / "subset_meta.csv"
    meta_csv.write_text(
        "TRACK_ID,PATH,DURATION,split\n"
        "1,present.wav,1.2,train\n"
        "2,missing.wav,1.2,train\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "melspecs"
    manifest, missing = extract_subset_melspecs(str(meta_csv), str(audio_dir), str(out_dir), cfg)

    assert missing == 1
    assert len(manifest) == 1
    assert manifest.iloc[0]["TRACK_ID"] == 1
    assert os.path.exists(manifest.iloc[0]["npy_path"])
