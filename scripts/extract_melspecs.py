"""Batch mel-spectrogram extraction from downloaded MTG-Jamendo audio.

Reads ``artifacts/subset_meta.csv`` (produced by download_audio.py), locates
each track's mp3 under ``--audio-dir``, computes a 30s log-mel spectrogram,
and saves it as ``.npy`` under ``--out``. A manifest CSV mapping track_id,
split, tags, and npy path is written for the training pipeline.

Usage:
  python scripts/extract_melspecs.py --audio-dir data/audio --out artifacts/melspecs
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from src.preprocess.melspec import MelspecConfig, load_or_compute_melspec

_DEFAULT_CFG = MelspecConfig()


def _audio_path(audio_dir: str, rel_path: str) -> str:
    return os.path.join(audio_dir, rel_path)


def _cache_path(out_dir: str, rel_path: str) -> str:
    base = rel_path.replace("/", os.sep).removesuffix(".mp3")
    return os.path.join(out_dir, base + ".npy")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta", default="artifacts/subset_meta.csv", help="subset metadata CSV")
    parser.add_argument("--audio-dir", default="data/audio", help="extracted mp3 root")
    parser.add_argument("--out", default="artifacts/melspecs", help="output dir for .npy files")
    parser.add_argument("--manifest", default="artifacts/melspec_manifest.csv", help="manifest CSV output")
    parser.add_argument("--sr", type=int, default=_DEFAULT_CFG.sr)
    parser.add_argument("--n-mels", type=int, default=_DEFAULT_CFG.n_mels)
    parser.add_argument("--segment-seconds", type=float, default=_DEFAULT_CFG.segment_seconds)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    cfg = MelspecConfig(
        sr=args.sr, n_mels=args.n_mels, segment_seconds=args.segment_seconds
    )

    meta = pd.read_csv(args.meta)
    print(f"Loaded {len(meta)} rows from {args.meta}")

    rows: list[dict] = []
    missing = 0
    for i, row in meta.iterrows():
        rel = row["PATH"]
        audio = _audio_path(args.audio_dir, rel)
        if not os.path.exists(audio):
            missing += 1
            continue
        cache = _cache_path(args.out, rel)
        mel = load_or_compute_melspec(audio, cache, cfg)
        rows.append(
            {
                "TRACK_ID": row["TRACK_ID"],
                "PATH": rel,
                "DURATION": row["DURATION"],
                "split": row["split"],
                "npy_path": cache,
                "n_mels": mel.shape[0],
                "n_frames": mel.shape[1],
            }
        )
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(meta)} (missing {missing})")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.manifest, index=False)
    print(f"\nManifest: {len(manifest)} tracks -> {args.manifest}")
    print(f"Missing audio files skipped: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
