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

from src.preprocessing.melspec import MelspecConfig, extract_subset_melspecs

_DEFAULT_CFG = MelspecConfig()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta", default="artifacts/subset_meta.csv", help="subset metadata CSV")
    parser.add_argument("--audio-dir", default="data/audio", help="extracted mp3 root")
    parser.add_argument("--out", default="artifacts/melspecs", help="output dir for .npy files")
    parser.add_argument("--manifest", default="artifacts/melspec_manifest.csv", help="manifest CSV output")
    parser.add_argument("--sr", type=int, default=_DEFAULT_CFG.sr)
    parser.add_argument("--n-mels", type=int, default=_DEFAULT_CFG.n_mels)
    parser.add_argument("--segment-seconds", type=float, default=_DEFAULT_CFG.segment_seconds)
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="If set, upload any melspec .npy not yet on this HF Hub dataset repo "
        "(keeps the deployed app's HF copy in sync after a TAR-count bump).",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    cfg = MelspecConfig(
        sr=args.sr, n_mels=args.n_mels, segment_seconds=args.segment_seconds
    )

    manifest, missing = extract_subset_melspecs(args.meta, args.audio_dir, args.out, cfg)
    manifest.to_csv(args.manifest, index=False)
    print(f"Manifest: {len(manifest)} tracks -> {args.manifest}")
    print(f"Missing audio files skipped: {missing}")

    if args.hf_repo_id:
        from src.data.hf_sync import upload_missing_files

        n = upload_missing_files(args.hf_repo_id, manifest["npy_path"].tolist())
        print(f"Uploaded {n} new melspec files to {args.hf_repo_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
