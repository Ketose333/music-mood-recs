"""Download MTG-Jamendo audio-low TARs and extract only the subset tracks.

The mood/theme audio-low set is ~46GB across 100 TARs (folders 00-99). Subset
tracks are spread across all folders, so selective TAR download only helps
when we restrict the subset to a prefix of folders (``--max-tars``).

Workflow:
  1. Build the top-N tag subset from split-0 metadata (src.data.load_jamendo).
  2. Download TARs 00..max_tars-1 from the MTG-fast mirror, one at a time.
  3. Extract only tracks whose PATH is in the subset; delete the TAR after.
  4. Save the restricted subset metadata to artifacts/subset_meta.csv so the
     training pipeline only references tracks that were actually downloaded.

Usage:
  python scripts/download_audio.py --top-n 5 --max-tars 20 --out data/audio
  python scripts/download_audio.py --top-n 5 --max-tars 100 --out data/audio

  # Upload extracted tracks straight to a HF dataset repo instead of --out
  # (requires `hf auth login` with write access to the repo beforehand):
  python scripts/download_audio.py --top-n 5 --max-tars 100 \
      --hf-repo-id Ketose333/music-mood-recs-assets
"""

from __future__ import annotations

import argparse
import os
import sys

from src.data.download_audio import download_and_extract_subset, restrict_subset_to_folders
from src.data.load_jamendo import build_subset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-n", type=int, default=5, help="number of top mood/theme tags")
    parser.add_argument("--max-tars", type=int, default=100, help="download TARs 00..max_tars-1 (100=all)")
    parser.add_argument("--out", default="data/audio", help="output directory for extracted mp3s")
    parser.add_argument("--meta-out", default="artifacts/subset_meta.csv", help="restricted subset metadata CSV")
    parser.add_argument("--dest-dir", default="data/jamendo", help="jamendo metadata cache dir")
    parser.add_argument("--parallel", type=int, default=3, help="number of TARs to download in parallel")
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="if set, upload extracted tracks straight to this HF dataset repo instead of --out "
        "(requires `hf auth login` with write access)",
    )
    parser.add_argument(
        "--hf-path-prefix",
        default="data/audio",
        help="path prefix inside the HF repo for uploaded tracks (only used with --hf-repo-id)",
    )
    parser.add_argument(
        "--keep-local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also write extracted tracks under --out even when --hf-repo-id is set (default: on). "
        "Use --no-keep-local for a disk-constrained bulk migration that only needs the HF copy.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_out), exist_ok=True)

    print(f"Building subset (top_n={args.top_n}) from split-0 metadata...", flush=True)
    subset = build_subset(top_n=args.top_n, dest_dir=args.dest_dir)
    print(f"  tags: {subset['tags']}", flush=True)
    for s in ("train", "validation", "test"):
        print(f"  {s}: {len(subset[s])}", flush=True)

    if args.max_tars < 100:
        print(f"Restricting subset to folders 00..{args.max_tars - 1:02d} (--max-tars {args.max_tars})")
        subset = restrict_subset_to_folders(subset, args.max_tars)
        for s in ("train", "validation", "test"):
            print(f"  {s} (restricted): {len(subset[s])}")

    download_and_extract_subset(
        subset,
        args.out,
        args.max_tars,
        args.parallel,
        hf_repo_id=args.hf_repo_id,
        hf_path_prefix=args.hf_path_prefix,
        keep_local=args.keep_local,
    )

    import pandas as pd

    combined = pd.concat(
        [subset[s].assign(split=s) for s in ("train", "validation", "test")],
        ignore_index=True,
    )
    combined.to_csv(args.meta_out, index=False)
    print(f"Saved restricted subset metadata ({len(combined)} rows) to {args.meta_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
