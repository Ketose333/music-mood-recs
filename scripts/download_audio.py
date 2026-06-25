"""Download MTG-Jamendo audio-low TARs and extract only the subset tracks.

The mood/theme audio-low set is ~46GB across 100 TARs (folders 00-99). Subset
tracks are spread across all folders, so selective TAR download only helps
when we restrict the subset to a prefix of folders (``--max-tars``).

Workflow:
  1. Build the top-N tag subset from split-0 metadata (src.data.jamendo).
  2. Download TARs 00..max_tars-1 from the MTG-fast mirror, one at a time.
  3. Extract only tracks whose PATH is in the subset; delete the TAR after.
  4. Save the restricted subset metadata to artifacts/subset_meta.csv so the
     training pipeline only references tracks that were actually downloaded.

Usage:
  python scripts/download_audio.py --top-n 5 --max-tars 20 --out data/audio
  python scripts/download_audio.py --top-n 5 --max-tars 100 --out data/audio
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from src.data.jamendo import build_subset

_MIRROR = "https://cdn.freesound.org/mtg-jamendo/autotagging_moodtheme/audio-low"
_TAR_TEMPLATE = "autotagging_moodtheme_audio-low-{idx:02d}.tar"


def _subset_path_set(subset: dict[str, pd.DataFrame]) -> set[str]:
    paths: set[str] = set()
    for split in ("train", "validation", "test"):
        paths.update(subset[split]["PATH"].tolist())
    return paths


def _restrict_subset_to_folders(
    subset: dict[str, pd.DataFrame], max_tars: int
) -> dict[str, pd.DataFrame]:
    allowed = {f"{i:02d}" for i in range(max_tars)}

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        folder = df["PATH"].str.split("/").str[0]
        return df.loc[folder.isin(allowed)].reset_index(drop=True)

    return {k: (_filter(v) if k in ("train", "validation", "test") else v) for k, v in subset.items()}


def _download_tar(idx: int, dest_dir: str, max_retries: int = 5) -> str:
    filename = _TAR_TEMPLATE.format(idx=idx)
    url = f"{_MIRROR}/{filename}"
    dest = os.path.join(dest_dir, filename)
    # Expected size from a previous HEAD or the known per-TAR size (~493MB).
    # We treat a file as complete only if size > 400MB (these TARs are ~490-517MB).
    min_complete_bytes = 400 * 1024 * 1024
    if os.path.exists(dest) and os.path.getsize(dest) >= min_complete_bytes:
        print(f"  TAR {filename} already complete ({os.path.getsize(dest)//1024//1024}MB), skipping", flush=True)
        return dest
    # Remove any partial file from a crashed run before retrying
    if os.path.exists(dest):
        partial_mb = os.path.getsize(dest) // 1024 // 1024
        print(f"  Removing partial {filename} ({partial_mb}MB)", flush=True)
        os.remove(dest)
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Downloading {url} (attempt {attempt}/{max_retries})", flush=True)
            with requests.get(url, stream=True, timeout=(15, 120)) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=512 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total and downloaded % (20 * 1024 * 1024) < len(chunk):
                                pct = 100 * downloaded / total if total else 0
                                print(f"    {filename}: {downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB ({pct:.0f}%)", flush=True)
            final_size = os.path.getsize(dest)
            if final_size >= min_complete_bytes:
                print(f"  TAR {filename} complete ({final_size//1024//1024}MB)", flush=True)
                return dest
            print(f"  TAR {filename} incomplete ({final_size//1024//1024}MB), retrying", flush=True)
            os.remove(dest)
        except (requests.exceptions.RequestException, ConnectionError) as e:
            print(f"  TAR {filename} attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
            if os.path.exists(dest):
                os.remove(dest)
            if attempt < max_retries:
                import time
                wait = 10 * attempt
                print(f"  waiting {wait}s before retry", flush=True)
                time.sleep(wait)
    raise RuntimeError(f"Failed to download {filename} after {max_retries} attempts")


def _extract_subset_from_tar(
    tar_path: str, out_dir: str, wanted_paths: set[str]
) -> int:
    extracted = 0
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if m.name in wanted_paths and m.isfile()]
        for m in members:
            tar.extract(m, path=out_dir)
            extracted += 1
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-n", type=int, default=5, help="number of top mood/theme tags")
    parser.add_argument("--max-tars", type=int, default=100, help="download TARs 00..max_tars-1 (100=all)")
    parser.add_argument("--out", default="data/audio", help="output directory for extracted mp3s")
    parser.add_argument("--meta-out", default="artifacts/subset_meta.csv", help="restricted subset metadata CSV")
    parser.add_argument("--dest-dir", default="data/jamendo", help="jamendo metadata cache dir")
    parser.add_argument("--parallel", type=int, default=3, help="number of TARs to download in parallel")
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
        subset = _restrict_subset_to_folders(subset, args.max_tars)
        for s in ("train", "validation", "test"):
            print(f"  {s} (restricted): {len(subset[s])}")

    wanted = _subset_path_set(subset)
    print(f"Total unique audio paths to extract: {len(wanted)}", flush=True)

    # Phase 1: download all TARs in parallel (keep tar files on disk)
    tar_indices = list(range(args.max_tars))
    print(f"Downloading {len(tar_indices)} TARs with parallelism={args.parallel}", flush=True)
    tar_paths: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(_download_tar, idx, args.out): idx for idx in tar_indices}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                tar_paths[idx] = fut.result()
                print(f"  TAR {idx:02d} downloaded", flush=True)
            except Exception as e:
                print(f"  TAR {idx:02d} FAILED: {e}", flush=True)

    # Phase 2: extract subset tracks from each TAR in order, then delete the tar
    total_extracted = 0
    for idx in tar_indices:
        if idx not in tar_paths:
            print(f"  TAR {idx:02d}: skipped (download failed)", flush=True)
            continue
        n = _extract_subset_from_tar(tar_paths[idx], args.out, wanted)
        total_extracted += n
        print(f"  TAR {idx:02d}: extracted {n} tracks (cumulative {total_extracted})", flush=True)
        os.remove(tar_paths[idx])

    print(f"\nExtracted {total_extracted} / {len(wanted)} subset tracks to {args.out}")

    combined = pd.concat(
        [subset[s].assign(split=s) for s in ("train", "validation", "test")],
        ignore_index=True,
    )
    combined.to_csv(args.meta_out, index=False)
    print(f"Saved restricted subset metadata ({len(combined)} rows) to {args.meta_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
