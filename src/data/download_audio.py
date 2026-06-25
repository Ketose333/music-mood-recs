"""Download MTG-Jamendo audio-low TARs and extract only the subset tracks.

Shared by scripts/download_audio.py (CLI) and the submission notebook, so the
download/extract/incremental-skip logic lives in exactly one place.
"""

from __future__ import annotations

import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

_MIRROR = "https://cdn.freesound.org/mtg-jamendo/autotagging_moodtheme/audio-low"
_TAR_TEMPLATE = "autotagging_moodtheme_audio-low-{idx:02d}.tar"


def subset_path_set(subset: dict[str, pd.DataFrame]) -> set[str]:
    paths: set[str] = set()
    for split in ("train", "validation", "test"):
        paths.update(subset[split]["PATH"].tolist())
    return paths


def restrict_subset_to_folders(
    subset: dict[str, pd.DataFrame], max_tars: int
) -> dict[str, pd.DataFrame]:
    allowed = {f"{i:02d}" for i in range(max_tars)}

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        folder = df["PATH"].str.split("/").str[0]
        return df.loc[folder.isin(allowed)].reset_index(drop=True)

    return {k: (_filter(v) if k in ("train", "validation", "test") else v) for k, v in subset.items()}


def download_tar(idx: int, dest_dir: str, max_retries: int = 5) -> str:
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


def extract_subset_from_tar(tar_path: str, out_dir: str, wanted_paths: set[str]) -> int:
    extracted = 0
    with tarfile.open(tar_path) as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            # audio-low TAR members are named "<folder>/<id>.low.mp3"; the subset
            # metadata's PATH column uses "<folder>/<id>.mp3" (no ".low").
            rel = m.name[: -len(".low.mp3")] + ".mp3" if m.name.endswith(".low.mp3") else m.name
            if rel not in wanted_paths:
                continue
            src = tar.extractfile(m)
            if src is None:
                continue
            dest_path = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(src.read())
            extracted += 1
    return extracted


def download_and_extract_subset(
    subset: dict[str, pd.DataFrame], out_dir: str, max_tars: int, parallel: int = 3
) -> tuple[int, int]:
    """Download+extract TARs 00..max_tars-1, skipping folders whose subset
    tracks are already on disk. Returns (total_extracted, total_wanted)."""
    os.makedirs(out_dir, exist_ok=True)
    wanted = subset_path_set(subset)
    print(f"Total unique audio paths to extract: {len(wanted)}", flush=True)

    folder_wanted: dict[int, set[str]] = {}
    tar_indices: list[int] = []
    already_extracted = 0
    for idx in range(max_tars):
        folder = f"{idx:02d}"
        paths = {p for p in wanted if p.startswith(f"{folder}/")}
        folder_wanted[idx] = paths
        if paths and all(os.path.exists(os.path.join(out_dir, p)) for p in paths):
            already_extracted += len(paths)
            print(f"  TAR {idx:02d}: {len(paths)} subset tracks already extracted, skipping", flush=True)
            continue
        tar_indices.append(idx)

    print(
        f"Downloading {len(tar_indices)} TARs with parallelism={parallel} "
        f"({max_tars - len(tar_indices)} skipped, already extracted)",
        flush=True,
    )
    tar_paths: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(download_tar, idx, out_dir): idx for idx in tar_indices}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                tar_paths[idx] = fut.result()
                print(f"  TAR {idx:02d} downloaded", flush=True)
            except Exception as e:
                print(f"  TAR {idx:02d} FAILED: {e}", flush=True)

    total_extracted = already_extracted
    for idx in tar_indices:
        if idx not in tar_paths:
            print(f"  TAR {idx:02d}: skipped (download failed)", flush=True)
            continue
        n = extract_subset_from_tar(tar_paths[idx], out_dir, folder_wanted[idx])
        total_extracted += n
        print(f"  TAR {idx:02d}: extracted {n} tracks (cumulative {total_extracted})", flush=True)
        os.remove(tar_paths[idx])

    print(f"\nExtracted {total_extracted} / {len(wanted)} subset tracks to {out_dir}")
    return total_extracted, len(wanted)
