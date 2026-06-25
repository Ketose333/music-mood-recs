"""MTG-Jamendo mood/theme subset loader.

Downloads metadata TSVs (small) from the official MTG/mtg-jamendo-dataset
GitHub repo on first use, selects the top-N most frequent mood/theme tags,
and returns split-0 train/val/test DataFrames filtered to tracks that carry
at least one of the selected tags. Audio download (audio-low tar, ~46GB full
mood/theme subset) is out of scope here — ``list_selected_audio_paths``
returns the relative ``path`` values so a separate download step can fetch
only the needed files.

References:
- https://github.com/MTG/mtg-jamendo-dataset
- data/autotagging_moodtheme.tsv (18,486 tracks, 56 mood/theme tags)
- data/splits/split-0/autotagging_moodtheme-{train,validation,test}.tsv

Per PRD §17.1: 상위 5~8 태그, 약 1,000~2,000곡, 30초 세그먼트.
"""

from __future__ import annotations

import os
import urllib.request
from collections import Counter
from typing import Optional

import pandas as pd

_BASE_URL = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master"
_FULL_MOODTHEME = "data/autotagging_moodtheme.tsv"
_SPLIT_FILES = {
    "train": "data/splits/split-0/autotagging_moodtheme-train.tsv",
    "validation": "data/splits/split-0/autotagging_moodtheme-validation.tsv",
    "test": "data/splits/split-0/autotagging_moodtheme-test.tsv",
}
_TAG_PREFIX = "mood/theme---"
_COLUMNS = ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"]


def _download(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)


def _ensure_local(url_path: str, dest_dir: str) -> str:
    local = os.path.join(dest_dir, url_path)
    if not os.path.exists(local):
        _download(f"{_BASE_URL}/{url_path}", local)
    return local


def _parse_tags(raw: str) -> list[str]:
    return [t for t in raw.split("\t") if t]


def _mood_tags(tags: list[str]) -> list[str]:
    return [t[len(_TAG_PREFIX):] for t in tags if t.startswith(_TAG_PREFIX)]


def _load_tsv(path: str) -> pd.DataFrame:
    """Parse a MTG-Jamendo mood/theme TSV.

    The TAGS field is itself tab-separated and varies per row (multi-label),
    so a plain ``read_csv`` with fixed column names fails. We split each line
    on tabs: first 5 fields are fixed columns, the remaining fields are tags.
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        f.readline()  # skip header (TRACK_ID ... TAGS)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            rows.append(
                {
                    "TRACK_ID": parts[0],
                    "ARTIST_ID": parts[1],
                    "ALBUM_ID": parts[2],
                    "PATH": parts[3],
                    "DURATION": parts[4],
                    "TAGS": "\t".join(parts[5:]),
                }
            )
    df = pd.DataFrame(rows, columns=_COLUMNS, dtype=str)
    df["DURATION"] = pd.to_numeric(df["DURATION"], errors="coerce")
    df["tags_list"] = df["TAGS"].fillna("").apply(_parse_tags)
    df["mood_tags"] = df["tags_list"].apply(_mood_tags)
    return df


def load_full_moodtheme(dest_dir: str = "data/jamendo") -> pd.DataFrame:
    """Load the full autotagging_moodtheme.tsv (18,486 tracks, all 56 tags)."""
    path = _ensure_local(_FULL_MOODTHEME, dest_dir)
    return _load_tsv(path)


def load_split(split: str, dest_dir: str = "data/jamendo") -> pd.DataFrame:
    """Load one split-0 file (train/validation/test) for the mood/theme subset."""
    if split not in _SPLIT_FILES:
        raise ValueError(f"Unknown split: {split}. Expected one of {list(_SPLIT_FILES)}")
    path = _ensure_local(_SPLIT_FILES[split], dest_dir)
    return _load_tsv(path)


def select_top_mood_tags(
    df: pd.DataFrame, top_n: int = 8, min_tracks: int = 1
) -> list[str]:
    """Return the top-N mood/theme tags by track frequency.

    Counts a track once per tag (multi-label aware). Tags with fewer than
    ``min_tracks`` occurrences are dropped before taking the top-N.
    """
    counter: Counter = Counter()
    for tags in df["mood_tags"]:
        for tag in tags:
            counter[tag] += 1
    eligible = [(tag, n) for tag, n in counter.items() if n >= min_tracks]
    eligible.sort(key=lambda x: (-x[1], x[0]))
    return [tag for tag, _ in eligible[:top_n]]


def filter_to_selected_tags(
    df: pd.DataFrame, selected_tags: list[str]
) -> pd.DataFrame:
    """Keep tracks that carry at least one of ``selected_tags`` and add one-hot columns."""
    selected_set = set(selected_tags)
    has_any = df["mood_tags"].apply(lambda tags: any(t in selected_set for t in tags))
    out = df.loc[has_any].copy().reset_index(drop=True)
    for tag in selected_tags:
        out[f"tag_{tag}"] = out["mood_tags"].apply(lambda tags: int(tag in tags)).astype("int8")
    out["n_selected_tags"] = out[[f"tag_{t}" for t in selected_tags]].sum(axis=1)
    return out


def build_subset(
    top_n: int = 8,
    dest_dir: str = "data/jamendo",
    splits: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """End-to-end: load splits, pick top-N tags from TRAIN only, filter all splits.

    Tag selection uses the TRAIN split to avoid test leakage. Returns
    ``{"train": df, "validation": df, "test": df, "tags": [...]}``-shaped dict
    where ``tags`` is the list of selected mood/theme tags.
    """
    splits = splits or ["train", "validation", "test"]
    raw = {s: load_split(s, dest_dir) for s in splits}
    selected = select_top_mood_tags(raw["train"], top_n=top_n)
    filtered = {s: filter_to_selected_tags(raw[s], selected) for s in splits}
    return {"tags": selected, **filtered}


def list_selected_audio_paths(df: pd.DataFrame) -> list[str]:
    """Relative audio paths (e.g. ``56/1376256.mp3``) for the given filtered subset."""
    return df["PATH"].tolist()
