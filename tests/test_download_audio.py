"""Tests for src.data.download_audio — pure logic only (no network)."""

from __future__ import annotations

import pandas as pd

from src.data.download_audio import restrict_subset_to_folders, subset_path_set


def _toy_subset() -> dict:
    df = pd.DataFrame({"PATH": ["00/1.mp3", "01/2.mp3", "00/3.mp3"]})
    return {"tags": ["happy"], "train": df, "validation": df.iloc[:1], "test": df.iloc[:0]}


def test_subset_path_set_collects_all_splits():
    subset = _toy_subset()
    paths = subset_path_set(subset)
    assert paths == {"00/1.mp3", "01/2.mp3", "00/3.mp3"}


def test_restrict_subset_to_folders_filters_by_prefix():
    subset = _toy_subset()
    restricted = restrict_subset_to_folders(subset, max_tars=1)
    assert restricted["train"]["PATH"].tolist() == ["00/1.mp3", "00/3.mp3"]
    assert restricted["tags"] == ["happy"]
