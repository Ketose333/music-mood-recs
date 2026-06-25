"""Tests for src.data.jamendo — subset selection logic with synthetic TSVs.

The download paths are exercised via dependency injection: we call the pure
parsing/filtering helpers on hand-built DataFrames so no network access is
needed.
"""

from __future__ import annotations

import pandas as pd

from src.data.jamendo import (
    _COLUMNS,
    _mood_tags,
    _parse_tags,
    filter_to_selected_tags,
    select_top_mood_tags,
)


def _row(track_id: str, tags: str, duration: str = "200.0") -> dict:
    return {
        "TRACK_ID": track_id,
        "ARTIST_ID": "a1",
        "ALBUM_ID": "al1",
        "PATH": f"{track_id[-2:]}/{track_id}.mp3",
        "DURATION": duration,
        "TAGS": tags,
    }


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df["tags_list"] = df["TAGS"].fillna("").apply(_parse_tags)
    df["mood_tags"] = df["tags_list"].apply(_mood_tags)
    return df


def test_parse_tags_handles_tabs_and_empty():
    assert _parse_tags("mood/theme---calm\tmood/theme---dark") == [
        "mood/theme---calm",
        "mood/theme---dark",
    ]
    assert _parse_tags("") == []


def test_mood_tags_strips_prefix_and_filters_non_mood():
    tags = ["mood/theme---calm", "genre---rock", "mood/theme---dark"]
    assert _mood_tags(tags) == ["calm", "dark"]


def test_select_top_mood_tags_returns_frequency_sorted_topn():
    df = _df(
        [
            _row("t1", "mood/theme---calm"),
            _row("t2", "mood/theme---calm\tmood/theme---dark"),
            _row("t3", "mood/theme---calm\tmood/theme---dark"),
            _row("t4", "mood/theme---dark"),
            _row("t5", "mood/theme---happy"),
        ]
    )
    top = select_top_mood_tags(df, top_n=2)
    # calm=3, dark=3, happy=1 → calm and dark tie; alphabetical tiebreaker
    assert top == ["calm", "dark"]


def test_select_top_mood_tags_respects_min_tracks():
    df = _df(
        [
            _row("t1", "mood/theme---calm"),
            _row("t2", "mood/theme---calm"),
            _row("t3", "mood/theme---rare"),
        ]
    )
    top = select_top_mood_tags(df, top_n=5, min_tracks=2)
    assert top == ["calm"]


def test_filter_to_selected_tags_keeps_only_matching_tracks_and_adds_onehot():
    df = _df(
        [
            _row("t1", "mood/theme---calm"),
            _row("t2", "mood/theme---dark"),
            _row("t3", "mood/theme---happy"),
        ]
    )
    out = filter_to_selected_tags(df, ["calm", "dark"])
    assert list(out["TRACK_ID"]) == ["t1", "t2"]
    assert list(out["tag_calm"]) == [1, 0]
    assert list(out["tag_dark"]) == [0, 1]
    assert list(out["n_selected_tags"]) == [1, 1]


def test_filter_to_selected_tags_keeps_track_with_multiple_selected_tags():
    df = _df([_row("t1", "mood/theme---calm\tmood/theme---dark")])
    out = filter_to_selected_tags(df, ["calm", "dark"])
    assert out.loc[0, "tag_calm"] == 1
    assert out.loc[0, "tag_dark"] == 1
    assert out.loc[0, "n_selected_tags"] == 2
