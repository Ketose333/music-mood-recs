"""Tests for src/llm/music_search.py — LLM candidate parsing, iTunes
verification (anti-hallucination gate), link building, and the keyword-search
fallback. All HTTP is monkeypatched; no network access."""

import pytest
import requests

from src.llm import music_search as ms


def _track(title="Song A", artist="Artist A"):
    return ms.RealTrack(title=title, artist=artist, links=ms.service_links(title, artist))


class TestServiceLinks:
    def test_all_three_services_present(self):
        links = ms.service_links("Dynamite", "BTS")
        assert set(links) == {"Spotify", "YouTube Music", "Apple Music"}
        assert "open.spotify.com/search/Dynamite%20BTS" in links["Spotify"]
        assert "music.youtube.com/search?q=Dynamite%20BTS" in links["YouTube Music"]

    def test_itunes_url_preferred_for_apple_music(self):
        links = ms.service_links("t", "a", itunes_url="https://music.apple.com/kr/album/x")
        assert links["Apple Music"] == "https://music.apple.com/kr/album/x"


class TestParseSongResponse:
    def test_valid_songs(self):
        raw = '{"songs": [{"title": "A", "artist": "B"}, {"title": "C", "artist": "D"}]}'
        assert ms.parse_song_response(raw) == [("A", "B"), ("C", "D")]

    def test_skips_incomplete_entries(self):
        raw = '{"songs": [{"title": "A"}, {"artist": "B"}, {"title": "C", "artist": "D"}]}'
        assert ms.parse_song_response(raw) == [("C", "D")]

    def test_unparseable_returns_empty(self):
        assert ms.parse_song_response("no json here") == []
        assert ms.parse_song_response('{"songs": "oops"}') == []


class TestRecommendRealTracks:
    def test_llm_candidates_verified_and_used(self, monkeypatch):
        monkeypatch.setattr(
            ms, "llm_chat",
            lambda prompt, groq_api_key=None: ('{"songs": [{"title": "Real", "artist": "One"}, {"title": "Fake", "artist": "Two"}]}', "ollama"),
        )
        # "Fake" fails verification (hallucination gate), "Real" passes.
        monkeypatch.setattr(
            ms, "verify_track",
            lambda title, artist, country="KR": _track(title, artist) if title == "Real" else None,
        )
        monkeypatch.setattr(ms, "itunes_search", lambda term, limit=5, country="KR": [])
        tracks, provider = ms.recommend_real_tracks("happy", k=5)
        assert provider == "ollama"
        assert [t.title for t in tracks] == ["Real"]

    def test_falls_back_to_itunes_keyword_search_without_llm(self, monkeypatch):
        monkeypatch.setattr(ms, "llm_chat", lambda prompt, groq_api_key=None: None)
        monkeypatch.setattr(
            ms, "itunes_search",
            lambda term, limit=5, country="KR": [_track(f"S{i}", f"A{i}") for i in range(7)],
        )
        tracks, provider = ms.recommend_real_tracks("relaxing", k=5)
        assert provider == "itunes"
        assert len(tracks) == 5

    def test_deduplicates_topup_results(self, monkeypatch):
        monkeypatch.setattr(
            ms, "llm_chat",
            lambda prompt, groq_api_key=None: ('{"songs": [{"title": "Same", "artist": "One"}]}', "groq"),
        )
        monkeypatch.setattr(ms, "verify_track", lambda title, artist, country="KR": _track(title, artist))
        monkeypatch.setattr(
            ms, "itunes_search",
            lambda term, limit=5, country="KR": [_track("Same", "One"), _track("Other", "Two")],
        )
        tracks, provider = ms.recommend_real_tracks("dark", k=5)
        titles = [t.title for t in tracks]
        assert titles.count("Same") == 1
        assert "Other" in titles
        assert provider == "groq"

    def test_survives_itunes_network_failure(self, monkeypatch):
        monkeypatch.setattr(ms, "llm_chat", lambda prompt, groq_api_key=None: None)

        def raise_conn(term, limit=5, country="KR"):
            raise requests.ConnectionError("offline")

        monkeypatch.setattr(ms, "itunes_search", raise_conn)
        tracks, provider = ms.recommend_real_tracks("film", k=5)
        assert tracks == []
        assert provider == "itunes"

    def test_unknown_mood_uses_generic_term(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(ms, "llm_chat", lambda prompt, groq_api_key=None: None)

        def capture(term, limit=5, country="KR"):
            captured["term"] = term
            return []

        monkeypatch.setattr(ms, "itunes_search", capture)
        ms.recommend_real_tracks("jazzy", k=5)
        assert captured["term"] == "jazzy music"

    def test_search_keywords_override_default_term(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(ms, "llm_chat", lambda prompt, groq_api_key=None: None)

        def capture(term, limit=5, country="KR"):
            captured["term"] = term
            return []

        monkeypatch.setattr(ms, "itunes_search", capture)
        ms.recommend_real_tracks("happy", search_keywords=["upbeat pop", "summer"], k=5)
        assert captured["term"] == "upbeat pop summer"
