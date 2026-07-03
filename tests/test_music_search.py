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
        raw = '{"songs": [{"title": "A", "artist": "B"}, {"title": "C", "artist": "D", "reason": "chill"}]}'
        assert ms.parse_song_response(raw) == [("A", "B", ""), ("C", "D", "chill")]

    def test_skips_incomplete_entries(self):
        raw = '{"songs": [{"title": "A"}, {"artist": "B"}, {"title": "C", "artist": "D"}]}'
        assert ms.parse_song_response(raw) == [("C", "D", "")]

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

    def test_reason_from_llm_is_attached(self, monkeypatch):
        monkeypatch.setattr(
            ms, "llm_chat",
            lambda prompt, groq_api_key=None: (
                '{"songs": [{"title": "Real", "artist": "One", "reason": "잔잔한 분위기"}]}', "ollama",
            ),
        )
        monkeypatch.setattr(ms, "verify_track", lambda title, artist, country="KR": _track(title, artist))
        monkeypatch.setattr(ms, "itunes_search", lambda term, limit=5, country="KR": [])
        tracks, _ = ms.recommend_real_tracks("relaxing", k=5)
        assert tracks[0].reason == "잔잔한 분위기"

    def test_fallback_track_gets_generic_reason(self, monkeypatch):
        monkeypatch.setattr(ms, "llm_chat", lambda prompt, groq_api_key=None: None)
        monkeypatch.setattr(ms, "itunes_search", lambda term, limit=5, country="KR": [_track("S1", "A1")])
        tracks, _ = ms.recommend_real_tracks("happy", k=5)
        assert tracks[0].reason


class TestExcludeReroll:
    def test_excluded_llm_candidate_is_dropped(self, monkeypatch):
        monkeypatch.setattr(
            ms, "llm_chat",
            lambda prompt, groq_api_key=None: (
                '{"songs": [{"title": "Real", "artist": "One"}, {"title": "New", "artist": "Two"}]}', "ollama",
            ),
        )
        monkeypatch.setattr(ms, "verify_track", lambda title, artist, country="KR": _track(title, artist))
        monkeypatch.setattr(ms, "itunes_search", lambda term, limit=5, country="KR": [])
        tracks, _ = ms.recommend_real_tracks("happy", k=5, exclude=[("Real", "One")])
        assert [t.title for t in tracks] == ["New"]

    def test_excluded_fallback_candidate_is_dropped(self, monkeypatch):
        monkeypatch.setattr(ms, "llm_chat", lambda prompt, groq_api_key=None: None)
        monkeypatch.setattr(
            ms, "itunes_search",
            lambda term, limit=5, country="KR": [_track("Seen", "A"), _track("Unseen", "B")],
        )
        tracks, _ = ms.recommend_real_tracks("happy", k=5, exclude=[("Seen", "A")])
        assert [t.title for t in tracks] == ["Unseen"]

    def test_prompt_lists_excluded_songs(self):
        prompt = ms.build_song_prompt("happy", "", 5, exclude=[("Real", "One")])
        assert "Real - One" in prompt

    def test_prompt_omits_exclude_rule_when_empty(self):
        prompt = ms.build_song_prompt("happy", "", 5)
        assert "이미 추천했으니" not in prompt


class TestCountryFilter:
    def test_matches_country_by_genre(self):
        t = ms.RealTrack(title="Song", artist="Artist", genre="K-Pop")
        assert ms.matches_country(t, "KR")

    def test_matches_country_by_hangul(self):
        t = ms.RealTrack(title="좋은 날", artist="IU")
        assert ms.matches_country(t, "KR")

    def test_matches_country_false_for_foreign(self):
        t = ms.RealTrack(title="Blinding Lights", artist="The Weeknd", genre="Pop")
        assert not ms.matches_country(t, "KR")

    def test_matches_country_japanese_by_genre(self):
        t = ms.RealTrack(title="Song", artist="Artist", genre="J-Pop")
        assert ms.matches_country(t, "JP")

    def test_matches_country_japanese_by_script(self):
        t = ms.RealTrack(title="さくら", artist="いきものがかり")
        assert ms.matches_country(t, "JP")

    def test_matches_country_korean_track_not_japanese(self):
        t = ms.RealTrack(title="좋은 날", artist="IU", genre="K-Pop")
        assert not ms.matches_country(t, "JP")

    def test_prompt_includes_country_rule_when_enabled(self):
        prompt = ms.build_song_prompt("happy", "", 5, artist_country="KR")
        assert "한국 국내 가수" in prompt

    def test_prompt_includes_japan_rule(self):
        prompt = ms.build_song_prompt("happy", "", 5, artist_country="JP")
        assert "일본 가수" in prompt

    def test_prompt_omits_country_rule_by_default(self):
        prompt = ms.build_song_prompt("happy", "", 5)
        assert "한국 국내 가수" not in prompt and "일본 가수" not in prompt

    def test_recommend_filters_out_non_domestic_llm_candidates(self, monkeypatch):
        monkeypatch.setattr(
            ms, "llm_chat",
            lambda prompt, groq_api_key=None: (
                '{"songs": [{"title": "Blinding Lights", "artist": "The Weeknd"}, '
                '{"title": "좋은 날", "artist": "IU"}]}', "ollama",
            ),
        )

        def fake_verify(title, artist, country="KR"):
            genre = "Pop" if artist == "The Weeknd" else "K-Pop"
            return ms.RealTrack(title=title, artist=artist, genre=genre)

        monkeypatch.setattr(ms, "verify_track", fake_verify)
        monkeypatch.setattr(ms, "itunes_search", lambda term, limit=5, country="KR": [])
        tracks, _ = ms.recommend_real_tracks("happy", k=5, artist_country="KR")
        assert [t.artist for t in tracks] == ["IU"]

    def test_recommend_filters_fallback_candidates_and_biases_term(self, monkeypatch):
        monkeypatch.setattr(ms, "llm_chat", lambda prompt, groq_api_key=None: None)
        captured = {}

        def fake_search(term, limit=5, country="KR"):
            captured["term"] = term
            return [
                ms.RealTrack(title="Foreign", artist="Bar", genre="Pop"),
                ms.RealTrack(title="좋은 날", artist="IU", genre="K-Pop"),
            ]

        monkeypatch.setattr(ms, "itunes_search", fake_search)
        tracks, _ = ms.recommend_real_tracks("happy", k=5, artist_country="KR")
        assert "케이팝" in captured["term"]
        assert [t.artist for t in tracks] == ["IU"]

    def test_recommend_biases_term_for_japan(self, monkeypatch):
        monkeypatch.setattr(ms, "llm_chat", lambda prompt, groq_api_key=None: None)
        captured = {}

        def fake_search(term, limit=5, country="KR"):
            captured["term"] = term
            return [ms.RealTrack(title="さくら", artist="いきものがかり", genre="J-Pop")]

        monkeypatch.setattr(ms, "itunes_search", fake_search)
        tracks, _ = ms.recommend_real_tracks("happy", k=5, artist_country="JP")
        assert "제이팝" in captured["term"]
        assert [t.artist for t in tracks] == ["いきものがかり"]
