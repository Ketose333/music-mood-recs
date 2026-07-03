"""Real-world music recommendation: LLM candidates verified by iTunes Search.

The dataset-internal Top-5 can only recommend Jamendo tracks nobody knows.
This module recommends *released* music instead: the LLM proposes songs that
fit the analyzed mood, and each candidate is verified against the iTunes
Search API (free, no key) before it is shown — hallucinated songs simply fail
verification and are dropped. When no LLM is available, a mood-keyword iTunes
search fills in, so the feature degrades gracefully like the mood analyzer.

No audio is ever played from these services: each result links to the
service's own page/search (Spotify, YouTube Music, Apple Music), avoiding any
copyright exposure — the Chosic-style UX.
"""

from __future__ import annotations

import json
import random
import re
import urllib.parse
from dataclasses import dataclass, field

import requests

from src.llm.mood_analyzer import llm_chat

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_TIMEOUT = 10

# LLM-less fallback search terms per trained mood tag. Keyed by tag name so a
# bigger tag set (different dataset size/config) just needs new entries; tags
# without an entry fall back to the tag name itself as the search term. Each
# mood has several phrasing variants — one is picked at random per call so
# repeated searches for the same mood don't always hit iTunes with the exact
# same query (which would otherwise return the exact same Top-5 every time,
# since the API itself is deterministic for identical params).
MOOD_SEARCH_TERMS: dict[str, list[str]] = {
    "happy": ["happy upbeat feel good pop", "sunny cheerful pop hits", "feel good summer pop"],
    "energetic": ["energetic workout power up", "high energy dance pop", "pump up gym anthem"],
    "relaxing": ["relaxing calm acoustic chill", "chill lofi calm", "soft acoustic mellow"],
    "film": ["epic cinematic soundtrack score", "orchestral movie score", "dramatic film score"],
    "dark": ["dark moody melancholic", "brooding dark alternative", "melancholic minor key"],
}


@dataclass
class RealTrack:
    """One verified, released track with per-service outbound links."""

    title: str
    artist: str
    album: str = ""
    artwork_url: str = ""
    genre: str = ""
    links: dict[str, str] = field(default_factory=dict)
    reason: str = ""


# Per-country filter config: iTunes primaryGenreName values that mark a track
# as belonging to that country, plus a script check on title/artist as a
# second signal (some K-pop/J-pop tracks are tagged under a generic
# "Pop"/"Hip-Hop/Rap" genre instead of the country-specific one). Adding a
# country only needs a new entry here — everything else keys off this dict.
_COUNTRY_FILTERS: dict[str, dict] = {
    "KR": {
        "prompt_label": "한국 국내 가수(K-pop/국내 발매곡)",
        "genres": {"K-Pop", "Korean Pop", "Trot", "Korean Hip-Hop", "Korean R&B/Soul"},
        "script_re": re.compile(r"[가-힣]"),
        "search_suffix": "케이팝",
    },
    "JP": {
        "prompt_label": "일본 가수(J-pop/국내 발매곡)",
        "genres": {"J-Pop", "J-Rock", "Anime", "Japanese Pop"},
        "script_re": re.compile(r"[぀-ヿ]"),  # Hiragana + Katakana
        "search_suffix": "제이팝",
    },
}


def matches_country(track: "RealTrack", country: str) -> bool:
    """Heuristic: country-specific genre tag, or that country's script in the
    title/artist. ``country`` is a key of ``_COUNTRY_FILTERS`` (e.g. "KR")."""
    cfg = _COUNTRY_FILTERS.get(country)
    if cfg is None:
        return True
    if track.genre in cfg["genres"]:
        return True
    return bool(cfg["script_re"].search(track.title) or cfg["script_re"].search(track.artist))


def service_links(title: str, artist: str, itunes_url: str = "") -> dict[str, str]:
    """Search-page links per streaming service (no playback, no API keys)."""
    q = urllib.parse.quote(f"{title} {artist}")
    links = {
        "Spotify": f"https://open.spotify.com/search/{q}",
        "YouTube Music": f"https://music.youtube.com/search?q={q}",
        "Apple Music": itunes_url or f"https://music.apple.com/kr/search?term={q}",
    }
    return links


def _to_real_track(item: dict) -> RealTrack:
    title = item.get("trackName", "")
    artist = item.get("artistName", "")
    return RealTrack(
        title=title,
        artist=artist,
        album=item.get("collectionName", ""),
        artwork_url=item.get("artworkUrl100", ""),
        genre=item.get("primaryGenreName", ""),
        links=service_links(title, artist, item.get("trackViewUrl", "")),
    )


def itunes_search(term: str, limit: int = 5, country: str = "KR") -> list[RealTrack]:
    """Search released tracks on the iTunes Search API (free, keyless)."""
    resp = requests.get(
        ITUNES_SEARCH_URL,
        params={"term": term, "media": "music", "entity": "song", "limit": limit, "country": country},
        timeout=ITUNES_TIMEOUT,
    )
    resp.raise_for_status()
    return [_to_real_track(r) for r in resp.json().get("results", []) if r.get("trackName")]


def verify_track(title: str, artist: str, country: str = "KR") -> RealTrack | None:
    """Return the track's real metadata if it exists on iTunes, else None.

    This is the anti-hallucination gate: an LLM-invented song won't match any
    catalog entry and gets dropped instead of shown to the user.
    """
    try:
        results = itunes_search(f"{title} {artist}", limit=1, country=country)
    except requests.RequestException:
        return None
    return results[0] if results else None


def build_song_prompt(
    mood: str, user_text: str, k: int, artist_country: str | None = None,
    exclude: list[tuple[str, str]] | None = None,
) -> str:
    context = f"\n사용자의 원래 문장(분위기 참고): {user_text}" if user_text else ""
    country_rule = ""
    if artist_country and artist_country in _COUNTRY_FILTERS:
        label = _COUNTRY_FILTERS[artist_country]["prompt_label"]
        country_rule = f"\n- 반드시 {label}만 추천하세요. 다른 국가 아티스트는 제외."
    # LLM은 같은 프롬프트에 매번 가장 유명한 곡부터 답하는 경향이 있어, "다른 곡"
    # 재시도가 그냥 같은 Top-5를 반복하는 문제가 있었다. 이전에 이미 보여준 곡을
    # 명시적으로 제외 목록에 넣어야 실제로 다른 곡이 나온다.
    exclude_rule = ""
    if exclude:
        exclude_list = ", ".join(f"{t} - {a}" for t, a in exclude)
        exclude_rule = f"\n- 다음 곡은 이미 추천했으니 절대 다시 추천하지 마세요: {exclude_list}"
    return (
        f"'{mood}' 무드에 어울리는, 실제로 발매된 유명한 곡을 {k + 3}개 추천하세요."
        f"{context}\n\n"
        "규칙:\n"
        "- 실존하는 곡만. 확실하지 않은 곡은 넣지 마세요."
        f"{country_rule}{exclude_rule}\n"
        "- 반드시 JSON 객체 하나만 출력: "
        '{"songs": [{"title": "<곡명>", "artist": "<아티스트>", '
        '"reason": "<이 곡이 왜 이 무드와 어울리는지 한국어 1문장. 가사·줄거리 등 사실 주장 금지, 분위기/장르 톤만 언급>"}, ...]}\n'
        "- 다양한 아티스트로 구성하세요 (한 아티스트 최대 1곡)."
    )


def parse_song_response(raw: str) -> list[tuple[str, str, str]]:
    """Extract (title, artist, reason) triples from the LLM reply; [] when unparseable."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    triples: list[tuple[str, str, str]] = []
    for song in data.get("songs", []) if isinstance(data, dict) else []:
        if not isinstance(song, dict):
            continue
        title = str(song.get("title", "")).strip()
        artist = str(song.get("artist", "")).strip()
        reason = str(song.get("reason", "")).strip()
        if title and artist:
            triples.append((title, artist, reason))
    return triples


def recommend_real_tracks(
    mood: str,
    user_text: str = "",
    search_keywords: list[str] | None = None,
    k: int = 5,
    groq_api_key: str | None = None,
    country: str = "KR",
    artist_country: str | None = None,
    exclude: list[tuple[str, str]] | None = None,
) -> tuple[list[RealTrack], str]:
    """Top-k released tracks for ``mood``; returns (tracks, provider).

    provider: "ollama"/"groq" when LLM candidates (iTunes-verified) were used,
    "itunes" when only the keyword search produced results. LLM candidates are
    verified one by one and topped up from keyword search if too few survive.
    ``artist_country`` (a key of ``_COUNTRY_FILTERS``, e.g. "KR"/"JP") drops
    candidates that don't match from both sources (see ``matches_country``)
    rather than relaxing the filter to hit ``k`` — fewer, correctly-filtered
    results beat a wrong-country song slipping in. ``country`` is unrelated:
    it's just the iTunes storefront region used to verify/search tracks.
    ``exclude``: (title, artist) pairs already shown to the user (e.g. from a
    previous "다른 곡" reroll) — the LLM is told to avoid them and they are
    filtered out of both sources, so rerolling actually returns new songs
    instead of the same well-known picks the LLM tends to default to.
    """
    tracks: list[RealTrack] = []
    provider = "itunes"
    seen: set[tuple[str, str]] = {(t.lower(), a.lower()) for t, a in (exclude or [])}

    chat = llm_chat(
        build_song_prompt(mood, user_text, k, artist_country=artist_country, exclude=exclude),
        groq_api_key=groq_api_key,
    )
    if chat is not None:
        raw, llm_provider = chat
        for title, artist, reason in parse_song_response(raw):
            if len(tracks) >= k:
                break
            found = verify_track(title, artist, country=country)
            if found is None:
                continue
            if artist_country and not matches_country(found, artist_country):
                continue
            key = (found.title.lower(), found.artist.lower())
            if key in seen:
                continue
            found.reason = reason or f"'{mood}' 무드와 어울리는 곡으로 LLM이 추천했습니다."
            seen.add(key)
            tracks.append(found)
        if tracks:
            provider = llm_provider

    if len(tracks) < k:
        if search_keywords:
            term = " ".join(search_keywords)
        else:
            term = random.choice(MOOD_SEARCH_TERMS.get(mood, [f"{mood} music"]))
        if artist_country and artist_country in _COUNTRY_FILTERS:
            term = f"{term} {_COUNTRY_FILTERS[artist_country]['search_suffix']}"
        try:
            candidates = itunes_search(term, limit=max(k * 3, 25), country=country)
        except requests.RequestException:
            candidates = []
        if artist_country:
            candidates = [c for c in candidates if matches_country(c, artist_country)]
        # iTunes returns the same ordering for the same query every time, so
        # without shuffling the fallback would recommend the exact same songs
        # on every call for a given mood. Shuffle first, then fill with unique
        # artists first, then allow repeats if still short (keyword search
        # often returns one artist's whole album back-to-back).
        random.shuffle(candidates)
        for allow_repeat_artist in (False, True):
            for found in candidates:
                if len(tracks) >= k:
                    break
                key = (found.title.lower(), found.artist.lower())
                if key in seen:
                    continue
                if not allow_repeat_artist and any(
                    t.artist.lower() == found.artist.lower() for t in tracks
                ):
                    continue
                found.reason = f"'{mood}' 무드 검색으로 찾은 {found.genre or '추천'} 곡입니다."
                seen.add(key)
                tracks.append(found)

    return tracks[:k], provider
