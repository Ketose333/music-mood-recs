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
import re
import urllib.parse
from dataclasses import dataclass, field

import requests

from src.llm.mood_analyzer import llm_chat

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_TIMEOUT = 10

# LLM-less fallback search terms per trained mood tag. Keyed by tag name so a
# bigger tag set (different dataset size/config) just needs new entries; tags
# without an entry fall back to the tag name itself as the search term.
MOOD_SEARCH_TERMS: dict[str, str] = {
    "happy": "happy upbeat feel good pop",
    "energetic": "energetic workout power up",
    "relaxing": "relaxing calm acoustic chill",
    "film": "epic cinematic soundtrack score",
    "dark": "dark moody melancholic",
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


def build_song_prompt(mood: str, user_text: str, k: int) -> str:
    context = f"\n사용자의 원래 문장(분위기 참고): {user_text}" if user_text else ""
    return (
        f"'{mood}' 무드에 어울리는, 실제로 발매된 유명한 곡을 {k + 3}개 추천하세요."
        f"{context}\n\n"
        "규칙:\n"
        "- 실존하는 곡만. 확실하지 않은 곡은 넣지 마세요.\n"
        "- 반드시 JSON 객체 하나만 출력: "
        '{"songs": [{"title": "<곡명>", "artist": "<아티스트>"}, ...]}\n'
        "- 다양한 아티스트로 구성하세요 (한 아티스트 최대 1곡)."
    )


def parse_song_response(raw: str) -> list[tuple[str, str]]:
    """Extract (title, artist) pairs from the LLM reply; [] when unparseable."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    pairs: list[tuple[str, str]] = []
    for song in data.get("songs", []) if isinstance(data, dict) else []:
        if not isinstance(song, dict):
            continue
        title = str(song.get("title", "")).strip()
        artist = str(song.get("artist", "")).strip()
        if title and artist:
            pairs.append((title, artist))
    return pairs


def recommend_real_tracks(
    mood: str,
    user_text: str = "",
    search_keywords: list[str] | None = None,
    k: int = 5,
    groq_api_key: str | None = None,
    country: str = "KR",
) -> tuple[list[RealTrack], str]:
    """Top-k released tracks for ``mood``; returns (tracks, provider).

    provider: "ollama"/"groq" when LLM candidates (iTunes-verified) were used,
    "itunes" when only the keyword search produced results. LLM candidates are
    verified one by one and topped up from keyword search if too few survive.
    """
    tracks: list[RealTrack] = []
    provider = "itunes"
    seen: set[tuple[str, str]] = set()

    chat = llm_chat(build_song_prompt(mood, user_text, k), groq_api_key=groq_api_key)
    if chat is not None:
        raw, llm_provider = chat
        for title, artist in parse_song_response(raw):
            if len(tracks) >= k:
                break
            found = verify_track(title, artist, country=country)
            if found is None:
                continue
            key = (found.title.lower(), found.artist.lower())
            if key in seen:
                continue
            seen.add(key)
            tracks.append(found)
        if tracks:
            provider = llm_provider

    if len(tracks) < k:
        term = " ".join(search_keywords) if search_keywords else MOOD_SEARCH_TERMS.get(mood, f"{mood} music")
        try:
            candidates = itunes_search(term, limit=max(k * 3, 25), country=country)
        except requests.RequestException:
            candidates = []
        # Keyword search often returns one artist's whole album back-to-back;
        # fill with unique artists first, then allow repeats if still short.
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
                seen.add(key)
                tracks.append(found)

    return tracks[:k], provider
