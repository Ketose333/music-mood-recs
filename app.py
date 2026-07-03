"""music-mood-recs Streamlit app — track select -> mood prediction -> top-5 similar.

Loads the trained MoodCNN, precomputed mel-spectrograms, and embeddings.
Reuses review-sentiment's st.cache_resource pattern for model loading.
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

# >>> AUTO-SYNCED from src/models/cnn.py (run scripts/sync_standalone_app.py) >>>
@dataclass(frozen=True)
class CNNConfig:
    n_mels: int = 128
    n_classes: int = 5
    embedding_dim: int = 64
    conv_channels: tuple[int, int, int] = (16, 32, 64)
    kernel_size: int = 3
    dropout: float = 0.3


class MoodCNN(nn.Module):
    def __init__(self, cfg: CNNConfig | None = None):
        super().__init__()
        self.cfg = cfg or CNNConfig()
        c1, c2, c3 = self.cfg.conv_channels

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, self.cfg.kernel_size, padding=self.cfg.kernel_size // 2),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c1, c2, self.cfg.kernel_size, padding=self.cfg.kernel_size // 2),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c2, c3, self.cfg.kernel_size, padding=self.cfg.kernel_size // 2),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, self.cfg.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
        )
        self.classifier = nn.Linear(self.cfg.embedding_dim, self.cfg.n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the embedding vector (batch, embedding_dim) used for recommendation."""
        h = self.features(x)
        h = self.pool(h)
        return self.embed_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        return self.classifier(z)
# <<< AUTO-SYNCED <<<


# >>> AUTO-SYNCED from src/preprocessing/melspec.py (run scripts/sync_standalone_app.py) >>>
@dataclass(frozen=True)
class MelspecConfig:
    sr: int = 22050
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    segment_seconds: float = 30.0
    offset_mode: str = "start"  # "start" or "center"
    top_db: float = 80.0

    @property
    def segment_samples(self) -> int:
        return int(self.sr * self.segment_seconds)

    @property
    def expected_frames(self) -> int:
        # librosa default center=True pads n_fft//2 on each side, so
        # n_frames = 1 + n_samples // hop_length.
        return 1 + self.segment_samples // self.hop_length


def load_segment(
    audio_path: str, cfg: MelspecConfig, duration_cap: Optional[float] = None
) -> np.ndarray:
    """Load a fixed 30-second mono segment as a 1D float32 numpy array."""
    total_duration = duration_cap
    if total_duration is None:
        try:
            total_duration = float(librosa.get_duration(path=audio_path))
        except Exception:
            total_duration = cfg.segment_seconds
    if cfg.offset_mode == "center" and total_duration > cfg.segment_seconds:
        offset = (total_duration - cfg.segment_seconds) / 2.0
    else:
        offset = 0.0
    y, _ = librosa.load(
        audio_path,
        sr=cfg.sr,
        mono=True,
        offset=offset,
        duration=cfg.segment_seconds,
    )
    target = cfg.segment_samples
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    elif len(y) > target:
        y = y[:target]
    return y.astype(np.float32)


def compute_melspec(y: np.ndarray, cfg: MelspecConfig) -> np.ndarray:
    """Compute a log-mel spectrogram (n_mels, n_frames) from a 1D waveform."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=cfg.top_db)
    return log_mel.astype(np.float32)


def extract_melspec(
    audio_path: str, cfg: Optional[MelspecConfig] = None
) -> np.ndarray:
    """Load segment + compute log-mel spectrogram in one call."""
    cfg = cfg or MelspecConfig()
    y = load_segment(audio_path, cfg)
    return compute_melspec(y, cfg)
# <<< AUTO-SYNCED <<<


# >>> AUTO-SYNCED from src/recommend/similar.py (run scripts/sync_standalone_app.py) >>>
@torch.no_grad()
def extract_embeddings(
    model: MoodCNN,
    mels: np.ndarray,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Compute embeddings for an array of mel-spectrograms.

    ``mels``: (N, n_mels, n_frames) float32. Returns (N, embedding_dim) float32.
    """
    model.eval()
    model.to(device)
    out: list[np.ndarray] = []
    for i in range(0, len(mels), batch_size):
        batch = mels[i : i + batch_size]
        x = torch.from_numpy(batch).unsqueeze(1).to(device)
        z = model.embed(x)
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


def top_k_similar(
    query_idx: int,
    embeddings: np.ndarray,
    k: int = 5,
    exclude_query: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, similarities) of the top-k most similar tracks.

    Cosine similarity is computed between the query embedding and all others.
    If ``exclude_query`` is True, the query itself is removed from the results.
    """
    query = embeddings[query_idx : query_idx + 1]
    sims = cosine_similarity(query, embeddings).ravel()
    order = np.argsort(-sims)
    if exclude_query:
        order = order[order != query_idx]
    top = order[:k]
    return top, sims[top]


def top_k_similar_to_vector(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Same ranking as ``top_k_similar`` but for a query embedding that is not
    already a row of ``embeddings`` (e.g. a user-uploaded track)."""
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(query, embeddings).ravel()
    order = np.argsort(-sims)
    top = order[:k]
    return top, sims[top]


@torch.no_grad()
def predict_mood_probs(model: MoodCNN, embeddings: np.ndarray) -> np.ndarray:
    """Run the trained classifier head on precomputed embeddings to get each
    track's mood probabilities without re-loading any mel-spectrogram.

    ``model.embed`` already produced ``embeddings``, so this only replays the
    final linear layer (``model.classifier``) — cheap enough to run over the
    whole library at app start-up.
    """
    model.eval()
    z = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    logits = model.classifier(z)
    return torch.sigmoid(logits).numpy()


MOOD_KEYWORDS: dict[str, list[str]] = {
    "happy": ["행복", "기쁘", "기뻐", "즐겁", "신나", "설레", "웃음", "웃기", "들뜨", "유쾌"],
    "energetic": ["에너지", "파워", "운동", "헬스", "달리", "파이팅", "격렬", "흥분", "활기", "신남"],
    "relaxing": ["편안", "휴식", "잠", "졸려", "힐링", "차분", "여유", "평화", "느긋", "안정"],
    "film": ["영화", "웅장", "드라마틱", "서사", "영상", "스토리", "장엄", "긴장감", "몰입"],
    "dark": ["우울", "슬픔", "슬프", "어둡", "외롭", "쓸쓸", "그리움", "눈물", "힘들", "지치", "불안"],
}


def infer_mood_from_text(text: str, tags: list[str]) -> tuple[str | None, dict[str, int]]:
    """Score ``text`` against each tag's keyword list and return the
    best-matching tag (``None`` if no keyword hits at all) plus the raw
    per-tag hit counts, so the caller can show why a tag was picked.
    """
    counts = {tag: 0 for tag in tags}
    for tag in tags:
        for kw in MOOD_KEYWORDS.get(tag, []):
            if kw in text:
                counts[tag] += 1
    best_tag = max(counts, key=counts.get) if counts else None
    if best_tag is not None and counts[best_tag] == 0:
        best_tag = None
    return best_tag, counts
# <<< AUTO-SYNCED <<<


# >>> AUTO-SYNCED from src/evaluation/metrics.py (run scripts/sync_standalone_app.py) >>>
def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """results = {model_display_name: {"accuracy":..., "f1_micro":..., "f1_macro":..., "roc_auc":...}}."""
    if not results:
        return pd.DataFrame(columns=["Accuracy", "F1(micro)", "F1(macro)", "ROC-AUC"])

    df = pd.DataFrame(results).T
    df = df.rename(
        columns={
            "accuracy": "Accuracy",
            "f1_micro": "F1(micro)",
            "f1_macro": "F1(macro)",
            "roc_auc": "ROC-AUC",
        }
    )
    return df[["Accuracy", "F1(micro)", "F1(macro)", "ROC-AUC"]]


def load_all_metrics(models_dir: str = "models") -> dict[str, dict]:
    """Scans models/*/metrics.json. Prefers the held-out "test" entry (final
    generalization check) when present; otherwise falls back to the last training
    epoch's val metrics."""
    results: dict[str, dict] = {}
    for metrics_path in sorted(glob.glob(os.path.join(models_dir, "*", "metrics.json"))):
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        model_dir_name = os.path.basename(os.path.dirname(metrics_path))
        display_name = data.get("display_name", model_dir_name)
        history = data.get("history")
        source = data.get("test") or (history[-1] if history else data)
        results[display_name] = {
            "accuracy": source.get("accuracy"),
            "f1_micro": source.get("f1_micro"),
            "f1_macro": source.get("f1_macro"),
            "roc_auc": source.get("roc_auc"),
        }
    return results
# <<< AUTO-SYNCED <<<


# >>> AUTO-SYNCED from src/llm/mood_analyzer.py (run scripts/sync_standalone_app.py) >>>
import json
import os
import re
from dataclasses import dataclass, field
import requests


OLLAMA_URL = os.environ.get("MMR_OLLAMA_URL", "http://localhost:11434")


OLLAMA_MODEL = os.environ.get("MMR_OLLAMA_MODEL", "gemma4:e2b")


OLLAMA_TIMEOUT = int(os.environ.get("MMR_OLLAMA_TIMEOUT", "90"))


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


GROQ_MODEL = os.environ.get("MMR_GROQ_MODEL", "llama-3.3-70b-versatile")


LLM_TIMEOUT = 30


@dataclass
class MoodAnalysis:
    """One text-input mood decision, with enough context to show *why*."""

    mood: str
    confidence: float
    reason: str
    search_keywords: list[str] = field(default_factory=list)
    provider: str = "keyword"  # "ollama" | "groq" | "keyword"


def build_mood_prompt(text: str, tags: list[str]) -> str:
    tag_list = ", ".join(tags)
    return (
        "당신은 음악 무드 분석가입니다. 사용자의 문장을 읽고 아래 5개 무드 태그 중 "
        "가장 어울리는 하나를 고르세요.\n"
        f"허용 태그(반드시 이 중 하나만): {tag_list}\n\n"
        "규칙:\n"
        "- 반드시 JSON 객체 하나만 출력하세요. 다른 텍스트/마크다운 금지.\n"
        '- 형식: {"mood": "<태그>", "confidence": 0.0~1.0, '
        '"reason": "<한국어 한두 문장>", "search_keywords": ["<영어 음악 검색 키워드 2~4개>"]}\n'
        "- search_keywords는 이 기분에 맞는 실제 음원을 찾기 위한 영어 검색어입니다 "
        '(예: "upbeat pop", "calm acoustic").\n\n'
        f"사용자 문장: {text}"
    )


def parse_mood_response(raw: str, tags: list[str]) -> dict | None:
    """Extract and validate the JSON object from an LLM reply.

    LLMs sometimes wrap JSON in markdown fences or prose, so grab the first
    ``{...}`` span. Returns None (caller falls through to the next tier) when
    the reply has no valid JSON or hallucinates a mood outside the tag set.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    mood = str(data.get("mood", "")).strip().lower()
    if mood not in tags:
        return None
    try:
        confidence = min(max(float(data.get("confidence", 0.5)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.5
    keywords = data.get("search_keywords") or []
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k).strip() for k in keywords if str(k).strip()][:4]
    return {
        "mood": mood,
        "confidence": confidence,
        "reason": str(data.get("reason", "")).strip(),
        "search_keywords": keywords,
    }


def chat_ollama(prompt: str, timeout: int = OLLAMA_TIMEOUT) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def chat_groq(prompt: str, api_key: str, timeout: int = LLM_TIMEOUT) -> str:
    resp = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def llm_chat(prompt: str, groq_api_key: str | None = None) -> tuple[str, str] | None:
    """Try each LLM tier in order; return (raw_reply, provider) or None.

    Network/HTTP errors just move on to the next tier — the caller always has
    the keyword heuristic as a final answer, so no tier failure is fatal.
    """
    try:
        return chat_ollama(prompt), "ollama"
    except (requests.RequestException, KeyError, ValueError):
        pass
    if groq_api_key:
        try:
            return chat_groq(prompt, groq_api_key), "groq"
        except (requests.RequestException, KeyError, ValueError):
            pass
    return None


def analyze_mood(text: str, tags: list[str], groq_api_key: str | None = None) -> MoodAnalysis:
    """Map a free-text feeling onto one of the trained mood ``tags``.

    Tier 1/2 (Ollama/Groq) return the LLM's own reasoning and music-search
    keywords; tier 3 reproduces the original keyword-heuristic behaviour so
    the feature keeps working with no LLM available at all.
    """
    prompt = build_mood_prompt(text, tags)
    chat = llm_chat(prompt, groq_api_key=groq_api_key)
    if chat is not None:
        raw, provider = chat
        parsed = parse_mood_response(raw, tags)
        if parsed is not None:
            return MoodAnalysis(provider=provider, **parsed)

    best_tag, counts = infer_mood_from_text(text, tags)
    if best_tag is None:
        best_tag = tags[0]
        reason = "키워드 매칭 없음 — 기본 태그로 대체 (LLM 미사용)"
    else:
        reason = f"한국어 감정 키워드 {counts[best_tag]}건 매칭 (LLM 미사용)"
    return MoodAnalysis(
        mood=best_tag,
        confidence=min(counts.get(best_tag, 0) / 3.0, 1.0),
        reason=reason,
        search_keywords=[],
        provider="keyword",
    )
# <<< AUTO-SYNCED <<<


# >>> AUTO-SYNCED from src/llm/music_search.py (run scripts/sync_standalone_app.py) >>>
import json
import random
import re
import urllib.parse
from dataclasses import dataclass, field
import requests


ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


ITUNES_TIMEOUT = 10


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
    # Apple's official 30s preview clip URL — consumed by
    # src/recommend/preview_rank.py for CNN-embedding re-ranking only,
    # never stored or played in the app.
    preview_url: str = ""


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
        preview_url=item.get("previewUrl", ""),
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
# <<< AUTO-SYNCED <<<


# >>> AUTO-SYNCED from src/recommend/preview_rank.py (run scripts/sync_standalone_app.py) >>>
import os
import subprocess
import tempfile
import urllib.parse
from typing import TypeVar
import numpy as np
import requests
import torch
from sklearn.metrics.pairwise import cosine_similarity


PREVIEW_TIMEOUT = 15


FFMPEG_TIMEOUT = 60


T = TypeVar("T")


def _decode_with_ffmpeg(src_path: str) -> str | None:
    """Transcode an audio file to a temp wav; returns the wav path or None.

    iTunes previews are AAC/m4a, which neither libsndfile nor audioread can
    decode without a system ffmpeg. imageio-ffmpeg ships a static ffmpeg
    binary through pip (no apt/brew needed — works on Streamlit Cloud and
    local Windows alike), so we shell out to it only when librosa's own
    loaders have already failed. The caller deletes the returned wav.
    """
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = get_ffmpeg_exe()
    except Exception:
        return None
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    try:
        subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", src_path, "-ac", "1", "-ar", "22050", wav_path],
            check=True,
            capture_output=True,
            timeout=FFMPEG_TIMEOUT,
        )
    except Exception:
        os.remove(wav_path)
        return None
    return wav_path


def melspec_from_preview(preview_url: str, n_mels: int = 128) -> np.ndarray | None:
    """Download one iTunes preview clip and return its log-mel spectrogram
    (the same array shape ``extract_melspec`` produces for a library track),
    or None on any failure (no URL, network error, undecodable audio).

    Split out from ``embed_preview`` so callers that also need the trained
    classifier's mood probabilities (not just the embedding) — e.g. the
    "search a released song" input mode — can run one download+decode and
    feed the resulting mel into ``model.embed``/``model.classifier`` both,
    exactly like ``app.py``'s uploaded-audio path already does.
    """
    if not preview_url:
        return None
    try:
        resp = requests.get(preview_url, timeout=PREVIEW_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    suffix = os.path.splitext(urllib.parse.urlparse(preview_url).path)[1] or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name
    try:
        try:
            return extract_melspec(tmp_path, MelspecConfig(n_mels=n_mels))
        except Exception:
            # librosa can't decode AAC/m4a directly — transcode via the
            # pip-bundled ffmpeg and retry once; still failing → give up.
            wav_path = _decode_with_ffmpeg(tmp_path)
            if wav_path is None:
                return None
            try:
                return extract_melspec(wav_path, MelspecConfig(n_mels=n_mels))
            except Exception:
                return None
            finally:
                os.remove(wav_path)
    finally:
        os.remove(tmp_path)


@torch.no_grad()
def embed_preview(preview_url: str, model: MoodCNN, n_mels: int = 128) -> np.ndarray | None:
    """Download one preview clip and return its CNN embedding, or None.

    The clip (~30s AAC/m4a) goes through the exact pipeline used for library
    tracks (``extract_melspec`` pads/trims to the fixed 30s segment), so the
    embedding lives in the same space as ``artifacts/embeddings.npy`` and is
    directly comparable by cosine similarity.
    """
    mel = melspec_from_preview(preview_url, n_mels=n_mels)
    if mel is None:
        return None
    model.eval()
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
    z = model.embed(x)
    return z[0].numpy()


def preview_similarities(
    query_embedding: np.ndarray, preview_embeddings: list[np.ndarray | None]
) -> list[float | None]:
    """Cosine similarity of each preview embedding to the query embedding,
    positionally aligned with the input; None entries stay None."""
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    sims: list[float | None] = []
    for emb in preview_embeddings:
        if emb is None:
            sims.append(None)
            continue
        vec = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        sims.append(float(cosine_similarity(query, vec)[0, 0]))
    return sims


def rerank_with_similarity(items: list[T], sims: list[float | None]) -> list[tuple[T, float | None]]:
    """Sort ``items`` by similarity (desc). Unscored items (sim None) keep
    their original relative order and follow the scored ones, so a missing
    preview never buries a track the LLM ranked high on its own."""
    scored = [(item, s) for item, s in zip(items, sims) if s is not None]
    unscored = [(item, s) for item, s in zip(items, sims) if s is None]
    scored.sort(key=lambda pair: -pair[1])
    return scored + unscored
# <<< AUTO-SYNCED <<<


MODEL_DIR = os.environ.get("MMR_MODEL_DIR", "models/cnn")
AUDIO_DIR = os.environ.get("MMR_AUDIO_DIR", "data/audio")
MELSPEC_DIR = os.environ.get("MMR_MELSPEC_DIR", "artifacts/melspecs")
MANIFEST_CSV = os.environ.get("MMR_MANIFEST", "artifacts/melspec_manifest.csv")
META_CSV = os.environ.get("MMR_META", "artifacts/subset_meta.csv")
EMBEDDINGS_NPY = os.environ.get("MMR_EMBEDDINGS", "artifacts/embeddings.npy")

# data/audio and artifacts/melspecs are no longer tracked in this git repo
# (they pushed GitHub's free LFS quota ~7x over) — they live in this public
# HF dataset repo instead, keyed by the same relative path used locally
# (e.g. "data/audio/00/12100.mp3", "artifacts/melspecs/00/12100.npy").
HF_ASSETS_REPO = os.environ.get("MMR_HF_ASSETS_REPO", "Ketose333/music-mood-recs-assets")


def _resolve(rel_path: str) -> str:
    """Returns a local path usable for rel_path, downloading it from
    HF_ASSETS_REPO on first access if it isn't already present on disk."""
    if os.path.exists(rel_path):
        return rel_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_ASSETS_REPO, repo_type="dataset", filename=rel_path.replace(os.sep, "/"))


@st.cache_resource(max_entries=1)
def load_model_artifacts(model_dir: str):
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg_dict = json.load(f)
    with open(os.path.join(model_dir, "tags.json"), encoding="utf-8") as f:
        tags = json.load(f)
    cfg = CNNConfig(
        n_mels=cfg_dict["n_mels"],
        n_classes=cfg_dict["n_classes"],
        embedding_dim=cfg_dict["embedding_dim"],
    )
    model = MoodCNN(cfg)
    model.load_state_dict(torch.load(_resolve(os.path.join(model_dir, "model.pt")), map_location="cpu"))
    model.eval()
    return model, tags, cfg


@st.cache_data
def load_manifest_and_meta(manifest_csv: str, meta_csv: str):
    manifest = pd.read_csv(_resolve(manifest_csv))
    meta = pd.read_csv(_resolve(meta_csv)).set_index("TRACK_ID")
    return manifest, meta


@st.cache_data
def load_embeddings(embeddings_npy: str, manifest: pd.DataFrame) -> np.ndarray:
    """Loads precomputed embeddings (scripts/precompute_embeddings.py), aligned
    1:1 with ``manifest`` row order.

    Streamlit Community Cloud only guarantees 1GB RAM: stacking every
    mel-spectrogram (~1.4GB for 2,247 tracks) to run a forward pass at startup
    would OOM-kill the app, so that forward pass happens offline instead and
    only this small embeddings array is loaded here.
    """
    embeddings = np.load(_resolve(embeddings_npy))
    if len(embeddings) != len(manifest):
        raise ValueError(
            f"embeddings ({len(embeddings)}) and manifest ({len(manifest)}) row counts differ — "
            "re-run scripts/precompute_embeddings.py"
        )
    return embeddings


@st.cache_data(max_entries=8)
def load_mel(npy_path: str) -> np.ndarray:
    """Lazily loads a single track's mel-spectrogram (only the selected track,
    never the full dataset — see load_embeddings for why)."""
    return np.load(_resolve(npy_path)).astype(np.float32)


def _track_display(track_id: str, meta: pd.DataFrame, tags: list[str]) -> str:
    if track_id not in meta.index:
        return track_id
    row = meta.loc[track_id]
    active = [t for t in tags if int(row.get(f"tag_{t}", 0)) == 1]
    return f"{track_id}  [{', '.join(active) if active else '-'}]"


def _audio_path(track_id: str, manifest: pd.DataFrame) -> str | None:
    rows = manifest.loc[manifest["TRACK_ID"] == track_id, "PATH"]
    if rows.empty:
        return None
    rel_path = os.path.join(AUDIO_DIR, rows.iloc[0])
    try:
        return _resolve(rel_path)
    except Exception:
        return None


_MOOD_EMOJI = {
    "happy": "😊",
    "energetic": "⚡",
    "relaxing": "🌿",
    "film": "🎬",
    "dark": "🌑",
}


st.set_page_config(page_title="music-mood-recs", page_icon="🎵", layout="wide")

with st.sidebar:
    st.title("🎵 음악 무드 분류 + 추천")

try:
    model, tags, cfg = load_model_artifacts(MODEL_DIR)
    manifest, meta = load_manifest_and_meta(MANIFEST_CSV, META_CSV)
    track_ids = manifest["TRACK_ID"].tolist()
    embeddings = load_embeddings(EMBEDDINGS_NPY, manifest)
except Exception as exc:
    st.error(f"모델/데이터를 불러오지 못했습니다: {exc}")
    st.info(
        "데모 실행 순서:\n"
        "1. `python scripts/download_audio.py --top-n 5 --max-tars 30`\n"
        "2. `python scripts/extract_melspecs.py`\n"
        "3. `python scripts/train_cnn.py`\n"
        "4. `python -m scripts.precompute_embeddings`\n"
        "5. `streamlit run app.py`\n\n"
        "또는 환경변수로 경로 지정: `MMR_MODEL_DIR`, `MMR_MANIFEST`, `MMR_META`, `MMR_EMBEDDINGS`"
    )
    st.stop()

with st.sidebar:
    st.divider()
    input_mode = st.selectbox(
        "입력 방식",
        ["음원 검색", "오디오 업로드", "텍스트로 찾기", "라이브러리 곡 선택"],
        help="무드를 예측할 방법을 선택하세요. '라이브러리 곡 선택'은 학습 데이터셋(DL 과제)에서 곡을 고르는 데모용입니다.",
    )
    rec_view = st.selectbox(
        "추천 결과 표시",
        ["전체", "추천만", "라이브러리 데모만"],
        help="추천은 외부 검색(LLM+iTunes)이 걸려 결과가 길어질 수 있습니다. 필요한 쪽만 골라 보세요.",
    )
    show_library = rec_view != "추천만"
    show_real = rec_view != "라이브러리 데모만"
    artist_country = None
    if show_real:
        country_choice = st.selectbox(
            "국가 선택",
            ["전체", "한국", "일본"],
            help="추천 Top-5를 특정 국가 가수(K-pop/J-pop 등 국내 발매곡)로 제한할지 선택하세요.",
        )
        artist_country = {"한국": "KR", "일본": "JP"}.get(country_choice)

st.success(f"모델 로드 완료 — {len(track_ids)}곡, 태그: {', '.join(tags)}")

def _render_recommendations(idxs: np.ndarray, sims: np.ndarray, score_label: str = "코사인 유사도") -> None:
    for i, sim in zip(idxs, sims):
        tid = track_ids[i]
        with st.container(border=True):
            rec_col_info, rec_col_audio = st.columns([2, 1])
            rec_col_info.markdown(f"**{_track_display(tid, meta, tags)}**")
            rec_col_info.caption(f"{score_label} {sim:.4f}")
            rec_audio = _audio_path(tid, manifest)
            if rec_audio:
                rec_col_audio.audio(rec_audio)


def _get_groq_api_key() -> str | None:
    """Groq free-tier key for the cloud LLM tier (Streamlit Cloud has no
    Ollama). Env var first, then Streamlit secrets; None disables the tier."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


GROQ_API_KEY = _get_groq_api_key()

_PROVIDER_LABEL = {
    "ollama": "🦙 Ollama 로컬 LLM",
    "groq": "☁️ Groq LLM API",
    "keyword": "🔤 키워드 휴리스틱 (LLM 미사용)",
    "itunes": "🔎 iTunes 무드 검색 (LLM 미사용)",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_real_tracks(
    mood: str, user_text: str, keywords: tuple[str, ...], _reroll: int = 0, artist_country: str | None = None,
    exclude: tuple[tuple[str, str], ...] = (),
):
    return recommend_real_tracks(
        mood, user_text=user_text, search_keywords=list(keywords) or None,
        k=5, groq_api_key=GROQ_API_KEY, artist_country=artist_country, exclude=list(exclude),
    )


@st.cache_data(ttl=3600, show_spinner=False, max_entries=64)
def _cached_preview_embedding(preview_url: str):
    """iTunes 30초 프리뷰의 CNN 임베딩. URL 기준 캐시 — 같은 곡이 재정렬
    대상에 다시 나타나도 프리뷰 다운로드+추론을 반복하지 않는다."""
    return embed_preview(preview_url, model, n_mels=cfg.n_mels)


def _render_real_tracks(
    mood: str, user_text: str = "", keywords: list[str] | None = None, artist_country: str | None = None,
    query_embedding: np.ndarray | None = None, extra_exclude: tuple[tuple[str, str], ...] = (),
) -> None:
    """'추천 Top-5' — LLM 추천 곡을 iTunes 카탈로그로 검증해 보여주고,
    각 서비스의 검색/곡 페이지 링크만 제공한다(직접 재생 없음 — 저작권 안전).

    ``query_embedding``이 있으면(음원 검색/업로드/라이브러리 모드 — 입력 오디오가
    존재) 각 곡의 iTunes 공식 30초 프리뷰를 같은 CNN에 통과시켜 얻은 임베딩과의
    코사인 유사도로 Top-5를 재정렬한다 — LLM이 후보를 내고 DL이 순위를 매기는
    구조. 텍스트 모드는 비교할 입력 오디오가 없어 재정렬하지 않는다.
    ``extra_exclude``는 처음부터 제외할 (제목, 아티스트) 목록 — 음원 검색
    모드에서 검색한 곡 자신이 자기 추천 목록에 나오지 않도록 쓴다."""
    # 캐시 키에는 (무드, 문장, 검색어)만 들어가므로, 캐시가 살아있는 동안
    # (ttl=3600) 같은 입력으로 다시 눌러도 원래는 같은 결과가 나온다 — 다른
    # 위젯(표시 옵션 등)을 조작할 때마다 LLM+iTunes를 다시 호출하지 않기
    # 위해 캐싱은 유지하되, "다른 곡" 버튼을 누르면 세션별 재시도 횟수를
    # 캐시 키에 얹어 강제로 새 결과를 뽑는다.
    reroll_key = f"real_tracks_reroll::{mood}::{user_text}::{tuple(keywords or [])}::{artist_country}::{extra_exclude}"
    seen_key = f"{reroll_key}::seen"
    reroll_n = st.session_state.get(reroll_key, 0)
    # LLM은 같은 프롬프트에 매번 가장 유명한 곡부터 답하는 경향이 있어, 캐시만
    # 무효화해서는 "다른 곡"이 여전히 같은 Top-5를 반복했다 — 지금까지 이
    # 무드/조건으로 보여준 모든 곡을 누적해 다음 호출에서 제외 목록으로 넘긴다.
    # extra_exclude는 seen_key가 없을 때(첫 렌더) 시드값으로만 쓰여 검색한
    # 곡 자신을 처음부터 걸러낸다.
    already_shown: list[tuple[str, str]] = st.session_state.get(seen_key, list(extra_exclude))
    col_title, col_reroll = st.columns([4, 1])
    col_title.subheader(f"🎧 '{mood}' 무드 추천 Top-5")
    if col_reroll.button("🔀 다른 곡", key=f"reroll_btn::{reroll_key}", help="같은 무드로 다른 추천을 다시 받습니다"):
        reroll_n += 1
        st.session_state[reroll_key] = reroll_n
    with st.spinner("추천 곡 찾는 중... (LLM 추천 + iTunes 검증)"):
        try:
            real_tracks, provider = _cached_real_tracks(
                mood, user_text, tuple(keywords or []), reroll_n,
                artist_country=artist_country, exclude=tuple(already_shown),
            )
        except Exception:
            real_tracks, provider = [], "itunes"
    if not real_tracks:
        msg = "선택한 국가 조건에 맞는 곡을 찾지 못했습니다. 필터를 해제하거나 다시 시도해보세요." if artist_country else (
            "추천 검색에 실패했습니다 (네트워크 확인). 라이브러리 데모는 아래에서 계속 사용할 수 있습니다."
        )
        st.info(msg)
        return
    st.session_state[seen_key] = already_shown + [(t.title, t.artist) for t in real_tracks]
    ranked: list[tuple] = [(rt, None) for rt in real_tracks]
    reranked = False
    if query_embedding is not None:
        with st.spinner("CNN 임베딩으로 무드 유사도 계산 중... (iTunes 30초 프리뷰 분석)"):
            preview_embs = [_cached_preview_embedding(rt.preview_url) for rt in real_tracks]
        sims = preview_similarities(query_embedding, preview_embs)
        ranked = rerank_with_similarity(real_tracks, sims)
        reranked = any(s is not None for s in sims)
    provider_note = f"추천 경로: {_PROVIDER_LABEL.get(provider, provider)} · 곡 존재 여부는 iTunes 카탈로그로 검증됨"
    if reranked:
        provider_note += " · 입력 곡과의 CNN 임베딩 유사도로 재정렬됨"
    st.caption(provider_note)
    for rt, sim in ranked:
        with st.container(border=True):
            col_art, col_info, col_links = st.columns([1, 3, 2])
            if rt.artwork_url:
                col_art.image(rt.artwork_url, width=80)
            col_info.markdown(f"**{rt.title}**  \n{rt.artist}")
            if rt.album or rt.genre:
                col_info.caption(" · ".join(x for x in [rt.album, rt.genre] if x))
            if sim is not None:
                col_info.caption(f"🎧 무드 유사도 {sim:.4f} — 입력 곡 임베딩 vs 이 곡의 30초 프리뷰 임베딩")
            if rt.reason:
                col_info.caption(f"💡 {rt.reason}")
            col_links.markdown(
                "  \n".join(f"[{name}]({url})" for name, url in rt.links.items())
            )


@st.cache_data(ttl=3600, show_spinner=False, max_entries=16)
def _predict_from_preview(preview_url: str, n_mels: int):
    """검색한 실제 음원의 (무드 확률, 임베딩)을 iTunes 30초 프리뷰로 계산한다.
    업로드 모드(_predict_uploaded_audio)와 동일한 멜스펙 -> CNN 경로를
    재사용한다 — 오디오 소스만 업로드 파일 대신 프리뷰 다운로드로 바뀐다.
    프리뷰가 없거나 디코딩에 실패하면 None."""
    mel = melspec_from_preview(preview_url, n_mels=n_mels)
    if mel is None:
        return None
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        z = model.embed(x)
        probs = torch.sigmoid(model.classifier(z))[0].numpy()
    return probs, z[0].numpy()


@st.cache_data(max_entries=4, show_spinner=False)
def _predict_uploaded_audio(file_bytes: bytes, suffix: str, n_mels: int):
    """업로드 파일의 (무드 확률, 임베딩)을 계산한다. 업로드 상태에서 다른
    위젯(라디오 등)만 바꿔도 rerun마다 멜스펙 추출부터 전부 재실행되던 것을
    파일 내용 기준 캐시로 방지한다."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        mel = extract_melspec(tmp_path, MelspecConfig(n_mels=n_mels))
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            z = model.embed(x)
            probs = torch.sigmoid(model.classifier(z))[0].numpy()
        return probs, z[0].numpy()
    finally:
        os.remove(tmp_path)


def _render_mood_probs(probs: np.ndarray) -> str:
    top_mood = tags[int(probs.argmax())]
    st.metric("최상위 무드", f"{_MOOD_EMOJI.get(top_mood, '')} {top_mood}", f"{probs.max():.1%}")
    for tag, prob in sorted(zip(tags, probs), key=lambda t: -t[1]):
        st.progress(float(prob), text=f"{_MOOD_EMOJI.get(tag, '')} {tag} {prob:.0%}")
    return top_mood


tab_predict, tab_compare, tab_eda, tab_about = st.tabs(
    ["🔍 예측", "📊 모델 성능", "📈 데이터 탐색(EDA)", "ℹ️ 프로젝트 소개"]
)

with tab_predict:
    if input_mode == "음원 검색":
        st.caption(
            "찾고 싶은 곡을 검색하면 그 곡의 무드를 예측하고, 같은 CNN으로 다른 실제 발매곡들과의 "
            "임베딩 유사도를 계산해 비슷한 곡을 추천합니다 — 음원 대 음원 비교입니다."
        )
        search_query = st.text_input("곡 검색", placeholder="예: 아이유 밤편지")
        search_clicked = st.button("검색", use_container_width=True)

        if search_clicked:
            if not search_query.strip():
                st.warning("검색어를 입력해주세요.")
            else:
                with st.spinner("검색 중..."):
                    try:
                        results = itunes_search(search_query, limit=5, country="KR")
                    except requests.RequestException:
                        results = []
                st.session_state.song_search = {"query": search_query, "results": results}

        search_state = st.session_state.get("song_search")
        if search_state is not None and search_state["query"] == search_query and search_query.strip():
            results = search_state["results"]
            if not results:
                st.info("검색 결과가 없습니다. 다른 검색어로 시도해보세요.")
            else:
                options = [f"{r.title} — {r.artist}" for r in results]
                picked = st.selectbox("검색 결과에서 선택", range(len(options)), format_func=lambda i: options[i])
                chosen = results[picked]
                if chosen.artwork_url:
                    st.image(chosen.artwork_url, width=100)

                predict_clicked = st.button("예측 + 추천", key="song_search_predict", use_container_width=True)

                if predict_clicked:
                    with st.spinner("미리듣기로 무드 예측 중..."):
                        prediction = _predict_from_preview(chosen.preview_url, cfg.n_mels)
                    st.session_state.song_result = {
                        "query": search_query, "picked": picked, "prediction": prediction, "chosen": chosen,
                    }

                song_result = st.session_state.get("song_result")
                if (
                    song_result is not None
                    and song_result["query"] == search_query
                    and song_result["picked"] == picked
                ):
                    prediction = song_result["prediction"]
                    if prediction is None:
                        st.warning("이 곡은 미리듣기를 분석할 수 없습니다 (프리뷰 없음 또는 디코딩 실패). 다른 곡을 선택해보세요.")
                    else:
                        probs, query_embedding = prediction
                        picked_chosen = song_result["chosen"]

                        st.divider()
                        st.subheader("예측 무드")
                        top_mood = _render_mood_probs(probs)

                        if show_real:
                            st.divider()
                            _render_real_tracks(
                                top_mood, artist_country=artist_country, query_embedding=query_embedding,
                                extra_exclude=((picked_chosen.title, picked_chosen.artist),),
                            )

                        if show_library:
                            st.divider()
                            with st.expander("📚 라이브러리 데모 — 학습 데이터셋에서 비슷한 무드 (DL 과제 연장)"):
                                idxs, sims = top_k_similar_to_vector(query_embedding, embeddings, k=5)
                                _render_recommendations(idxs, sims)

    elif input_mode == "오디오 업로드":
        st.caption("내 컴퓨터에 있는 오디오 파일을 직접 올려서 무드를 예측하고, 비슷한 무드의 곡을 추천받습니다.")

        if "uploader_reset_n" not in st.session_state:
            st.session_state.uploader_reset_n = 0

        uploaded = st.file_uploader(
            "오디오 파일 업로드 (mp3/wav/ogg/flac/m4a, 최대 50MB)",
            type=["mp3", "wav", "ogg", "flac", "m4a"],
            key=f"audio_uploader_{st.session_state.uploader_reset_n}",
        )

        # 리셋 버튼은 항상 노출한다. 50MB 초과 파일은 업로드가 거부되면서
        # (uploaded=None) 지워지지 않는 오류 칩만 남기는데, 그 칩의 ×버튼은
        # Streamlit 오류 툴팁에 가려 눌리지 않는다. 업로드 성공 여부와 무관하게
        # 이 버튼으로 위젯 키를 바꿔 새 업로더를 그려 항상 비울 수 있게 한다.
        if st.button(
            "🔄 업로더 초기화 / 다른 파일 선택",
            key="reset_uploader_btn",
            help="올린 파일을 지우거나, 50MB 초과 오류 칩이 지워지지 않을 때 눌러주세요.",
        ):
            st.session_state.uploader_reset_n += 1
            st.rerun()

        if uploaded is not None:
            st.audio(uploaded)
            with st.spinner("멜스펙트로그램 추출 + 무드 예측 중..."):
                suffix = os.path.splitext(uploaded.name)[1] or ".mp3"
                probs, query_embedding = _predict_uploaded_audio(
                    uploaded.getvalue(), suffix, cfg.n_mels
                )

            st.divider()
            st.subheader("예측 무드")
            top_mood = _render_mood_probs(probs)

            if show_real:
                st.divider()
                _render_real_tracks(
                    top_mood, artist_country=artist_country, query_embedding=query_embedding,
                )

            if show_library:
                st.divider()
                with st.expander("📚 라이브러리 데모 — 학습 데이터셋에서 비슷한 무드 (DL 과제 연장)"):
                    idxs, sims = top_k_similar_to_vector(query_embedding, embeddings, k=5)
                    _render_recommendations(idxs, sims)

    elif input_mode == "텍스트로 찾기":
        st.caption(
            "지금 기분이나 원하는 분위기를 문장으로 입력하면 LLM이 무드를 분석해 그에 맞는 곡을 추천합니다. "
            "(Ollama 로컬 → Groq API → 키워드 휴리스틱 순 자동 폴백)"
        )
        text_input = st.text_input("지금 기분이 어떤가요?", placeholder="예: 오늘 너무 우울하고 힘들어서 위로받을 음악 듣고 싶어")
        text_clicked = st.button("무드 찾기", use_container_width=True)

        if text_clicked:
            if not text_input.strip():
                st.warning("문장을 입력해주세요.")
            else:
                with st.spinner("LLM이 무드를 분석하는 중..."):
                    analysis = analyze_mood(text_input, tags, groq_api_key=GROQ_API_KEY)
                # 라이브러리 모드와 같은 이유로 세션에 보관 — 결과를 본 뒤
                # 사이드바 위젯을 바꿔도 분석 결과가 유지된다(문장을 수정하면
                # 무효화되어 다시 "무드 찾기"를 눌러야 함).
                st.session_state.text_result = {"text": text_input, "analysis": analysis}

        text_result = st.session_state.get("text_result")
        if text_result is not None and text_result["text"] == text_input:
            analysis = text_result["analysis"]
            best_tag = analysis.mood

            st.divider()
            st.subheader("예측 무드")
            st.success(
                f"추정된 무드: {_MOOD_EMOJI.get(best_tag, '')} **{best_tag}** "
                f"(확신도 {analysis.confidence:.0%})"
            )
            st.caption(f"분석 경로: {_PROVIDER_LABEL.get(analysis.provider, analysis.provider)}")
            if analysis.reason:
                st.info(f"💡 {analysis.reason}")

            if show_real:
                st.divider()
                _render_real_tracks(
                    best_tag, user_text=text_input, keywords=analysis.search_keywords, artist_country=artist_country,
                )

            if show_library:
                st.divider()
                with st.expander(f"📚 라이브러리 데모 — '{best_tag}' 무드에 가장 잘 맞는 학습 곡 (DL 과제 연장)"):
                    track_probs = predict_mood_probs(model, embeddings)
                    tag_idx = tags.index(best_tag)
                    order = np.argsort(-track_probs[:, tag_idx])[:5]
                    sims = track_probs[order, tag_idx]
                    _render_recommendations(order, sims, score_label=f"{best_tag} 확률")

    else:  # "라이브러리 곡 선택" — DL 과제 데모: 학습 데이터셋 안에서 곡을 고른다.
        st.caption("학습에 쓰인 데이터셋에서 곡을 선택해 무드를 예측합니다 (DL 과제 데모).")
        display_options = [_track_display(tid, meta, tags) for tid in track_ids]
        selected = st.selectbox("곡 선택", range(len(display_options)), format_func=lambda i: display_options[i])
        st.caption(f"트랙 ID: {track_ids[selected]}")

        selected_audio = _audio_path(track_ids[selected], manifest)
        if selected_audio:
            st.audio(selected_audio)
        else:
            st.caption("🔇 오디오 파일을 찾을 수 없습니다.")

        predict_clicked = st.button("예측 + 추천", use_container_width=True)

        if predict_clicked:
            mel = load_mel(manifest.iloc[selected]["npy_path"])
            x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits)[0].numpy()
            # 버튼 클릭 상태는 다음 rerun에서 사라지므로, 결과를 세션에
            # 보관해 사이드바 위젯을 조작해도 예측 결과가 화면에서 사라지지
            # 않게 한다(곡을 바꾸면 무효화).
            st.session_state.lib_result = {"selected": selected, "probs": probs}

        lib_result = st.session_state.get("lib_result")
        if lib_result is not None and lib_result["selected"] == selected:
            probs = lib_result["probs"]

            st.divider()
            st.subheader("예측 무드")
            top_mood = _render_mood_probs(probs)

            if show_real:
                st.divider()
                _render_real_tracks(
                    top_mood, artist_country=artist_country, query_embedding=embeddings[selected],
                )

            if show_library:
                st.divider()
                with st.expander("📚 라이브러리 데모 — 학습 데이터셋에서 비슷한 무드 (DL 과제 연장)"):
                    idxs, sims = top_k_similar(selected, embeddings, k=5)
                    _render_recommendations(idxs, sims)

with tab_compare:
    all_metrics = load_all_metrics()
    comparison_df = build_comparison_table(all_metrics)
    if comparison_df.empty:
        st.info("아직 학습된 모델 성능 데이터가 없습니다.")
    else:
        metric_cols = st.columns(len(comparison_df))
        for col, (model_display_name, row) in zip(metric_cols, comparison_df.iterrows()):
            col.metric(model_display_name, f"{row['F1(micro)']:.1%}", help="F1(micro), held-out test 기준")

        st.caption("test split(held-out)으로 평가한 최종 일반화 성능. test가 없는 모델은 마지막 epoch 검증 성능으로 대체.")
        st.divider()
        st.dataframe(comparison_df, use_container_width=True)
        st.bar_chart(comparison_df[["Accuracy", "F1(micro)", "F1(macro)"]], stack=False)

with tab_eda:
    tag_counts = {t: int(meta[f"tag_{t}"].sum()) for t in tags}
    sum_cols = st.columns(len(tags) + 1)
    sum_cols[0].metric("전체 트랙", f"{len(meta):,}곡")
    for col, (tag, count) in zip(sum_cols[1:], tag_counts.items()):
        col.metric(f"{_MOOD_EMOJI.get(tag, '')} {tag}", f"{count:,}곡")

    st.divider()
    st.markdown("**무드 태그 분포** — 곡당 다중 태그 가능")
    tag_df = pd.DataFrame({"곡 수": tag_counts})
    st.bar_chart(tag_df)

    st.divider()
    split_counts = meta["split"].value_counts()
    st.markdown("**train/validation/test 분할**")
    st.bar_chart(split_counts)

    st.divider()
    st.markdown("**곡 길이(초) 분포**")
    st.bar_chart(pd.cut(meta["DURATION"], bins=10).value_counts().sort_index().rename(lambda i: str(i)))

with tab_about:
    st.subheader("음악 오디오 무드 분류 + 콘텐츠 기반 추천")
    st.markdown(
        "CNN으로 오디오 무드를 분류하고, 그 분류 과정에서 학습된 임베딩을 코사인 유사도로 재사용해 "
        "비슷한 무드의 곡을 추천합니다. 분류와 추천을 별도 파이프라인으로 이어붙이지 않고 하나의 모델로 "
        "증명하는 DL 포트폴리오 프로젝트입니다.\n\n"
        "**곡 검색**으로 실제 발매곡의 무드를 예측하고 비슷한 곡을 찾거나, **오디오 파일을 업로드**해 "
        "동일한 모델로 예측하거나, **지금 기분을 문장으로 입력**해 그 무드에 맞는 곡을 찾을 수 있습니다"
        "(🔍 예측 탭 상단의 입력 방식 선택). 학습에 쓰인 데이터셋에서 곡을 고르는 '라이브러리 곡 선택'은 "
        "DL 과제 파이프라인을 그대로 보여주는 데모 모드입니다.\n\n"
        "**LLM 확장**: 텍스트 입력은 LLM(Ollama 로컬 → Groq API → 키워드 휴리스틱 폴백)이 무드를 분석하고, "
        "모든 추천 결과에는 LLM이 제안하고 iTunes 카탈로그로 실존을 검증한 **추천 Top-5**가 "
        "Spotify · YouTube Music · Apple Music 링크와 함께 표시됩니다(직접 재생 없음 — 저작권 안전).\n\n"
        "**DL × LLM 결합 랭킹**: 입력 오디오가 있는 모드(곡 검색/업로드/라이브러리)에서는 추천 후보 각각의 "
        "iTunes 공식 30초 프리뷰를 동일한 CNN에 통과시켜 임베딩을 뽑고, 입력 곡 임베딩과의 코사인 유사도로 "
        "Top-5를 재정렬합니다 — LLM이 후보를 찾고, 학습된 DL 모델이 순위를 매기는 구조입니다 "
        "(프리뷰는 특징 추출 직후 삭제, 저장·재생 없음). '곡 검색' 모드는 이 경로를 라이브러리 없이 "
        "실제 발매곡 대 실제 발매곡으로 직접 수행합니다."
    )
    stat_cols = st.columns(4)
    stat_cols[0].metric("데이터", "MTG-Jamendo")
    stat_cols[1].metric("모델", "MoodCNN")
    stat_cols[2].metric("추천", "임베딩 코사인 유사도")
    stat_cols[3].metric("LLM", "Ollama · Groq")
