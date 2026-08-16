"""LLM-based mood analysis for free-text input.

Replaces the keyword-heuristic-only text mood search with a 3-tier chain:
Ollama (local, free) -> Groq free API (cloud demo, Streamlit Cloud has no
Ollama) -> keyword heuristic (``infer_mood_from_text``, offline last resort).
Every tier maps the user's sentence onto the same 5 trained mood tags, so the
downstream recommendation pipeline (classifier-probability ranking over the
library) is untouched regardless of which tier answered — and regardless of
dataset size (this module only ever sees the tag list, never the tracks).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import requests

from src.recommend.similar import infer_mood_from_text

OLLAMA_URL = os.environ.get("MMR_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("MMR_OLLAMA_MODEL", "gemma4:e2b")
# Local CPU inference: warm calls run ~15s but the first call also loads the
# model (~1min), so the Ollama tier gets a longer budget than the cloud tier.
OLLAMA_TIMEOUT = int(os.environ.get("MMR_OLLAMA_TIMEOUT", "90"))
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("MMR_GROQ_MODEL", "openai/gpt-oss-120b")
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
