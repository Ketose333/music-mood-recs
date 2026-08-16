"""Tests for src/llm/mood_analyzer.py — parsing, validation, fallback chain.

No network access: every LLM/HTTP call is monkeypatched, so these run the
same offline as the keyword-heuristic tier they protect.
"""

import pytest
import requests

from src.llm import mood_analyzer as ma

TAGS = ["happy", "energetic", "relaxing", "film", "dark"]


class TestChatGroq:
    def test_default_model_is_sent_in_http_payload(self, monkeypatch):
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": '{"mood": "happy"}'}}]}

        def fake_post(url, *, headers, json, timeout):
            captured.update(url=url, headers=headers, json=json, timeout=timeout)
            return FakeResponse()

        monkeypatch.setattr(ma.requests, "post", fake_post)

        result = ma.chat_groq("기분이 좋아", "dummy-key-for-test")

        assert result == '{"mood": "happy"}'
        assert captured["url"] == ma.GROQ_API_URL
        assert captured["json"]["model"] == "openai/gpt-oss-120b"
        assert captured["json"]["response_format"] == {"type": "json_object"}


class TestParseMoodResponse:
    def test_valid_json(self):
        raw = '{"mood": "dark", "confidence": 0.9, "reason": "우울함", "search_keywords": ["sad songs"]}'
        parsed = ma.parse_mood_response(raw, TAGS)
        assert parsed == {
            "mood": "dark",
            "confidence": 0.9,
            "reason": "우울함",
            "search_keywords": ["sad songs"],
        }

    def test_json_wrapped_in_markdown_fence(self):
        raw = '```json\n{"mood": "happy", "confidence": 0.7, "reason": "r", "search_keywords": []}\n```'
        parsed = ma.parse_mood_response(raw, TAGS)
        assert parsed["mood"] == "happy"

    def test_hallucinated_mood_rejected(self):
        raw = '{"mood": "melancholic", "confidence": 0.9, "reason": "r"}'
        assert ma.parse_mood_response(raw, TAGS) is None

    def test_no_json_rejected(self):
        assert ma.parse_mood_response("dark이 어울립니다", TAGS) is None

    def test_invalid_json_rejected(self):
        assert ma.parse_mood_response("{mood: dark", TAGS) is None

    def test_confidence_clamped_and_defaulted(self):
        assert ma.parse_mood_response('{"mood": "dark", "confidence": 3.0}', TAGS)["confidence"] == 1.0
        assert ma.parse_mood_response('{"mood": "dark", "confidence": "?"}', TAGS)["confidence"] == 0.5

    def test_mood_case_normalized(self):
        assert ma.parse_mood_response('{"mood": " Dark "}', TAGS)["mood"] == "dark"

    def test_keywords_capped_at_four(self):
        raw = '{"mood": "dark", "search_keywords": ["a", "b", "c", "d", "e"]}'
        assert len(ma.parse_mood_response(raw, TAGS)["search_keywords"]) == 4


class TestAnalyzeMood:
    def test_uses_ollama_when_available(self, monkeypatch):
        monkeypatch.setattr(
            ma, "chat_ollama",
            lambda prompt, timeout=30: '{"mood": "relaxing", "confidence": 0.8, "reason": "쉬고 싶음", "search_keywords": ["chill"]}',
        )
        result = ma.analyze_mood("쉬고 싶어", TAGS)
        assert result.provider == "ollama"
        assert result.mood == "relaxing"
        assert result.search_keywords == ["chill"]

    def test_falls_back_to_groq_when_ollama_down(self, monkeypatch):
        def raise_conn(prompt, timeout=30):
            raise requests.ConnectionError("no ollama")

        monkeypatch.setattr(ma, "chat_ollama", raise_conn)
        monkeypatch.setattr(
            ma, "chat_groq",
            lambda prompt, api_key, timeout=30: '{"mood": "happy", "confidence": 0.9, "reason": "신남"}',
        )
        result = ma.analyze_mood("너무 신나!", TAGS, groq_api_key="dummy-key-for-test")
        assert result.provider == "groq"
        assert result.mood == "happy"

    def test_falls_back_to_keyword_heuristic_when_no_llm(self, monkeypatch):
        def raise_conn(*args, **kwargs):
            raise requests.ConnectionError("offline")

        monkeypatch.setattr(ma, "chat_ollama", raise_conn)
        result = ma.analyze_mood("오늘 너무 우울하고 힘들어", TAGS)  # no groq key
        assert result.provider == "keyword"
        assert result.mood == "dark"

    def test_keyword_fallback_defaults_to_first_tag_on_no_match(self, monkeypatch):
        monkeypatch.setattr(ma, "llm_chat", lambda prompt, groq_api_key=None: None)
        result = ma.analyze_mood("asdf qwer", TAGS)
        assert result.provider == "keyword"
        assert result.mood == TAGS[0]

    def test_bad_llm_json_falls_through_to_keyword(self, monkeypatch):
        monkeypatch.setattr(ma, "llm_chat", lambda prompt, groq_api_key=None: ("not json at all", "ollama"))
        result = ma.analyze_mood("편안한 음악 듣고 싶어", TAGS)
        assert result.provider == "keyword"
        assert result.mood == "relaxing"


class TestBuildMoodPrompt:
    def test_prompt_contains_tags_and_text(self):
        prompt = ma.build_mood_prompt("졸려", TAGS)
        for tag in TAGS:
            assert tag in prompt
        assert "졸려" in prompt
