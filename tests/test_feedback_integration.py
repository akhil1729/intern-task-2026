"""Integration tests -- require ANTHROPIC_API_KEY or OPENAI_API_KEY to be set.

Run with: pytest tests/test_feedback_integration.py -v

These tests make real LLM API calls. They validate accuracy, schema
compliance, and response time across 8+ languages including non-Latin scripts.
"""

import os
import time

import pytest
from dotenv import load_dotenv

load_dotenv()

from app.feedback import get_feedback
from app.models import FeedbackRequest, VALID_ERROR_TYPES, VALID_DIFFICULTIES

HAS_API_KEY = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))

pytestmark = pytest.mark.skipif(
    not HAS_API_KEY,
    reason="No API key set -- skipping integration tests",
)


def _validate_response(result, *, expect_correct: bool | None = None):
    """Common validation applied to every response."""
    # Schema compliance
    assert result.difficulty in VALID_DIFFICULTIES
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES
        assert len(error.explanation) > 0
        assert len(error.original) > 0
        assert len(error.correction) > 0

    # Consistency
    if result.is_correct:
        assert result.errors == [], "is_correct=True but errors list is not empty"
    else:
        assert len(result.errors) >= 1, "is_correct=False but no errors listed"

    # Explicit correctness check if provided
    if expect_correct is not None:
        assert result.is_correct is expect_correct


# ---------------------------------------------------------------------------
# 1. Spanish — conjugation error
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_spanish_conjugation():
    start = time.time()
    result = await get_feedback(
        FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )
    elapsed = time.time() - start

    _validate_response(result, expect_correct=False)
    assert elapsed < 30, f"Response took {elapsed:.1f}s (over 30s limit)"
    assert "fui" in result.corrected_sentence.lower() or "fue" not in result.corrected_sentence.lower()


# ---------------------------------------------------------------------------
# 2. French — gender agreement
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_french_gender():
    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )
    _validate_response(result, expect_correct=False)
    assert len(result.errors) >= 1


# ---------------------------------------------------------------------------
# 3. German — correct sentence
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_german_correct():
    original = "Ich habe gestern einen interessanten Film gesehen."
    result = await get_feedback(
        FeedbackRequest(
            sentence=original,
            target_language="German",
            native_language="English",
        )
    )
    _validate_response(result, expect_correct=True)
    assert result.corrected_sentence == original


# ---------------------------------------------------------------------------
# 4. Japanese — particle error (CJK)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_japanese_particle():
    result = await get_feedback(
        FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
    )
    _validate_response(result, expect_correct=False)
    assert "に" in result.corrected_sentence


# ---------------------------------------------------------------------------
# 5. Korean — particle error (CJK)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_korean_particle():
    result = await get_feedback(
        FeedbackRequest(
            sentence="나는 학교를 갑니다.",
            target_language="Korean",
            native_language="English",
        )
    )
    _validate_response(result, expect_correct=False)


# ---------------------------------------------------------------------------
# 6. Russian — spelling error (Cyrillic)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_russian_spelling():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Я хочу купить малоко.",
            target_language="Russian",
            native_language="English",
        )
    )
    _validate_response(result, expect_correct=False)
    assert "молоко" in result.corrected_sentence


# ---------------------------------------------------------------------------
# 7. Chinese — word order error
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_chinese_word_order():
    result = await get_feedback(
        FeedbackRequest(
            sentence="我去了昨天商店。",
            target_language="Chinese",
            native_language="English",
        )
    )
    _validate_response(result, expect_correct=False)


# ---------------------------------------------------------------------------
# 8. Portuguese — mixed errors (spelling + grammar)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_portuguese_mixed():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Eu quero comprar um prezente para minha irmã, mas não sei o que ela gosta.",
            target_language="Portuguese",
            native_language="English",
        )
    )
    _validate_response(result, expect_correct=False)
    assert "presente" in result.corrected_sentence


# ---------------------------------------------------------------------------
# 9. Arabic — gender agreement (RTL)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_arabic_gender():
    result = await get_feedback(
        FeedbackRequest(
            sentence="الطالبة ذهب إلى المدرسة.",
            target_language="Arabic",
            native_language="English",
        )
    )
    _validate_response(result, expect_correct=False)


# ---------------------------------------------------------------------------
# 10. Italian — correct sentence
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_italian_correct():
    original = "Ho mangiato una pizza deliziosa ieri sera."
    result = await get_feedback(
        FeedbackRequest(
            sentence=original,
            target_language="Italian",
            native_language="English",
        )
    )
    _validate_response(result, expect_correct=True)
    assert result.corrected_sentence == original


# ---------------------------------------------------------------------------
# 11. Edge case — very short sentence
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_short_sentence():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Hola.",
            target_language="Spanish",
            native_language="English",
        )
    )
    _validate_response(result, expect_correct=True)


# ---------------------------------------------------------------------------
# 12. Edge case — native language explanation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_explanation_in_native_language():
    """Verify the explanation is written in the native language (Japanese)."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="She go to school yesterday.",
            target_language="English",
            native_language="Japanese",
        )
    )
    _validate_response(result, expect_correct=False)
    # At least one explanation should contain Japanese characters
    has_japanese = any(
        any("\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff" for c in e.explanation)
        for e in result.errors
    )
    assert has_japanese, "Explanations should be in native language (Japanese)"
