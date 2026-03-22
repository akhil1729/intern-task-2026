"""Unit tests -- run without an API key using mocked LLM responses.

Covers: multiple languages, non-Latin scripts, correct sentences, error
validation, consistency checks, edge cases (short/long sentences).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.feedback import get_feedback, _extract_json
from app.models import (
    ErrorDetail,
    FeedbackRequest,
    FeedbackResponse,
    VALID_ERROR_TYPES,
    VALID_DIFFICULTIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_anthropic_response(response_data: dict) -> MagicMock:
    """Build a mock Anthropic response."""
    text_block = MagicMock()
    text_block.text = json.dumps(response_data)
    response = MagicMock()
    response.content = [text_block]
    return response


def _mock_openai_response(response_data: dict) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(response_data)
    completion = MagicMock()
    completion.choices = [choice]
    return completion


async def _run_feedback_with_mock(mock_response_data: dict, request: FeedbackRequest):
    """Run get_feedback with a mocked Anthropic response."""
    with patch("app.feedback._get_anthropic_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(mock_response_data)
        )
        mock_client_fn.return_value = mock_client

        with patch("app.feedback.settings") as mock_settings:
            mock_settings.PRIMARY_PROVIDER = "anthropic"
            mock_settings.ANTHROPIC_API_KEY = "test-key"
            mock_settings.OPENAI_API_KEY = None
            mock_settings.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
            mock_settings.TEMPERATURE = 0.1
            mock_settings.MAX_TOKENS = 2048
            mock_settings.MAX_RETRIES = 1
            mock_settings.REQUEST_TIMEOUT = 25

            return await get_feedback(request)


# ---------------------------------------------------------------------------
# 1. Spanish — conjugation error
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_spanish_conjugation_error():
    mock_response = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You mixed two verb forms.",
            }
        ],
        "difficulty": "A2",
    }

    request = FeedbackRequest(
        sentence="Yo soy fue al mercado ayer.",
        target_language="Spanish",
        native_language="English",
    )
    result = await _run_feedback_with_mock(mock_response, request)

    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.difficulty == "A2"


# ---------------------------------------------------------------------------
# 2. French — multiple gender agreement errors
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_french_multiple_gender_errors():
    mock_response = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": "'Chat' est masculin.",
            },
            {
                "original": "le table",
                "correction": "la table",
                "error_type": "gender_agreement",
                "explanation": "'Table' est féminin.",
            },
        ],
        "difficulty": "A1",
    }

    request = FeedbackRequest(
        sentence="La chat noir est sur le table.",
        target_language="French",
        native_language="French",
    )
    result = await _run_feedback_with_mock(mock_response, request)

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)


# ---------------------------------------------------------------------------
# 3. German — correct sentence (no errors)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_german_correct_sentence():
    original = "Ich habe gestern einen interessanten Film gesehen."
    mock_response = {
        "corrected_sentence": original,
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    }

    request = FeedbackRequest(
        sentence=original,
        target_language="German",
        native_language="English",
    )
    result = await _run_feedback_with_mock(mock_response, request)

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == original


# ---------------------------------------------------------------------------
# 4. Japanese — particle error (non-Latin script)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_japanese_particle_error():
    mock_response = {
        "corrected_sentence": "私は東京に住んでいます。",
        "is_correct": False,
        "errors": [
            {
                "original": "を",
                "correction": "に",
                "error_type": "grammar",
                "explanation": "The verb 住む (to live) takes the particle に to indicate location.",
            }
        ],
        "difficulty": "A2",
    }

    request = FeedbackRequest(
        sentence="私は東京を住んでいます。",
        target_language="Japanese",
        native_language="English",
    )
    result = await _run_feedback_with_mock(mock_response, request)

    assert result.is_correct is False
    assert len(result.errors) == 1
    assert "に" in result.errors[0].correction


# ---------------------------------------------------------------------------
# 5. Korean — particle error (non-Latin script)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_korean_particle_error():
    mock_response = {
        "corrected_sentence": "나는 학교에 갑니다.",
        "is_correct": False,
        "errors": [
            {
                "original": "학교를",
                "correction": "학교에",
                "error_type": "grammar",
                "explanation": "여기서는 방향을 나타내는 조사 '에'를 사용해야 합니다.",
            }
        ],
        "difficulty": "A1",
    }

    request = FeedbackRequest(
        sentence="나는 학교를 갑니다.",
        target_language="Korean",
        native_language="Korean",
    )
    result = await _run_feedback_with_mock(mock_response, request)

    assert result.is_correct is False
    assert len(result.errors) >= 1


# ---------------------------------------------------------------------------
# 6. Russian — spelling error (Cyrillic)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_russian_spelling_error():
    mock_response = {
        "corrected_sentence": "Я хочу купить молоко.",
        "is_correct": False,
        "errors": [
            {
                "original": "малоко",
                "correction": "молоко",
                "error_type": "spelling",
                "explanation": "Правильное написание: 'молоко', через 'о'.",
            }
        ],
        "difficulty": "A1",
    }

    request = FeedbackRequest(
        sentence="Я хочу купить малоко.",
        target_language="Russian",
        native_language="Russian",
    )
    result = await _run_feedback_with_mock(mock_response, request)

    assert result.is_correct is False
    assert result.errors[0].error_type == "spelling"


# ---------------------------------------------------------------------------
# 7. Chinese — word order error
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_chinese_word_order_error():
    mock_response = {
        "corrected_sentence": "我昨天去了商店。",
        "is_correct": False,
        "errors": [
            {
                "original": "去了昨天",
                "correction": "昨天去了",
                "error_type": "word_order",
                "explanation": "In Chinese, time words like 昨天 come before the verb.",
            }
        ],
        "difficulty": "A2",
    }

    request = FeedbackRequest(
        sentence="我去了昨天商店。",
        target_language="Chinese",
        native_language="English",
    )
    result = await _run_feedback_with_mock(mock_response, request)

    assert result.is_correct is False
    assert result.errors[0].error_type == "word_order"


# ---------------------------------------------------------------------------
# 8. Arabic — gender agreement (RTL script)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_arabic_gender_agreement():
    mock_response = {
        "corrected_sentence": "الطالبة ذهبت إلى المدرسة.",
        "is_correct": False,
        "errors": [
            {
                "original": "ذهب",
                "correction": "ذهبت",
                "error_type": "gender_agreement",
                "explanation": "الفعل يجب أن يتوافق مع المؤنث: 'ذهبت' بدلاً من 'ذهب'.",
            }
        ],
        "difficulty": "A2",
    }

    request = FeedbackRequest(
        sentence="الطالبة ذهب إلى المدرسة.",
        target_language="Arabic",
        native_language="Arabic",
    )
    result = await _run_feedback_with_mock(mock_response, request)

    assert result.is_correct is False
    assert result.errors[0].error_type == "gender_agreement"


# ---------------------------------------------------------------------------
# 9. Portuguese — spelling + grammar (multiple error types)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_portuguese_mixed_errors():
    mock_response = {
        "corrected_sentence": "Eu quero comprar um presente para minha irmã, mas não sei do que ela gosta.",
        "is_correct": False,
        "errors": [
            {
                "original": "prezente",
                "correction": "presente",
                "error_type": "spelling",
                "explanation": "'Gift' in Portuguese is 'presente' with an 's', not 'z'.",
            },
            {
                "original": "o que ela gosta",
                "correction": "do que ela gosta",
                "error_type": "grammar",
                "explanation": "'Gostar' requires the preposition 'de'.",
            },
        ],
        "difficulty": "B1",
    }

    request = FeedbackRequest(
        sentence="Eu quero comprar um prezente para minha irmã, mas não sei o que ela gosta.",
        target_language="Portuguese",
        native_language="English",
    )
    result = await _run_feedback_with_mock(mock_response, request)

    assert result.is_correct is False
    assert len(result.errors) == 2
    error_types = {e.error_type for e in result.errors}
    assert "spelling" in error_types
    assert "grammar" in error_types


# ---------------------------------------------------------------------------
# Model Validation Tests
# ---------------------------------------------------------------------------


class TestErrorTypeValidation:
    """Verify error_type validator catches and maps invalid types."""

    def test_valid_error_type(self):
        detail = ErrorDetail(
            original="x", correction="y", error_type="grammar", explanation="test"
        )
        assert detail.error_type == "grammar"

    def test_maps_verb_conjugation_to_conjugation(self):
        detail = ErrorDetail(
            original="x",
            correction="y",
            error_type="verb_conjugation",
            explanation="test",
        )
        assert detail.error_type == "conjugation"

    def test_maps_unknown_to_other(self):
        detail = ErrorDetail(
            original="x",
            correction="y",
            error_type="totally_unknown_type",
            explanation="test",
        )
        assert detail.error_type == "other"

    def test_maps_particle_to_grammar(self):
        detail = ErrorDetail(
            original="を", correction="に", error_type="particle", explanation="test"
        )
        assert detail.error_type == "grammar"

    def test_case_insensitive(self):
        detail = ErrorDetail(
            original="x", correction="y", error_type="GRAMMAR", explanation="test"
        )
        assert detail.error_type == "grammar"


class TestDifficultyValidation:
    """Verify difficulty validator enforces CEFR levels."""

    def test_valid_difficulty(self):
        resp = FeedbackResponse(
            corrected_sentence="test",
            is_correct=True,
            errors=[],
            difficulty="B2",
        )
        assert resp.difficulty == "B2"

    def test_lowercase_normalized(self):
        resp = FeedbackResponse(
            corrected_sentence="test",
            is_correct=True,
            errors=[],
            difficulty="b2",
        )
        assert resp.difficulty == "B2"

    def test_invalid_defaults_to_b1(self):
        resp = FeedbackResponse(
            corrected_sentence="test",
            is_correct=True,
            errors=[],
            difficulty="Z9",
        )
        assert resp.difficulty == "B1"


class TestConsistencyCheck:
    """Verify the ensure_consistency method."""

    def test_correct_with_errors_becomes_incorrect(self):
        resp = FeedbackResponse(
            corrected_sentence="fixed",
            is_correct=True,
            errors=[
                ErrorDetail(
                    original="x",
                    correction="y",
                    error_type="grammar",
                    explanation="test",
                )
            ],
            difficulty="A1",
        )
        resp = resp.ensure_consistency()
        assert resp.is_correct is False

    def test_incorrect_with_no_errors_becomes_correct(self):
        resp = FeedbackResponse(
            corrected_sentence="test",
            is_correct=False,
            errors=[],
            difficulty="A1",
        )
        resp = resp.ensure_consistency()
        assert resp.is_correct is True


# ---------------------------------------------------------------------------
# JSON Extraction Tests
# ---------------------------------------------------------------------------


class TestJsonExtraction:
    """Verify _extract_json handles various LLM output formats."""

    def test_plain_json(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_code_fence(self):
        text = '```json\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"key": "value"}\nDone!'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            _extract_json("not json at all")
