"""System prompt and LLM interaction for language feedback.

Uses Anthropic Claude as primary provider with OpenAI fallback.
Includes retry logic, response validation, and timeout enforcement.
"""

import asyncio
import json
import logging
import re

import anthropic
from openai import AsyncOpenAI

from app.config import settings
from app.models import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton clients – reused across requests to leverage connection pooling
# ---------------------------------------------------------------------------
_anthropic_client: anthropic.AsyncAnthropic | None = None
_openai_client: AsyncOpenAI | None = None


def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI()
    return _openai_client


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert multilingual language tutor embedded in a language-learning \
app. A student has written a sentence in their TARGET language and needs \
feedback. Your job is to analyze the sentence and return structured JSON.

━━━  INSTRUCTIONS  ━━━

1. CORRECTED SENTENCE
   • Apply the MINIMAL corrections needed to make the sentence grammatically \
correct and natural-sounding.
   • Preserve the learner's original meaning, vocabulary, and personal style.
   • If the sentence is already correct, return the original sentence EXACTLY \
(same punctuation, spacing, capitalization).

2. ERROR DETECTION
   • Identify every distinct error. For each one, extract the EXACT erroneous \
span from the original sentence as "original" and provide its "correction".
   • The "original" field MUST be a substring that appears verbatim in the \
input sentence.
   • Classify each error using EXACTLY one of these types:
     grammar | spelling | word_choice | punctuation | word_order | \
missing_word | extra_word | conjugation | gender_agreement | \
number_agreement | tone_register | other
   • Write a concise, friendly explanation (1–2 sentences) in the learner's \
NATIVE language. Explain WHY it is wrong and HOW to remember the correct form. \
Avoid jargon.

3. IS_CORRECT
   • Set to true ONLY when the sentence has zero errors.
   • If true, the "errors" array MUST be empty and "corrected_sentence" MUST \
equal the input.

4. DIFFICULTY (CEFR)
   Rate the SENTENCE COMPLEXITY (not the learner's skill) on the CEFR scale:
   • A1 — very basic phrases, high-frequency words, simple present tense
   • A2 — short sentences, everyday topics, basic past/future tense
   • B1 — connected sentences, opinions, compound tenses
   • B2 — complex arguments, abstract topics, subjunctive/conditional
   • C1 — nuanced expression, idiomatic language, sophisticated structures
   • C2 — near-native complexity, literary/academic register

5. LANGUAGE HANDLING
   • Work with ANY language or writing system (Latin, CJK, Cyrillic, Arabic, \
Devanagari, etc.).
   • For languages with particles (Japanese, Korean), check particle usage.
   • For gendered languages (French, German, Spanish, Arabic), check agreement.
   • For tonal languages, check tone marks where applicable.

━━━  OUTPUT FORMAT  ━━━

Return ONLY valid JSON matching this EXACT structure (no markdown, no \
commentary, no extra keys):

{
  "corrected_sentence": "...",
  "is_correct": true/false,
  "errors": [
    {
      "original": "exact span from input",
      "correction": "corrected span",
      "error_type": "one of the 12 types above",
      "explanation": "in native language"
    }
  ],
  "difficulty": "A1|A2|B1|B2|C1|C2"
}

If there are no errors, use: "errors": []
"""


def _build_user_message(request: FeedbackRequest) -> str:
    """Build the user prompt from the feedback request."""
    return (
        f"Target language: {request.target_language}\n"
        f"Native language: {request.native_language}\n"
        f"Sentence to analyze: {request.sentence}"
    )


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code fences."""
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find anything that looks like JSON
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM response: {text[:200]}")


# ---------------------------------------------------------------------------
# Provider-specific call functions
# ---------------------------------------------------------------------------


async def _call_anthropic(user_message: str) -> dict:
    """Call Anthropic Claude and return parsed JSON."""
    client = _get_anthropic_client()
    response = await client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=settings.MAX_TOKENS,
        temperature=settings.TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    content = response.content[0].text
    return _extract_json(content)


async def _call_openai(user_message: str) -> dict:
    """Call OpenAI and return parsed JSON."""
    client = _get_openai_client()
    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
    )
    content = response.choices[0].message.content
    return _extract_json(content)


# ---------------------------------------------------------------------------
# Main feedback function with retry and fallback
# ---------------------------------------------------------------------------


async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Analyze a learner sentence and return structured feedback.

    Strategy:
    1. Try the primary provider (Anthropic) with retries.
    2. If primary fails, try the fallback provider (OpenAI) with retries.
    3. Validate and auto-correct the response for consistency.
    4. Enforce a hard timeout to stay within the 30-second requirement.
    """
    user_message = _build_user_message(request)

    # Determine provider order based on config and key availability
    providers: list[tuple[str, object]] = []

    if settings.PRIMARY_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
        providers.append(("anthropic", _call_anthropic))
        if settings.OPENAI_API_KEY:
            providers.append(("openai", _call_openai))
    elif settings.PRIMARY_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        providers.append(("openai", _call_openai))
        if settings.ANTHROPIC_API_KEY:
            providers.append(("anthropic", _call_anthropic))
    elif settings.ANTHROPIC_API_KEY:
        providers.append(("anthropic", _call_anthropic))
    elif settings.OPENAI_API_KEY:
        providers.append(("openai", _call_openai))
    else:
        raise RuntimeError("No LLM API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")

    last_error: Exception | None = None

    for provider_name, call_fn in providers:
        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                logger.info(
                    "Calling %s (attempt %d/%d)",
                    provider_name,
                    attempt,
                    settings.MAX_RETRIES,
                )
                data = await asyncio.wait_for(
                    call_fn(user_message),
                    timeout=settings.REQUEST_TIMEOUT,
                )
                response = FeedbackResponse(**data)
                response = response.ensure_consistency()
                return response

            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"{provider_name} timed out after {settings.REQUEST_TIMEOUT}s"
                )
                logger.warning("Timeout on %s attempt %d", provider_name, attempt)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Error on %s attempt %d: %s", provider_name, attempt, exc
                )

            # Brief pause before retry (exponential-ish backoff)
            if attempt < settings.MAX_RETRIES:
                await asyncio.sleep(0.5 * attempt)

        logger.warning("All %d attempts on %s exhausted", settings.MAX_RETRIES, provider_name)

    raise RuntimeError(f"All providers failed. Last error: {last_error}")
