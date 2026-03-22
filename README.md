# Language Feedback API

An LLM-powered language feedback API that analyzes learner-written sentences and returns structured correction feedback. Built for the [Pangea Chat](https://pangea.chat) Gen AI Intern Task.

## Design Decisions

### Architecture

The API follows a straightforward request → LLM → validated response pipeline:

```
POST /feedback → build prompt → call LLM → extract JSON → validate → respond
```

Key design choices and their rationale:

1. **Anthropic Claude as primary, OpenAI as fallback.** Claude Sonnet provides excellent multilingual accuracy and instruction-following. The fallback ensures reliability — if one provider has an outage, the API keeps working.

2. **Detailed, constraint-heavy system prompt.** Rather than relying on model intelligence alone, the prompt explicitly defines all 12 error types, provides a CEFR rubric, and specifies exact JSON output format. This reduces hallucinated error types and inconsistent difficulty ratings.

3. **Response validation and auto-correction.** The API validates every response before returning it:
   - `error_type` values are checked against the allowed enum, with a fuzzy mapping for common LLM mislabels (e.g., "verb_conjugation" → "conjugation", "particle" → "grammar")
   - `difficulty` is validated against CEFR levels
   - Logical consistency is enforced: `is_correct=true` ↔ empty errors list

4. **Retry with provider fallback.** Each provider gets up to 2 attempts with exponential backoff. If the primary provider fails entirely, the fallback provider is tried. This handles transient API failures without exceeding the 30-second timeout.

5. **25-second hard timeout per LLM call.** Leaves 5 seconds of buffer within the 30-second requirement for network overhead and response processing.

### Prompt Strategy

The system prompt is designed around three principles:

1. **Exhaustive constraints prevent LLM drift.** Every allowed error type, every CEFR level, and the exact JSON schema are enumerated in the prompt. This eliminates fuzzy interpretation.

2. **Multilingual awareness is explicit.** The prompt includes specific instructions for particles (Japanese/Korean), gendered languages (French/German/Spanish/Arabic), and non-Latin scripts. This steers the model to check language-specific error patterns rather than defaulting to a Latin-language bias.

3. **Minimal correction preserves learner voice.** The prompt emphasizes making only necessary corrections. This is pedagogically important — over-correction discourages learners and obscures the specific error they need to understand.

### Cost Efficiency

- **Claude Sonnet** is the best quality-per-dollar for structured output tasks
- **Temperature 0.1** reduces randomness without sacrificing accuracy
- **Max 2048 tokens** caps response cost (typical responses use ~200-400 tokens)
- **Singleton clients** reuse HTTP connections across requests
- The architecture supports swapping to cheaper models via environment variables without code changes

## How to Run

### Prerequisites

- Python 3.11+
- An Anthropic API key (or OpenAI API key)

### Local Development

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (and/or OPENAI_API_KEY)

# 4. Start the server
uvicorn app.main:app --reload

# 5. Test it
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Yo soy fue al mercado ayer.", "target_language": "Spanish", "native_language": "English"}'
```

### Docker

```bash
cp .env.example .env
# Edit .env with your API key
docker compose up --build
```

### Running Tests

```bash
# Unit tests (no API key needed — uses mocked responses)
pytest tests/test_feedback_unit.py tests/test_schema.py -v

# Integration tests (requires API key in .env — makes real LLM calls)
pytest tests/test_feedback_integration.py -v

# All tests
pytest tests/ -v
```

## Test Coverage

### Unit Tests (19 tests, no API key required)

| Test | What it covers |
|------|---------------|
| Spanish conjugation | Single verb error detection |
| French gender agreement | Multiple errors in one sentence |
| German correct sentence | `is_correct=true` handling |
| Japanese particles | Non-Latin script (CJK), particle usage |
| Korean particles | Non-Latin script (CJK), particle usage |
| Russian spelling | Cyrillic script |
| Chinese word order | Character-based writing system |
| Arabic gender agreement | RTL script, gender agreement |
| Portuguese mixed errors | Multiple error types in one sentence |
| Error type validation | Enum enforcement and fuzzy mapping |
| Difficulty validation | CEFR level enforcement |
| Consistency checks | `is_correct` ↔ errors list agreement |
| JSON extraction | Handles bare JSON, code fences, surrounding text |

### Integration Tests (12 tests, requires API key)

Covers 10 languages (Spanish, French, German, Japanese, Korean, Russian, Chinese, Portuguese, Arabic, Italian) plus edge cases (very short sentences, native-language explanations). Every test validates response time, schema compliance, and logical consistency.

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `PRIMARY_PROVIDER` | `anthropic` | Which provider to try first |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic model to use |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `TEMPERATURE` | `0.1` | LLM temperature (lower = more deterministic) |
| `MAX_TOKENS` | `2048` | Maximum tokens in LLM response |
| `MAX_RETRIES` | `2` | Retry attempts per provider |
| `REQUEST_TIMEOUT` | `25` | Timeout per LLM call (seconds) |

## API Reference

### `POST /feedback`

Analyze a sentence and return structured correction feedback.

**Request:**
```json
{
  "sentence": "La chat noir est sur le table.",
  "target_language": "French",
  "native_language": "English"
}
```

**Response:**
```json
{
  "corrected_sentence": "Le chat noir est sur la table.",
  "is_correct": false,
  "errors": [
    {
      "original": "La chat",
      "correction": "Le chat",
      "error_type": "gender_agreement",
      "explanation": "'Chat' (cat) is masculine in French, so it uses 'le', not 'la'."
    },
    {
      "original": "le table",
      "correction": "la table",
      "error_type": "gender_agreement",
      "explanation": "'Table' is feminine in French, so it uses 'la', not 'le'."
    }
  ],
  "difficulty": "A1"
}
```

### `GET /health`

Returns `{"status": "ok"}` with HTTP 200.

## Trade-offs and Future Improvements

**What I'd add given more time:**

- **Response caching** — hash the (sentence, target_language) pair and cache results. Most language learning apps see repeated sentences. Even a simple LRU cache would cut costs significantly.
- **Batch endpoint** — accept multiple sentences in one request to reduce HTTP overhead for classroom use.
- **Confidence scoring** — have the LLM rate its confidence for each error, enabling the UI to flag uncertain corrections.
- **Streaming** — use SSE to stream partial results for faster perceived response time on slower models.

**What I deliberately kept simple:**

- No database or persistent storage — the API is stateless, making it trivially scalable.
- No authentication — the scorer provides its own keys, and auth would add complexity without value here.
- No language detection — trusting the user's `target_language` field rather than adding a detection step that could misidentify closely-related languages.
