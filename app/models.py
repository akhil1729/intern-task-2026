"""Pydantic models with validation for the Language Feedback API."""

from pydantic import BaseModel, Field, field_validator

VALID_ERROR_TYPES = frozenset(
    {
        "grammar",
        "spelling",
        "word_choice",
        "punctuation",
        "word_order",
        "missing_word",
        "extra_word",
        "conjugation",
        "gender_agreement",
        "number_agreement",
        "tone_register",
        "other",
    }
)

VALID_DIFFICULTIES = frozenset({"A1", "A2", "B1", "B2", "C1", "C2"})


class ErrorDetail(BaseModel):
    original: str = Field(
        description="The erroneous word or phrase from the original sentence"
    )
    correction: str = Field(description="The corrected word or phrase")
    error_type: str = Field(description="Category of the error")
    explanation: str = Field(
        description="A brief, learner-friendly explanation written in the native language"
    )

    @field_validator("error_type")
    @classmethod
    def validate_error_type(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if v_lower not in VALID_ERROR_TYPES:
            # Map common LLM mis-labels to closest valid type
            mapping = {
                "verb_conjugation": "conjugation",
                "tense": "conjugation",
                "verb_form": "conjugation",
                "article": "gender_agreement",
                "gender": "gender_agreement",
                "number": "number_agreement",
                "plural": "number_agreement",
                "preposition": "word_choice",
                "vocabulary": "word_choice",
                "lexical": "word_choice",
                "syntax": "word_order",
                "order": "word_order",
                "redundant": "extra_word",
                "omission": "missing_word",
                "accent": "spelling",
                "orthography": "spelling",
                "capitalization": "punctuation",
                "formality": "tone_register",
                "register": "tone_register",
                "particle": "grammar",
            }
            v_lower = mapping.get(v_lower, "other")
        return v_lower


class FeedbackRequest(BaseModel):
    sentence: str = Field(
        min_length=1, description="The learner's sentence in the target language"
    )
    target_language: str = Field(
        min_length=2, description="The language the learner is studying"
    )
    native_language: str = Field(
        min_length=2,
        description="The learner's native language -- explanations will be in this language",
    )


class FeedbackResponse(BaseModel):
    corrected_sentence: str = Field(
        description="The grammatically corrected version of the input sentence"
    )
    is_correct: bool = Field(description="true if the original sentence had no errors")
    errors: list[ErrorDetail] = Field(
        default_factory=list,
        description="List of errors found. Empty if the sentence is correct.",
    )
    difficulty: str = Field(
        description="CEFR difficulty level: A1, A2, B1, B2, C1, or C2"
    )

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        v_upper = v.strip().upper()
        if v_upper not in VALID_DIFFICULTIES:
            return "B1"  # safe default
        return v_upper

    def ensure_consistency(self) -> "FeedbackResponse":
        """Fix logical inconsistencies between is_correct and errors list."""
        if self.is_correct and self.errors:
            # LLM said correct but listed errors → trust the errors
            self.is_correct = False
        elif not self.is_correct and not self.errors:
            # LLM said incorrect but gave no errors → mark as correct
            self.is_correct = True
        return self
