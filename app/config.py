"""Centralized configuration via environment variables.

Settings are read lazily via properties so that dotenv has a chance
to load the .env file before the values are accessed.
"""

import os


class Settings:
    """Application settings loaded from environment variables."""

    @property
    def PRIMARY_PROVIDER(self) -> str:
        return os.getenv("PRIMARY_PROVIDER", "anthropic")

    @property
    def ANTHROPIC_MODEL(self) -> str:
        return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    @property
    def OPENAI_MODEL(self) -> str:
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def TEMPERATURE(self) -> float:
        return float(os.getenv("TEMPERATURE", "0.1"))

    @property
    def MAX_TOKENS(self) -> int:
        return int(os.getenv("MAX_TOKENS", "2048"))

    @property
    def MAX_RETRIES(self) -> int:
        return int(os.getenv("MAX_RETRIES", "2"))

    @property
    def REQUEST_TIMEOUT(self) -> int:
        return int(os.getenv("REQUEST_TIMEOUT", "25"))

    @property
    def OPENAI_API_KEY(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    @property
    def ANTHROPIC_API_KEY(self) -> str | None:
        return os.getenv("ANTHROPIC_API_KEY")


settings = Settings()
