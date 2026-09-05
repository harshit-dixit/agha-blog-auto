from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "history.db"
TOPICS_FILE = PROJECT_ROOT / "config" / "topics.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    GEMINI_API_KEY: str | None = None
    # Flash-Lite rather than Flash: the free tier bills a per-day request count, and the
    # Lite tier's is 200/day against Flash's 20/day. Ten times the daily budget is worth
    # more here than the larger model's prose, since a spent quota defers whole runs.
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    # Posts per batched review request. Batching trades quota for blast radius: one reply
    # has to hold every review inside the output-token ceiling, and a reply that comes back
    # unusable defers every post in it. With 200 requests/day the quota side is no longer
    # tight - a full run costs single-digit requests at this size - so it stays low.
    GEMINI_BATCH_SIZE: int = 2

    BLOGGER_BLOG_ID: str | None = None
    # Blogger rejects rapid bursts of inserts with HTTP 429 well before any daily cap is
    # reached, so writes are both retried with backoff and spaced out. The retry count is
    # handed to googleapiclient, whose num_retries already backs off on 429/5xx.
    BLOGGER_MAX_RETRIES: int = 5
    BLOGGER_MIN_WRITE_INTERVAL: float = 3.0
    BLOGGER_CLIENT_ID: str | None = None
    BLOGGER_CLIENT_SECRET: str | None = None
    BLOGGER_REFRESH_TOKEN: str | None = None
    BLOGGER_CLIENT_SECRET_FILE: str = "client_secret.json"
    BLOGGER_TOKEN_FILE: str = "token.json"

    # Days a cached review is reused before it is dropped as stale. Prose is cached when a
    # post is generated but not published (a Blogger 429, say), so it only has to survive
    # long enough for the next few scheduled runs to retry the post.
    PROSE_CACHE_TTL_DAYS: int = 14

    DEFAULT_PUBLISH_STATUS: Literal["DRAFT", "LIVE"] = "DRAFT"
    MIN_WORD_COUNT: int = 1200

    @field_validator("DEFAULT_PUBLISH_STATUS", mode="before")
    @classmethod
    def _normalize_status(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @property
    def client_secret_path(self) -> Path:
        path = Path(self.BLOGGER_CLIENT_SECRET_FILE)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def token_path(self) -> Path:
        path = Path(self.BLOGGER_TOKEN_FILE)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def require_gemini_key(self) -> str:
        if not self.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file or environment."
            )
        return self.GEMINI_API_KEY

    def require_blog_id(self) -> str:
        if not self.BLOGGER_BLOG_ID:
            raise RuntimeError(
                "BLOGGER_BLOG_ID is not set. Add it to your .env file or environment."
            )
        return self.BLOGGER_BLOG_ID


@lru_cache
def get_settings() -> Settings:
    return Settings()
