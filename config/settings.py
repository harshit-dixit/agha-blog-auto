from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

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

    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3.5-flash"
    # Posts per batched review request. The free tier bills a per-day request count, so
    # batching is what keeps a run affordable; keep this small enough that one reply fits
    # inside the output-token ceiling and a truncated response costs only a few posts.
    GEMINI_BATCH_SIZE: int = 5

    BLOGGER_BLOG_ID: Optional[str] = None
    BLOGGER_CLIENT_ID: Optional[str] = None
    BLOGGER_CLIENT_SECRET: Optional[str] = None
    BLOGGER_REFRESH_TOKEN: Optional[str] = None
    BLOGGER_CLIENT_SECRET_FILE: str = "client_secret.json"
    BLOGGER_TOKEN_FILE: str = "token.json"

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
