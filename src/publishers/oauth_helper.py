from __future__ import annotations

import contextlib
import json
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from config.settings import Settings

# Blogger API v3 requires only this single scope for reading and publishing posts.
SCOPES = ["https://www.googleapis.com/auth/blogger"]


def _load_from_token_file(token_path: Path) -> Credentials | None:
    if not token_path.exists():
        return None
    return Credentials.from_authorized_user_file(str(token_path), scopes=SCOPES)


def _load_from_env(settings: Settings) -> Credentials | None:
    if not (settings.BLOGGER_REFRESH_TOKEN and settings.BLOGGER_CLIENT_ID and settings.BLOGGER_CLIENT_SECRET):
        return None
    return Credentials(
        token=None,
        refresh_token=settings.BLOGGER_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.BLOGGER_CLIENT_ID,
        client_secret=settings.BLOGGER_CLIENT_SECRET,
        scopes=SCOPES,
    )


def _save_token(creds: Credentials, token_path: Path) -> None:
    token_path.write_text(creds.to_json(), encoding="utf-8")


def get_credentials(settings: Settings) -> Credentials:
    """Load Blogger credentials from token.json, falling back to env vars for CI.

    Local development uses the cached token.json produced by `auth`/interactive_login.
    GitHub Actions (and any environment without a local token file) instead supplies
    BLOGGER_REFRESH_TOKEN/BLOGGER_CLIENT_ID/BLOGGER_CLIENT_SECRET as secrets.
    """
    creds = _load_from_token_file(settings.token_path)

    if creds is None:
        creds = _load_from_env(settings)
        if creds is None:
            raise RuntimeError(
                "No credentials available. Run `python -m src.main auth` locally to log in, "
                "or set BLOGGER_REFRESH_TOKEN / BLOGGER_CLIENT_ID / BLOGGER_CLIENT_SECRET "
                "for CI environments."
            )

    if not creds.valid:
        if creds.refresh_token:
            creds.refresh(Request())
        else:
            raise RuntimeError(
                "Stored credentials are invalid and carry no refresh token. "
                "Run `python -m src.main auth --force` to re-authenticate."
            )

    # A read-only checkout (or a CI runner without a writable workspace) is fine here:
    # the refreshed credentials are still returned, just not cached to disk.
    with contextlib.suppress(OSError):
        _save_token(creds, settings.token_path)

    return creds


def interactive_login(settings: Settings) -> Credentials:
    """Run the local OAuth consent flow and cache the result to token.json."""
    if not settings.client_secret_path.exists():
        raise RuntimeError(
            f"Client secret file not found at {settings.client_secret_path}. "
            "Download it from Google Cloud Console (OAuth client, Desktop app type)."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(settings.client_secret_path), SCOPES)
    creds = flow.run_local_server(port=0)
    _save_token(creds, settings.token_path)
    return creds


def export_secrets(settings: Settings) -> dict[str, str]:
    """Collect the values needed to populate GitHub Actions secrets."""
    values: dict[str, str] = {}

    if settings.GEMINI_API_KEY:
        values["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
    if settings.BLOGGER_BLOG_ID:
        values["BLOGGER_BLOG_ID"] = settings.BLOGGER_BLOG_ID

    client_id = settings.BLOGGER_CLIENT_ID
    client_secret = settings.BLOGGER_CLIENT_SECRET
    if (not client_id or not client_secret) and settings.client_secret_path.exists():
        raw = json.loads(settings.client_secret_path.read_text(encoding="utf-8"))
        installed = raw.get("installed") or raw.get("web") or {}
        client_id = client_id or installed.get("client_id")
        client_secret = client_secret or installed.get("client_secret")
    if client_id:
        values["BLOGGER_CLIENT_ID"] = client_id
    if client_secret:
        values["BLOGGER_CLIENT_SECRET"] = client_secret

    refresh_token = settings.BLOGGER_REFRESH_TOKEN
    if not refresh_token and settings.token_path.exists():
        raw = json.loads(settings.token_path.read_text(encoding="utf-8"))
        refresh_token = raw.get("refresh_token")
    if refresh_token:
        values["BLOGGER_REFRESH_TOKEN"] = refresh_token

    return values


def mask_secret(value: str, visible: int = 4) -> str:
    if len(value) <= visible * 2:
        return "*" * len(value)
    return f"{value[:visible]}{'*' * (len(value) - visible * 2)}{value[-visible:]}"
