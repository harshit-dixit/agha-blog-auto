from __future__ import annotations

import time
from collections.abc import Iterator

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config.settings import Settings
from src.db.history import HistoryDB
from src.publishers.oauth_helper import get_credentials

RATE_LIMIT_REASONS = {"ratelimitexceeded", "userratelimitexceeded", "quotaexceeded"}


class BloggerRateLimitError(RuntimeError):
    """Raised when Blogger rejects a write because the rate limit or quota is spent.

    Like Gemini's QuotaExhaustedError this is a run-level condition: once retries with
    backoff have been exhausted, every later write in the same run fails the same way, so
    callers should stop publishing rather than burn the rest of the backlog on doomed
    inserts. Subclasses RuntimeError so existing per-post handlers still treat it as a
    skip.
    """


def _is_rate_limit_error(exc: HttpError) -> bool:
    """Detect HTTP 429 / rate-limit rejections in a Blogger API error."""
    if getattr(exc.resp, "status", None) in (429, 403):
        # 403 is only a rate limit when the reason says so - it is otherwise a genuine
        # permission failure, which must keep surfacing as a hard error.
        if exc.resp.status == 403:
            details = getattr(exc, "error_details", None) or []
            reasons = {
                str(d.get("reason", "")).lower()
                for d in details
                if isinstance(d, dict)
            }
            return bool(reasons & RATE_LIMIT_REASONS)
        return True
    return False


class BloggerClient:
    """Thin wrapper around the Blogger API v3 for publishing and reconciling posts."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.blog_id = settings.require_blog_id()
        credentials = get_credentials(settings)
        self._service = build("blogger", "v3", credentials=credentials, cache_discovery=False)
        self._num_retries = max(0, settings.BLOGGER_MAX_RETRIES)
        self._min_write_interval = max(0.0, settings.BLOGGER_MIN_WRITE_INTERVAL)
        self._last_write_at: float | None = None

    def _throttle_writes(self) -> None:
        """Space consecutive writes out so a batch of publishes does not trip a burst limit.

        Blogger accepts a handful of back-to-back inserts and then 429s the rest, so the
        retry backoff alone is not enough - the requests have to arrive slower in the
        first place.
        """
        if self._min_write_interval <= 0:
            return
        if self._last_write_at is not None:
            elapsed = time.monotonic() - self._last_write_at
            if elapsed < self._min_write_interval:
                time.sleep(self._min_write_interval - elapsed)
        self._last_write_at = time.monotonic()

    def get_blog_info(self) -> dict:
        return self._service.blogs().get(blogId=self.blog_id).execute()

    def publish_article(
        self,
        *,
        title: str,
        content: str,
        labels: list[str],
        is_draft: bool = True,
    ) -> dict:
        body = {"kind": "blogger#post", "title": title, "content": content, "labels": labels}
        self._throttle_writes()
        try:
            return (
                self._service.posts()
                .insert(blogId=self.blog_id, body=body, isDraft=is_draft)
                .execute(num_retries=self._num_retries)
            )
        except HttpError as exc:
            message = f"Blogger publish failed: {exc}"
            if _is_rate_limit_error(exc):
                raise BloggerRateLimitError(message) from exc
            raise RuntimeError(message) from exc

    def update_article(
        self,
        post_id: str,
        *,
        title: str,
        content: str,
        labels: list[str],
    ) -> dict:
        """Update an existing post in Blogger (e.g. for version bumps)."""
        body = {"kind": "blogger#post", "id": post_id, "title": title, "content": content, "labels": labels}
        self._throttle_writes()
        try:
            return (
                self._service.posts()
                .patch(blogId=self.blog_id, postId=post_id, body=body)
                .execute(num_retries=self._num_retries)
            )
        except HttpError as exc:
            message = f"Blogger update failed for post {post_id}: {exc}"
            if _is_rate_limit_error(exc):
                raise BloggerRateLimitError(message) from exc
            raise RuntimeError(message) from exc

    def list_recent_posts(self, max_results: int = 10) -> list[dict]:
        response = (
            self._service.posts()
            .list(blogId=self.blog_id, maxResults=max_results, fetchBodies=False, status=["live", "draft"])
            .execute()
        )
        return response.get("items", [])

    def _iter_all_posts(self) -> Iterator[dict]:
        request = self._service.posts().list(
            blogId=self.blog_id, maxResults=500, fetchBodies=False, status=["live", "draft"]
        )
        while request is not None:
            response = request.execute()
            yield from response.get("items", [])
            request = self._service.posts().list_next(request, response)

    def reconcile_with_db(self, history_db: HistoryDB) -> int:
        """Backfill DB rows for posts that exist on Blogger but aren't tracked locally.

        Guards against a crash between a successful Blogger publish and the local
        history.db write (e.g. a killed CI job) leaving the ledger out of sync.
        """
        known_ids = history_db.get_known_blogger_post_ids()
        reconciled = 0
        for post in self._iter_all_posts():
            post_id = post.get("id")
            if post_id in known_ids:
                continue
            history_db.record_publication(
                topic_id=f"reconciled-{post_id}",
                title=post.get("title", "Untitled"),
                url=post.get("url"),
                blogger_post_id=post_id,
                status=post.get("status", "LIVE"),
                category="reconciled",
                word_count=0,
            )
            reconciled += 1
        return reconciled
