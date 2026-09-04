from __future__ import annotations

from typing import Iterator

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config.settings import Settings
from src.db.history import HistoryDB
from src.publishers.oauth_helper import get_credentials


class BloggerClient:
    """Thin wrapper around the Blogger API v3 for publishing and reconciling posts."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.blog_id = settings.require_blog_id()
        credentials = get_credentials(settings)
        self._service = build("blogger", "v3", credentials=credentials, cache_discovery=False)

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
        try:
            return (
                self._service.posts()
                .insert(blogId=self.blog_id, body=body, isDraft=is_draft)
                .execute()
            )
        except HttpError as exc:
            raise RuntimeError(f"Blogger publish failed: {exc}") from exc

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
        try:
            return (
                self._service.posts()
                .patch(blogId=self.blog_id, postId=post_id, body=body)
                .execute()
            )
        except HttpError as exc:
            raise RuntimeError(f"Blogger update failed for post {post_id}: {exc}") from exc

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
