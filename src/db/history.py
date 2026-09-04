from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from src.catalog import Topic, TopicCatalog

SCHEMA = """
CREATE TABLE IF NOT EXISTS published_posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL UNIQUE,
    url TEXT,
    blogger_post_id TEXT,
    published_at TEXT NOT NULL,
    status TEXT NOT NULL,
    category TEXT NOT NULL,
    word_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_published_posts_category ON published_posts(category);
"""


@dataclass
class PublishedPost:
    id: int
    topic_id: str
    title: str
    url: Optional[str]
    blogger_post_id: Optional[str]
    published_at: str
    status: str
    category: str
    word_count: int


class HistoryDB:
    """SQLite-backed publication ledger used for duplicate detection and topic rotation."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def is_topic_published(self, topic_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM published_posts WHERE topic_id = ?", (topic_id,)
            ).fetchone()
        return row is not None

    def is_title_published(self, title: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM published_posts WHERE title = ?", (title,)
            ).fetchone()
        return row is not None

    def record_publication(
        self,
        *,
        topic_id: str,
        title: str,
        url: Optional[str],
        blogger_post_id: Optional[str],
        status: str,
        category: str,
        word_count: int,
    ) -> int:
        published_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO published_posts
                    (topic_id, title, url, blogger_post_id, published_at, status, category, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (topic_id, title, url, blogger_post_id, published_at, status, category, word_count),
            )
            return cursor.lastrowid

    def get_published_topic_ids(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT topic_id FROM published_posts").fetchall()
        return {row["topic_id"] for row in rows}

    def get_known_blogger_post_ids(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT blogger_post_id FROM published_posts WHERE blogger_post_id IS NOT NULL"
            ).fetchall()
        return {row["blogger_post_id"] for row in rows}

    def get_category_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT category, COUNT(*) AS n FROM published_posts GROUP BY category"
            ).fetchall()
        return {row["category"]: row["n"] for row in rows}

    def list_recent(self, limit: int = 10) -> list[PublishedPost]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM published_posts ORDER BY published_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [PublishedPost(**dict(row)) for row in rows]

    def get_next_topic(
        self,
        catalog: TopicCatalog,
        *,
        category_id: Optional[str] = None,
        topic_id: Optional[str] = None,
    ) -> tuple[str, Topic]:
        """Pick the next unpublished topic, balancing rotation across categories."""
        published_ids = self.get_published_topic_ids()

        if topic_id:
            found = catalog.get_topic(topic_id)
            if found is None:
                raise ValueError(f"Unknown topic id: {topic_id!r}")
            found_category_id, topic = found
            if topic_id in published_ids:
                raise ValueError(f"Topic {topic_id!r} has already been published")
            if category_id and category_id != found_category_id:
                raise ValueError(
                    f"Topic {topic_id!r} belongs to category {found_category_id!r}, not {category_id!r}"
                )
            return found_category_id, topic

        candidate_category_ids = [category_id] if category_id else list(catalog.categories.keys())
        for cid in candidate_category_ids:
            if cid not in catalog.categories:
                raise ValueError(f"Unknown category id: {cid!r}")

        # Rank candidate categories by how few posts they've published so far,
        # so the rotation stays balanced instead of draining one category first.
        counts = self.get_category_counts()
        ranked = sorted(candidate_category_ids, key=lambda cid: counts.get(cid, 0))

        for cid in ranked:
            for topic in catalog.categories[cid].topics:
                if topic.id not in published_ids:
                    return cid, topic

        scope = f"category {category_id!r}" if category_id else "the catalog"
        raise ValueError(f"No unpublished topics remaining in {scope}")

    def get_stats(self, catalog: TopicCatalog) -> dict[str, dict[str, int]]:
        counts = self.get_category_counts()
        stats: dict[str, dict[str, int]] = {}
        for cid, category in catalog.categories.items():
            total = len(category.topics)
            published = counts.get(cid, 0)
            stats[cid] = {
                "total": total,
                "published": published,
                "remaining": total - published,
            }
        return stats
