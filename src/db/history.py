from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

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

CREATE TABLE IF NOT EXISTS an1_posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL,
    title TEXT NOT NULL,
    direct_download_url TEXT,
    dw_page_url TEXT,
    blogger_post_id TEXT,
    blogger_url TEXT,
    published_at TEXT NOT NULL,
    status TEXT NOT NULL,
    UNIQUE(post_id, version)
);

CREATE TABLE IF NOT EXISTS an1_prose_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '',
    prose_html TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(post_id, version)
);
CREATE INDEX IF NOT EXISTS idx_an1_prose_lookup ON an1_prose_cache(post_id, version);
"""


def _parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp for TTL comparison.

    An unparseable or naive value is treated as expired rather than kept forever - the
    only cost of dropping a cache entry is regenerating its prose.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass
class PublishedPost:
    id: int
    topic_id: str
    title: str
    url: str | None
    blogger_post_id: str | None
    published_at: str
    status: str
    category: str
    word_count: int


@dataclass
class AN1PublishedPost:
    id: int
    post_id: str
    version: str
    source_url: str
    title: str
    direct_download_url: str | None
    dw_page_url: str | None
    blogger_post_id: str | None
    blogger_url: str | None
    published_at: str
    status: str


class HistoryDB:
    """SQLite-backed publication ledger used for duplicate detection and topic rotation."""

    def __init__(
        self,
        db_path: Path,
        json_tracker_path: Path | None = None,
        prose_cache_path: Path | None = None,
        prose_cache_ttl_days: int = 14,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_tracker_path = json_tracker_path or (self.db_path.parent / "an1_published.json")
        self.prose_cache_path = prose_cache_path or (self.db_path.parent / "an1_prose_cache.json")
        self.prose_cache_ttl_days = prose_cache_ttl_days
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            # Automatic migration for existing DBs
            cursor = conn.execute("PRAGMA table_info(an1_posts)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "version" not in columns:
                conn.execute("ALTER TABLE an1_posts ADD COLUMN version TEXT NOT NULL DEFAULT ''")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_an1_posts_lookup ON an1_posts(post_id, version)")
        self._sync_from_json_tracker()
        self._sync_from_prose_cache()

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
        url: str | None,
        blogger_post_id: str | None,
        status: str,
        category: str,
        word_count: int,
    ) -> int:
        published_at = datetime.now(UTC).isoformat()
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
        category_id: str | None = None,
        topic_id: str | None = None,
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

    # -------------------------------------------------------------------------
    # AN1.com Post Tracking & Ledger Methods
    # -------------------------------------------------------------------------

    def _sync_from_json_tracker(self) -> None:
        """Seed SQLite an1_posts table from an1_published.json if table is empty or missing entries."""
        if not self.json_tracker_path.exists():
            return
        try:
            content = self.json_tracker_path.read_text(encoding="utf-8")
            if not content.strip():
                return
            data = json.loads(content)
            if not isinstance(data, list):
                raise ValueError(f"Expected list in {self.json_tracker_path}, got {type(data).__name__}")
        except Exception as exc:
            raise RuntimeError(
                f"Corrupt or unreadable AN1 publication ledger at {self.json_tracker_path}: {exc}"
            ) from exc

        with self._connect() as conn:
            for item in data:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO an1_posts
                        (post_id, version, source_url, title, direct_download_url,
                         dw_page_url, blogger_post_id, blogger_url, published_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.get("post_id"),
                        item.get("version", ""),
                        item.get("source_url"),
                        item.get("title", ""),
                        item.get("direct_download_url"),
                        item.get("dw_page_url"),
                        item.get("blogger_post_id"),
                        item.get("blogger_url"),
                        item.get("published_at", datetime.now(UTC).isoformat()),
                        item.get("status", "LIVE"),
                    ),
                )

    def _save_to_json_tracker(self) -> None:
        """Export current AN1 publications to an1_published.json for persistent Git tracking.

        Reads the ledger unbounded: in CI the SQLite file is disposable and rebuilt from
        this JSON each run, so any row missing here is a post that gets scraped, generated
        and published a second time.
        """
        posts = self.list_an1_posts(limit=None)
        serialized = [
            {
                "post_id": p.post_id,
                "version": p.version,
                "source_url": p.source_url,
                "title": p.title,
                "direct_download_url": p.direct_download_url,
                "dw_page_url": p.dw_page_url,
                "blogger_post_id": p.blogger_post_id,
                "blogger_url": p.blogger_url,
                "published_at": p.published_at,
                "status": p.status,
            }
            for p in posts
        ]
        self.json_tracker_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_tracker_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Generated-prose cache
    #
    # Gemini's free tier bills a per-day request count, so prose that was generated but
    # never published - a post that hit a Blogger 429, say - must not be regenerated on
    # the next run. Cached entries are keyed by (post_id, version) and dropped once the
    # post publishes, so the cache only ever holds the in-flight backlog.
    # ------------------------------------------------------------------

    def _sync_from_prose_cache(self) -> None:
        """Seed the prose cache table from its JSON file, skipping stale entries.

        Like the publication ledger, the SQLite file is disposable in CI and rebuilt from
        JSON each run. A corrupt cache is recoverable (worst case the prose is generated
        again), so it is dropped with a warning rather than failing the run.
        """
        if not self.prose_cache_path.exists():
            return
        try:
            content = self.prose_cache_path.read_text(encoding="utf-8")
            if not content.strip():
                return
            data = json.loads(content)
            if not isinstance(data, list):
                raise ValueError(f"Expected list, got {type(data).__name__}")
        except Exception:
            return

        cutoff = datetime.now(UTC) - timedelta(days=self.prose_cache_ttl_days)
        with self._connect() as conn:
            for item in data:
                if not isinstance(item, dict):
                    continue
                post_id = item.get("post_id")
                prose_html = item.get("prose_html")
                created_at = item.get("created_at") or datetime.now(UTC).isoformat()
                if not post_id or not prose_html:
                    continue
                if _parse_timestamp(created_at) < cutoff:
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO an1_prose_cache
                        (post_id, version, prose_html, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (post_id, item.get("version", ""), prose_html, created_at),
                )

    def _save_prose_cache(self) -> None:
        """Export the prose cache to JSON so it survives the next CI run."""
        cutoff = (datetime.now(UTC) - timedelta(days=self.prose_cache_ttl_days)).isoformat()
        with self._connect() as conn:
            conn.execute("DELETE FROM an1_prose_cache WHERE created_at < ?", (cutoff,))
            rows = conn.execute(
                "SELECT post_id, version, prose_html, created_at FROM an1_prose_cache "
                "ORDER BY created_at"
            ).fetchall()
        serialized = [
            {
                "post_id": r["post_id"],
                "version": r["version"],
                "prose_html": r["prose_html"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
        self.prose_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.prose_cache_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    def get_cached_prose(self, post_id: str, version: str = "") -> str | None:
        """Return previously generated prose for this exact (post_id, version), if any."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT prose_html FROM an1_prose_cache WHERE post_id = ? AND version = ?",
                (post_id, version or ""),
            ).fetchone()
        return row["prose_html"] if row else None

    def cache_prose(self, post_id: str, version: str, prose_html: str) -> None:
        """Store generated prose so a failed publish does not cost a second Gemini call."""
        if not prose_html:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO an1_prose_cache (post_id, version, prose_html, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(post_id, version) DO UPDATE SET
                    prose_html = excluded.prose_html,
                    created_at = excluded.created_at
                """,
                (post_id, version or "", prose_html, datetime.now(UTC).isoformat()),
            )
        self._save_prose_cache()

    def cache_prose_many(self, entries: Sequence[tuple[str, str, str]]) -> None:
        """Cache a whole batch of (post_id, version, prose_html) in one JSON rewrite."""
        rows = [(pid, ver or "", html) for pid, ver, html in entries if html]
        if not rows:
            return
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO an1_prose_cache (post_id, version, prose_html, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(post_id, version) DO UPDATE SET
                    prose_html = excluded.prose_html,
                    created_at = excluded.created_at
                """,
                [(pid, ver, html, now) for pid, ver, html in rows],
            )
        self._save_prose_cache()

    def discard_cached_prose(self, post_id: str, version: str = "") -> None:
        """Drop a cache entry once its post is published and the prose is no longer needed."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM an1_prose_cache WHERE post_id = ? AND version = ?",
                (post_id, version or ""),
            )
        self._save_prose_cache()

    def is_an1_published(self, post_id: str, version: str | None = None) -> bool:
        """Check if an AN1 post ID (or specific version) has already been published."""
        with self._connect() as conn:
            if version is not None:
                row = conn.execute(
                    "SELECT 1 FROM an1_posts WHERE post_id = ? AND version = ?", (post_id, version)
                ).fetchone()
            else:
                row = conn.execute("SELECT 1 FROM an1_posts WHERE post_id = ?", (post_id,)).fetchone()
        return row is not None

    def get_existing_blogger_post(self, post_id: str) -> AN1PublishedPost | None:
        """Return the most recent publication record for an AN1 post ID, if one exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM an1_posts WHERE post_id = ? ORDER BY id DESC LIMIT 1",
                (post_id,),
            ).fetchone()
        return AN1PublishedPost(**dict(row)) if row else None

    def get_published_an1_keys(self) -> set[tuple[str, str]]:
        """Return set of all published (post_id, version) pairs."""
        with self._connect() as conn:
            rows = conn.execute("SELECT post_id, version FROM an1_posts").fetchall()
        return {(row["post_id"], row["version"]) for row in rows}

    def record_an1_publication(
        self,
        *,
        post_id: str,
        version: str,
        source_url: str,
        title: str,
        direct_download_url: str | None = None,
        dw_page_url: str | None = None,
        blogger_post_id: str | None = None,
        blogger_url: str | None = None,
        status: str = "LIVE",
    ) -> int:
        """Record an AN1 published article and sync to JSON tracker file."""
        published_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO an1_posts
                    (post_id, version, source_url, title, direct_download_url,
                     dw_page_url, blogger_post_id, blogger_url, published_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    post_id, version, source_url, title, direct_download_url,
                    dw_page_url, blogger_post_id, blogger_url, published_at, status,
                ),
            )
            row_id = cursor.lastrowid or 0
        self._save_to_json_tracker()
        return row_id

    def list_an1_posts(self, limit: int | None = 500) -> list[AN1PublishedPost]:
        """List published AN1 posts in reverse chronological order.

        Pass limit=None for the complete ledger. Anything that rewrites the JSON tracker
        must do so, because a truncated read would drop the oldest rows from the file and
        those posts would look unpublished on the next run.
        """
        query = "SELECT * FROM an1_posts ORDER BY published_at DESC"
        params: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [AN1PublishedPost(**dict(row)) for row in rows]
