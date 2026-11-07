"""Small SQLite cache for fetched pages and search results.

The cache is deliberately opt-in at the CLI level: a stale price or policy is
worse than a slow run.  When enabled, every cached row has a timestamp and a
TTL, and the caller can always bypass it with ``--no-cache``.

Only successful page bodies and successful search payloads are stored.  Errors
are cheap to retry and caching them would hide transient outages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_CACHE_NAME = ".cache/web2spec.sqlite"
DEFAULT_TTL_HOURS = 24.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    url TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL,
    status INTEGER NOT NULL,
    content_type TEXT NOT NULL DEFAULT '',
    html TEXT NOT NULL,
    content_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS search_results (
    provider TEXT NOT NULL,
    query TEXT NOT NULL,
    language TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (provider, query, language)
);
"""


@dataclass(frozen=True, slots=True)
class CachedPage:
    url: str
    status: int
    html: str
    content_type: str
    content_hash: str
    fetched_at: str

    @property
    def ok(self) -> bool:
        return self.status == 200 and bool(self.html)


class PageCache:
    """Thread-safe, TTL-bound cache backed by one SQLite file."""

    def __init__(self, path: Path | str, *, ttl_seconds: float = DEFAULT_TTL_HOURS * 3600) -> None:
        if ttl_seconds <= 0:
            raise ValueError("cache TTL must be positive")
        self.path = Path(path)
        self.ttl_seconds = float(ttl_seconds)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA busy_timeout=5000")
            self._db.executescript(_SCHEMA)
            self._db.commit()
        self.stats: dict[str, int] = {
            "page_hits": 0,
            "page_misses": 0,
            "search_hits": 0,
            "search_misses": 0,
            "page_writes": 0,
            "search_writes": 0,
        }

    def __enter__(self) -> PageCache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def _fresh(self, fetched_at: str) -> bool:
        try:
            stamp = datetime.fromisoformat(fetched_at)
        except ValueError:
            return False
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return self._now() - stamp <= timedelta(seconds=self.ttl_seconds)

    @staticmethod
    def _hash(html: str) -> str:
        return hashlib.sha256(html.encode("utf-8")).hexdigest()

    def get_page(self, url: str) -> CachedPage | None:
        """Return a fresh cached page or ``None``; expired rows are removed."""

        with self._lock:
            row = self._db.execute(
                "SELECT url, status, html, content_type, content_hash, fetched_at "
                "FROM pages WHERE url = ?",
                (url,),
            ).fetchone()
            if row is None:
                self.stats["page_misses"] += 1
                return None
            if not self._fresh(row["fetched_at"]):
                self._db.execute("DELETE FROM pages WHERE url = ?", (url,))
                self._db.commit()
                self.stats["page_misses"] += 1
                return None
            self.stats["page_hits"] += 1
            return CachedPage(
                url=row["url"],
                status=int(row["status"]),
                html=row["html"],
                content_type=row["content_type"],
                content_hash=row["content_hash"],
                fetched_at=row["fetched_at"],
            )

    def put_page(
        self,
        url: str,
        *,
        status: int,
        html: str,
        content_type: str = "",
    ) -> str | None:
        """Store a successful page and return its content hash."""

        if status != 200 or not html:
            return None
        digest = self._hash(html)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO pages "
                "(url, fetched_at, status, content_type, html, content_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (url, self._now().isoformat(), status, content_type, html, digest),
            )
            self._db.commit()
            self.stats["page_writes"] += 1
        return digest

    def get_search(self, provider: str, query: str, language: str) -> list[dict[str, Any]] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT fetched_at, payload FROM search_results "
                "WHERE provider = ? AND query = ? AND language = ?",
                (provider, query, language),
            ).fetchone()
            if row is None:
                self.stats["search_misses"] += 1
                return None
            if not self._fresh(row["fetched_at"]):
                self._db.execute(
                    "DELETE FROM search_results "
                    "WHERE provider = ? AND query = ? AND language = ?",
                    (provider, query, language),
                )
                self._db.commit()
                self.stats["search_misses"] += 1
                return None
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                payload = None
            if not isinstance(payload, list):
                self.stats["search_misses"] += 1
                return None
            self.stats["search_hits"] += 1
            return payload

    def put_search(
        self,
        provider: str,
        query: str,
        language: str,
        results: list[dict[str, Any]],
    ) -> None:
        if not results:
            return
        payload = json.dumps(results, ensure_ascii=False)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO search_results "
                "(provider, query, language, fetched_at, payload) VALUES (?, ?, ?, ?, ?)",
                (provider, query, language, self._now().isoformat(), payload),
            )
            self._db.commit()
            self.stats["search_writes"] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "ttl_hours": round(self.ttl_seconds / 3600, 3),
            **self.stats,
        }


def open_cache(
    path: Path | str,
    *,
    ttl_hours: float = DEFAULT_TTL_HOURS,
    disabled: bool = False,
) -> PageCache | None:
    if disabled or ttl_hours <= 0:
        return None
    return PageCache(path, ttl_seconds=ttl_hours * 3600)


def add_cache_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_path: Path,
) -> None:
    parser.add_argument(
        "--cache",
        type=Path,
        default=default_path,
        help=f"SQLite cache path (default: {default_path})",
    )
    parser.add_argument(
        "--cache-ttl",
        type=float,
        default=DEFAULT_TTL_HOURS,
        metavar="HOURS",
        help="cache freshness in hours; 0 disables the cache (default: 24)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="do not read or write the cache",
    )


def cache_from_args(args: argparse.Namespace) -> PageCache | None:
    return open_cache(
        args.cache,
        ttl_hours=args.cache_ttl,
        disabled=bool(getattr(args, "no_cache", False)),
    )
