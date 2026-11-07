from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from web2spec.cache import PageCache, add_cache_arguments, cache_from_args, open_cache


def test_page_cache_roundtrip(tmp_path) -> None:
    with PageCache(tmp_path / "cache.sqlite", ttl_seconds=3600) as cache:
        digest = cache.put_page(
            "https://example.com/page",
            status=200,
            html="<html><body>hello</body></html>",
            content_type="text/html",
        )
        assert digest is not None
        cached = cache.get_page("https://example.com/page")
        assert cached is not None
        assert cached.html == "<html><body>hello</body></html>"
        assert cached.content_hash == digest
        assert cache.get_page("https://example.com/missing") is None
        assert cache.stats["page_hits"] == 1
        assert cache.stats["page_misses"] == 1


def test_page_cache_ttl_expires(tmp_path, monkeypatch) -> None:
    cache = PageCache(tmp_path / "cache.sqlite", ttl_seconds=60)
    cache.put_page("https://example.com/old", status=200, html="<html>old</html>")
    later = datetime.now(UTC) + timedelta(minutes=2)
    monkeypatch.setattr(cache, "_now", lambda: later)
    assert cache.get_page("https://example.com/old") is None
    cache.close()


def test_search_cache_roundtrip(tmp_path) -> None:
    with PageCache(tmp_path / "cache.sqlite", ttl_seconds=3600) as cache:
        results = [
            {"title": "Example", "url": "https://example.com", "snippet": "hi"},
        ]
        cache.put_search("ddg", "example query", "en", results)
        assert cache.get_search("ddg", "example query", "en") == results
        assert cache.get_search("ddg", "other query", "en") is None
        assert cache.stats["search_hits"] == 1
        assert cache.stats["search_misses"] == 1


def test_cache_ignores_errors_and_empty_pages(tmp_path) -> None:
    with PageCache(tmp_path / "cache.sqlite", ttl_seconds=3600) as cache:
        assert cache.put_page("https://example.com/404", status=404, html="nope") is None
        assert cache.put_page("https://example.com/empty", status=200, html="") is None
        assert cache.get_page("https://example.com/404") is None


def test_open_cache_respects_disable_flags(tmp_path) -> None:
    assert open_cache(tmp_path / "cache.sqlite", disabled=True) is None
    assert open_cache(tmp_path / "cache.sqlite", ttl_hours=0) is None
    cache = open_cache(tmp_path / "cache.sqlite", ttl_hours=1)
    assert cache is not None
    cache.close()


def test_cache_arguments_defaults(tmp_path) -> None:
    parser = argparse.ArgumentParser()
    add_cache_arguments(parser, default_path=tmp_path / "cache.sqlite")
    args = parser.parse_args([])
    assert args.cache == tmp_path / "cache.sqlite"
    assert args.cache_ttl == 24.0
    assert cache_from_args(args) is not None

    args = parser.parse_args(["--no-cache"])
    assert cache_from_args(args) is None
