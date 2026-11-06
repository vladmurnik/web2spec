"""Question-first research: task -> brief -> sources -> notes -> report.

The 4B analyst runs alone (the 0.5B collector is not loaded), reads semantically
selected chunks from 20-50 pages and writes extractive notes with exact quotes.
The final facts are assembled deterministically; conflicts stay visible.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from web2spec.collect import RobotsCache, fetch  # noqa: E402
from web2spec.llama_runtime import (  # noqa: E402
    DEFAULT_ANALYST,
    DEFAULT_DEVICE,
    LlamaServer,
)
from web2spec.research import (  # noqa: E402
    ResearchConfig,
    clamp_pages,
    run_research,
)

SEARCH_SCRIPT = REPO_ROOT / "scripts" / "22_search_and_collect.py"
_spec = importlib.util.spec_from_file_location("web2spec_search", SEARCH_SCRIPT)
assert _spec is not None and _spec.loader is not None
search_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(search_cli)


class Analyst:
    """Adapter around one llama-server process with call-level timing."""

    def __init__(self, server: LlamaServer) -> None:
        self.server = server
        self.calls = 0
        self.prompt_tokens = 0
        self.generated_tokens = 0
        self.generate_seconds = 0.0
        self.request_seconds = 0.0

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        started = time.perf_counter()
        text = self.server.generate(prompt, max_new_tokens=max_new_tokens)
        elapsed = time.perf_counter() - started
        timings = self.server.last_timings
        self.calls += 1
        self.request_seconds += elapsed
        self.prompt_tokens += int(timings.get("prompt_n") or 0)
        self.generated_tokens += int(timings.get("predicted_n") or 0)
        self.generate_seconds += float(timings.get("predicted_ms") or 0.0) / 1000.0
        return text

    def speed(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "generate_seconds": round(self.generate_seconds, 3),
            "request_seconds": round(self.request_seconds, 3),
            "tokens_per_second": round(
                self.generated_tokens / self.generate_seconds, 2
            )
            if self.generate_seconds
            else 0.0,
        }


def _domains(values: list[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        for part in value.split(","):
            domain = part.strip().casefold()
            if "//" in domain:
                domain = domain.split("//", 1)[1]
            domain = domain.split("/", 1)[0].split(":", 1)[0]
            if domain.startswith("www."):
                domain = domain[4:]
            if domain and domain not in result:
                result.append(domain)
    return result


def _read_urls(path: Path | None) -> list[str]:
    if path is None:
        return []
    text = path.read_text(encoding="utf-8-sig")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _local_html(html_dir: Path | None, url: str) -> Path | None:
    if html_dir is None:
        return None
    return html_dir / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.html"


def _fetch_fn(html_dir: Path | None, timeout: float, delay: float):
    robots = RobotsCache(timeout=min(timeout, 15.0))

    def load(url: str) -> tuple[str | None, str | None]:
        local = _local_html(html_dir, url)
        if local is not None:
            if local.is_file():
                return local.read_text(encoding="utf-8", errors="replace"), None
            return None, f"local fixture not found: {local.name}"
        if not robots.allowed(url):
            return None, "robots.txt disallows this URL or is unavailable"
        time.sleep(max(delay, robots.delay(url)))
        result = fetch(url, timeout=timeout)
        if result.error:
            return None, result.error
        if not result.html:
            return None, f"empty response (http {result.status})"
        return result.html, None

    return load


def _search_fn(provider: str, searxng_url: str, language: str, timeout: float):
    def search(query: str) -> tuple[list[dict[str, Any]], str | None]:
        if provider == "searxng":
            return search_cli.search_searxng(
                query,
                base_url=searxng_url,
                language=language,
                timeout=timeout,
            )
        return search_cli.search_ddg(query, timeout=timeout)

    return search


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Question-first research: 20-50 pages, exact evidence quotes"
    )
    parser.add_argument("--task", required=True, help="what to research")
    parser.add_argument("--min-pages", type=int, default=20)
    parser.add_argument("--max-pages", type=int, default=30)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--queries-per-round", type=int, default=6)
    parser.add_argument("--provider", choices=("ddg", "searxng"), default="ddg")
    parser.add_argument("--searxng-url", default="http://localhost:8080")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--domain", action="append", default=[], help="repeatable")
    parser.add_argument("--domains", default="", help="comma-separated domains")
    parser.add_argument("--source-hint", action="append", default=[])
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--urls", type=Path, default=None, help="file with URLs")
    parser.add_argument("--html-dir", type=Path, default=None, help="offline fixtures")
    parser.add_argument("--analyst", type=Path, default=DEFAULT_ANALYST)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--llama-server", default=None)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--fetch-timeout", type=float, default=20.0)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--chunk-chars", type=int, default=1400)
    parser.add_argument("--selected-chars", type=int, default=7000)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data/reports/research.json",
    )
    args = parser.parse_args(argv)

    min_pages, max_pages = clamp_pages(args.min_pages, args.max_pages)
    domains = _domains([*args.domain, args.domains])
    urls = [*args.url, *_read_urls(args.urls)]
    config = ResearchConfig(
        min_pages=min_pages,
        max_pages=max_pages,
        max_rounds=max(1, min(args.max_rounds, 4)),
        queries_per_round=max(1, min(args.queries_per_round, 10)),
        max_chunks=max(1, args.max_chunks),
        chunk_chars=max(300, args.chunk_chars),
        selected_chars=max(1000, args.selected_chars),
        fetch_timeout=args.fetch_timeout,
        delay=args.delay,
        domains=domains,
    )

    started = time.perf_counter()
    with LlamaServer(
        args.analyst,
        server=args.llama_server,
        device=args.device,
        context=args.context,
        startup_timeout=args.startup_timeout,
        request_timeout=args.request_timeout,
    ) as runtime:
        load_seconds = time.perf_counter() - started
        analyst = Analyst(runtime)
        report = run_research(
            args.task,
            llm=analyst,
            search_fn=_search_fn(
                args.provider, args.searxng_url, args.language, args.fetch_timeout
            ),
            fetch_fn=_fetch_fn(args.html_dir, args.fetch_timeout, args.delay),
            config=config,
            urls=urls,
            source_hints=[*args.source_hint, *domains],
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )

    report["speed"] = {
        **analyst.speed(),
        "load_seconds": round(load_seconds, 3),
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    report["config"] = {
        "min_pages": min_pages,
        "max_pages": max_pages,
        "max_rounds": config.max_rounds,
        "queries_per_round": config.queries_per_round,
        "domains": domains,
        "provider": args.provider,
        "analyst": str(args.analyst),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "goal": report["goal"],
        "entities": len(report["entities"]),
        "notes": len(report["notes"]),
        "sources": len(report["sources"]),
        "errors": len(report["errors"]),
        "gaps": len(report["gaps"]),
        "stats": report["stats"],
        "speed": report["speed"],
        "out": str(args.out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
