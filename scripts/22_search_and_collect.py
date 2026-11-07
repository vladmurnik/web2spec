"""Task -> search queries -> local web search -> task+URLs collector.

The pipeline has three stages:

1. A *local* LLM (optional) turns the task into 3-5 search queries.  It can be
   Ollama (default) or any OpenAI-compatible server.  Without a model the task
   text itself is used, so the script still works.
2. A *local* metasearch provider finds candidate links without paid APIs:
   SearXNG JSON (recommended; self-hosted) or DuckDuckGo HTML (no key, best
   effort).
3. The existing ``scripts/21_collect_urls.py`` cascade extracts facts from the
   best links and returns one JSON with no HTML.

No paid search API is called.  SearXNG itself queries public engines; DDG is a
public HTML endpoint.  Both can rate-limit or block, so this is best effort.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from web2spec.cache import (  # noqa: E402
    PageCache,
    add_cache_arguments,
    cache_from_args,
)
from web2spec.llama_runtime import (  # noqa: E402, F401
    DEFAULT_ANALYST,
    DEFAULT_DEVICE,
    DEFAULT_EXTRACTOR,
    DEFAULT_PLANNER,
)

COLLECTOR = REPO_ROOT / "scripts" / "21_collect_urls.py"
USER_AGENT = "web2spec-search/0.4 (local research; contact: local)"
MAX_QUERIES = 6

_QUERY_PROMPT = """Ты поисковый ассистент. По задаче пользователя составь от 3 до 5
коротких поисковых запросов, которые помогут найти страницы с нужными данными.
Верни ТОЛЬКО JSON вида {{"queries": ["...", "..."]}}.
Не добавляй объяснения. Язык запросов — язык задачи.
Задача: {task}
"""


def _http(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> tuple[str | None, str | None]:
    request = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, str(exc)
    return raw.decode("utf-8", errors="replace"), None


def _http_json(
    url: str,
    *,
    data: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> tuple[Any | None, str | None]:
    body = None
    headers = {"Content-Type": "application/json"}
    if data is not None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    text, error = _http(url, data=body, headers=headers, timeout=timeout)
    if error is not None or text is None:
        return None, error
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON from {url}: {exc}"


def normalize_url(url: str) -> str:
    """Canonical URL for deduplication; tracking parameters are removed."""

    parts = urllib.parse.urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return ""
    query = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    path = parts.path.rstrip("/") or "/"
    return urllib.parse.urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, urllib.parse.urlencode(query), "")
    )


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[\w-]+", text.casefold()) if len(token) > 2}


def deterministic_queries(task: str) -> list[str]:
    """Fallback when no local LLM is configured: the task is already a query."""

    task = " ".join(task.split())
    queries = [task]
    # A short keyword variant helps engines when the task is a long sentence.
    keywords = [token for token in re.findall(r"[\w-]+", task) if len(token) > 3]
    if len(keywords) >= 3:
        short = " ".join(keywords[:8])
        if short.casefold() != task.casefold():
            queries.append(short)
    return queries


def _parse_queries(raw: str) -> list[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = payload.get("queries", [])
    if not isinstance(payload, list):
        return []
    result: list[str] = []
    for item in payload:
        if isinstance(item, str) and item.strip():
            result.append(" ".join(item.split()))
    return result[:MAX_QUERIES]


def llm_queries(
    task: str,
    *,
    model: str,
    base_url: str,
    provider: str = "ollama",
    timeout: float = 60.0,
) -> tuple[list[str], str | None]:
    """Ask a local LLM for queries; return (queries, error)."""

    prompt = _QUERY_PROMPT.format(task=task)
    if provider == "openai":
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        response, error = _http_json(url, data=payload, timeout=timeout)
        if error is not None:
            return [], error
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return [], "unexpected OpenAI-compatible response"
        return _parse_queries(content), None

    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2, "num_predict": 256},
    }
    response, error = _http_json(url, data=payload, timeout=timeout)
    if error is not None:
        return [], error
    if not isinstance(response, dict) or not isinstance(response.get("response"), str):
        return [], "unexpected Ollama response"
    return _parse_queries(response["response"]), None


def _decode_ddg_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parts = urllib.parse.urlsplit(href)
    if "duckduckgo.com" in parts.netloc and parts.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parts.query).get("uddg", [""])[0]
        if target:
            return target
    return href


def search_ddg(query: str, *, timeout: float = 20.0) -> tuple[list[dict[str, Any]], str | None]:
    """DuckDuckGo HTML endpoint, no API key.  Best effort, may rate-limit."""

    from bs4 import BeautifulSoup

    body = urllib.parse.urlencode({"q": query}).encode("utf-8")
    text, error = _http(
        "https://html.duckduckgo.com/html/",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )
    if error is not None or text is None:
        return [], error
    soup = BeautifulSoup(text, "lxml")
    results: list[dict[str, Any]] = []
    for anchor in soup.select("a.result__a"):
        href = _decode_ddg_url(anchor.get("href") or "")
        if not href:
            continue
        parent = anchor.find_parent("div", class_="result")
        snippet_el = parent.select_one(".result__snippet") if parent else None
        results.append(
            {
                "url": href,
                "title": anchor.get_text(" ", strip=True),
                "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else "",
            }
        )
    return results, None


def search_searxng(
    query: str,
    *,
    base_url: str,
    language: str = "ru",
    timeout: float = 20.0,
) -> tuple[list[dict[str, Any]], str | None]:
    """Query a local SearXNG instance via its JSON API."""

    params = urllib.parse.urlencode(
        {"q": query, "format": "json", "language": language, "safesearch": 1}
    )
    url = f"{base_url.rstrip('/')}/search?{params}"
    payload, error = _http_json(url, timeout=timeout)
    if error is not None:
        return [], error
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return [], "SearXNG returned no results list (is JSON format enabled?)"
    results: list[dict[str, Any]] = []
    for item in payload["results"]:
        if not isinstance(item, dict) or not isinstance(item.get("url"), str):
            continue
        results.append(
            {
                "url": item["url"],
                "title": str(item.get("title") or ""),
                "snippet": str(item.get("content") or ""),
            }
        )
    return results, None


def _clean_domains(domains: list[str] | None) -> list[str]:
    cleaned: list[str] = []
    for item in domains or []:
        for chunk in str(item).split(","):
            text = chunk.strip().casefold()
            if not text:
                continue
            if "://" in text:
                text = urllib.parse.urlsplit(text).netloc
            text = text.split("/", 1)[0].lstrip(".")
            if text and text not in cleaned:
                cleaned.append(text)
    return cleaned


def _host_matches(url: str, domains: list[str]) -> bool:
    host = urllib.parse.urlsplit(url).netloc.casefold().split(":", 1)[0]
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def rank_candidates(
    task: str,
    per_query: list[tuple[str, list[dict[str, Any]]]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Merge results from several queries and rank with RRF + token overlap."""

    task_tokens = _tokens(task)
    merged: dict[str, dict[str, Any]] = {}
    for query, results in per_query:
        for rank, item in enumerate(results, 1):
            url = normalize_url(str(item.get("url") or ""))
            if not url:
                continue
            row = merged.setdefault(
                url,
                {
                    "url": url,
                    "title": str(item.get("title") or ""),
                    "snippet": str(item.get("snippet") or ""),
                    "queries": [],
                    "rrf": 0.0,
                },
            )
            row["rrf"] += 1.0 / (rank + 10)
            if query not in row["queries"]:
                row["queries"].append(query)
            if not row["title"] and item.get("title"):
                row["title"] = str(item["title"])
            if not row["snippet"] and item.get("snippet"):
                row["snippet"] = str(item["snippet"])
    for row in merged.values():
        text = f"{row['title']} {row['snippet']}".casefold()
        overlap = len(task_tokens & _tokens(text)) / (len(task_tokens) or 1)
        row["score"] = round(row["rrf"] + 0.5 * overlap, 6)
    ranked = sorted(merged.values(), key=lambda row: (-row["score"], row["url"]))
    return ranked[:limit]


def _clean_task(value: str) -> str:
    return " ".join(value.split())


def search_urls(
    task: str,
    *,
    provider: str,
    searxng_url: str,
    language: str,
    max_results: int,
    llm_model: str | None = None,
    llm_url: str = "http://localhost:11434",
    llm_provider: str = "ollama",
    llm_timeout: float = 60.0,
    timeout: float = 20.0,
    search_only: bool = False,
    domains: list[str] | None = None,
    cache: PageCache | None = None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Run query generation + search; return (search_block, urls, warnings)."""

    warnings: list[str] = []
    domain_filter = _clean_domains(domains)
    query_source = "deterministic"
    queries: list[str] = []
    if llm_model:
        queries, error = llm_queries(
            task,
            model=llm_model,
            base_url=llm_url,
            provider=llm_provider,
            timeout=llm_timeout,
        )
        if queries:
            query_source = f"{llm_provider}:{llm_model}"
        elif error is not None:
            warnings.append(f"local LLM unavailable, using deterministic queries: {error}")
    if not queries:
        queries = deterministic_queries(task)
    if domain_filter:
        # Site-scoped queries are much more reliable than filtering generic
        # search results after the fact.  Keep the original queries too, then
        # deduplicate and cap the total number of provider requests.
        scoped = [f"{task} site:{domain}" for domain in domain_filter[:3]]
        queries = list(dict.fromkeys(scoped + queries))[:8]

    search_fn = search_searxng if provider == "searxng" else search_ddg
    cache_provider = provider if provider == "ddg" else f"searxng:{searxng_url}"
    cached_queries = 0
    per_query: list[tuple[str, list[dict[str, Any]]]] = []
    for query in queries:
        results = cache.get_search(cache_provider, query, language) if cache else None
        if results is not None:
            cached_queries += 1
        else:
            if provider == "searxng":
                results, error = search_fn(
                    query, base_url=searxng_url, language=language, timeout=timeout
                )
            else:
                results, error = search_fn(query, timeout=timeout)
            if error is not None:
                warnings.append(f"search failed for {query!r}: {error}")
                results = []
            elif cache is not None:
                cache.put_search(cache_provider, query, language, results)
        per_query.append((query, results))

    candidates = rank_candidates(
        task, per_query, limit=max_results * 3 if domain_filter else max_results
    )
    if domain_filter:
        filtered = [
            row for row in candidates if _host_matches(row["url"], domain_filter)
        ]
        if not filtered:
            warnings.append(
                "no search results matched domains: " + ", ".join(domain_filter)
            )
        candidates = filtered[:max_results]
    urls = [row["url"] for row in candidates]
    block = {
        "provider": provider,
        "query_source": query_source,
        "queries": queries,
        "candidates": candidates,
        "count": len(candidates),
        "search_only": search_only,
        "domains": domain_filter,
        "cached_queries": cached_queries,
        "warnings": warnings,
    }
    return block, urls, warnings


def _collector_argv(args: argparse.Namespace, urls: list[str], out: Path) -> list[str]:
    argv = [
        sys.executable,
        str(COLLECTOR),
        "--task",
        args.task,
        "--out",
        str(out),
        "--max-pages",
        str(min(args.max_pages, len(urls))),
        "--max-chars",
        str(args.max_chars),
        "--timeout",
        str(args.timeout),
        "--delay",
        str(args.delay),
        "--planner",
        args.planner,
        "--extractor",
        args.extractor,
        "--backend",
        getattr(args, "backend", "auto"),
        "--plan-max-new-tokens",
        str(args.plan_max_new_tokens),
        "--fact-max-new-tokens",
        str(args.fact_max_new_tokens),
        "--allow-partial" if getattr(args, "allow_partial", True) else "--no-allow-partial",
    ]
    device = getattr(args, "device", None)
    if device:
        argv.extend(["--device", device])
    llama_server = getattr(args, "llama_server", None)
    if llama_server:
        argv.extend(["--llama-server", llama_server])
    context = getattr(args, "context", None)
    if context:
        argv.extend(["--context", str(context)])
    startup_timeout = getattr(args, "startup_timeout", None)
    if startup_timeout:
        argv.extend(["--startup-timeout", str(startup_timeout)])
    request_timeout = getattr(args, "request_timeout", None)
    if request_timeout:
        argv.extend(["--request-timeout", str(request_timeout)])
    for url in urls:
        argv.extend(["--url", url])
    if args.html_dir is not None:
        argv.extend(["--html-dir", str(args.html_dir)])
    cache_path = getattr(args, "cache", None)
    if getattr(args, "no_cache", False):
        argv.append("--no-cache")
    elif cache_path is not None:
        argv.extend(
            [
                "--cache",
                str(cache_path),
                "--cache-ttl",
                str(getattr(args, "cache_ttl", 24.0)),
            ]
        )
    if getattr(args, "previous", None) is not None:
        argv.extend(["--previous", str(args.previous)])
    for field in getattr(args, "field", []) or []:
        argv.extend(["--field", str(field)])
    return argv


def _run_collector(
    args: argparse.Namespace, urls: list[str]
) -> tuple[dict[str, Any] | None, str | None]:
    """Run the existing task+URLs collector in a child process."""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="search-collect-", suffix=".json", dir=args.out.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
    try:
        completed = subprocess.run(
            _collector_argv(args, urls, temp_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.collector_timeout,
            check=False,
        )
        if completed.stderr.strip():
            print(completed.stderr.strip(), file=sys.stderr)
        if not temp_path.is_file():
            detail = completed.stderr.strip() or completed.stdout.strip() or "no report"
            return None, f"collector failed (exit {completed.returncode}): {detail}"
        return json.loads(temp_path.read_text(encoding="utf-8")), None
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as exc:
        return None, f"collector failed: {exc}"
    finally:
        temp_path.unlink(missing_ok=True)


def build_output(
    task: str,
    search_block: dict[str, Any],
    collector_report: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    if collector_report is not None:
        output = dict(collector_report)
        output["generated_at"] = datetime.now(UTC).isoformat()
    else:
        output = {
            "task": task,
            "generated_at": datetime.now(UTC).isoformat(),
            "pages": [],
            "aggregate": {},
            "stats": {"requested": 0, "ok": 0, "failed": 0, "by_status": {}},
        }
    output["search"] = search_block
    if error is not None:
        output["search"]["collector_error"] = error
    return output


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Задача -> локальный поиск ссылок -> сбор фактов без HTML"
    )
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument(
        "--field",
        action="append",
        default=[],
        metavar="NAME[:KIND[:DESCRIPTION]]",
        help="explicit report field; skips the planner model",
    )
    parser.add_argument(
        "--provider", choices=("searxng", "ddg"), default="searxng"
    )
    parser.add_argument(
        "--searxng-url",
        type=str,
        default="http://localhost:8080",
        help="локальный SearXNG (JSON API должен быть включён)",
    )
    parser.add_argument("--language", type=str, default="ru")
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="ограничить поиск доменом (можно повторять), например sber.ru",
    )
    parser.add_argument(
        "--domains",
        action="append",
        default=[],
        metavar="LIST",
        help="то же, но списком через запятую: sber.ru,cloud.vk.com",
    )
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--max-pages", type=int, default=8)
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="имя модели в локальном Ollama/OpenAI-сервере; без неё задача идёт как запрос",
    )
    parser.add_argument("--llm-url", type=str, default="http://localhost:11434")
    parser.add_argument("--llm-provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    add_cache_arguments(parser, default_path=REPO_ROOT / ".cache/web2spec.sqlite")
    parser.add_argument(
        "--previous",
        type=Path,
        default=None,
        help="prior collector report; unchanged pages are reused",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="только найти ссылки, не запускать сбор фактов",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data/reports/search_collected.json"
    )
    parser.add_argument(
        "--planner", type=str, default=str(DEFAULT_PLANNER)
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default=str(DEFAULT_EXTRACTOR),
    )
    parser.add_argument(
        "--backend", choices=("auto", "gguf", "hf"), default="auto"
    )
    parser.add_argument("--llama-server", type=str, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-chars", type=int, default=12_000)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument(
        "--allow-partial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="поля плана без факта уходят в missing, а не роняют страницу",
    )
    parser.add_argument("--html-dir", type=Path, default=None)
    parser.add_argument("--plan-max-new-tokens", type=int, default=320)
    parser.add_argument("--fact-max-new-tokens", type=int, default=384)
    parser.add_argument("--collector-timeout", type=float, default=3600.0)
    args = parser.parse_args(argv)

    task = _clean_task(args.task)
    if len(task) < 5:
        parser.error("--task должен содержать минимум 5 символов")
    if len(task) > 500:
        parser.error("--task должен содержать не больше 500 символов")
    if args.max_results < 1 or args.max_pages < 1:
        parser.error("--max-results и --max-pages должны быть >= 1")

    cache = cache_from_args(args)
    block, urls, warnings = search_urls(
        task,
        provider=args.provider,
        searxng_url=args.searxng_url,
        language=args.language,
        max_results=args.max_results,
        llm_model=args.llm_model,
        llm_url=args.llm_url,
        llm_provider=args.llm_provider,
        llm_timeout=args.llm_timeout,
        timeout=args.timeout,
        search_only=args.search_only,
        domains=args.domain
        + [
            part.strip()
            for group in args.domains
            for part in group.split(",")
            if part.strip()
        ],
        cache=cache,
    )
    for warning in warnings:
        print(f"[warning] {warning}", file=sys.stderr)

    if not urls:
        payload = build_output(task, block, None, "no candidate URLs found")
        if cache is not None:
            payload["search"]["cache"] = cache.as_dict()
            cache.close()
        _write(args.out, payload)
        print(f"Ссылки не найдены -> {args.out}", file=sys.stderr)
        return 1

    print(
        f"Найдено {len(urls)} ссылок ({block['query_source']}): {args.out}",
        file=sys.stderr,
    )
    if args.search_only:
        payload = build_output(task, block, None, None)
        if cache is not None:
            payload["search"]["cache"] = cache.as_dict()
            cache.close()
        _write(args.out, payload)
        return 0

    collector_report, error = _run_collector(args, urls)
    if error is not None:
        print(f"[error] {error}", file=sys.stderr)
    payload = build_output(task, block, collector_report, error)
    if cache is not None:
        payload["search"]["cache"] = cache.as_dict()
        cache.close()
    _write(args.out, payload)
    stats = payload["stats"]
    print(
        f"Готово: {stats['ok']}/{stats['requested']} страниц ok, "
        f"{stats['failed']} с ошибкой -> {args.out}"
    )
    if error is not None:
        return 1
    return 0 if stats["ok"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
