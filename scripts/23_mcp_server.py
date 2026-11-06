"""MCP server: one call -> local research or collection -> evidence JSON.

Any MCP-capable client (Claude Desktop, Cursor, Cline, custom agent) can call

* ``research`` for a vague/complex task: a 4B local analyst builds a brief,
  searches, reads 20-50 pages and returns evidence-backed findings;
* ``collect`` for a fast URL-first path: search/fetch/extract with the 0.5B
  planner and extractor.

Both return compact JSON, never HTML.  Transport is MCP stdio:
newline-delimited JSON-RPC 2.0 on stdin/stdout.  All diagnostics go to stderr
so stdout stays protocol-clean.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_SCRIPT = REPO_ROOT / "scripts" / "22_search_and_collect.py"
RESEARCH_SCRIPT = REPO_ROOT / "scripts" / "26_research.py"

_spec = importlib.util.spec_from_file_location("web2spec_search", SEARCH_SCRIPT)
assert _spec is not None and _spec.loader is not None
search_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(search_cli)

SERVER_NAME = "web2spec"
SERVER_VERSION = "0.4.0"
DEFAULT_PROTOCOL = "2024-11-05"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "research",
        "description": (
            "ОДИН вызов для сложной или расплывчатой задачи: локальный 4B-аналитик "
            "сам уточняет вопросы, ищет, читает 20-50 страниц и возвращает отчёт "
            "с точными цитатами, конфликтами и пробелами. Используй это, когда "
            "заранее неизвестны точные страницы. Если есть официальные URL — "
            "лучше быстрый collect."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Что исследовать, на естественном языке.",
                },
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Необязательные стартовые URL.",
                },
                "domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ограничить поиск доменами, например ['sber.ru'].",
                },
                "source_hints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Подсказки об официальных доменах/типах страниц.",
                },
                "min_pages": {
                    "type": "integer",
                    "minimum": 20,
                    "maximum": 50,
                    "description": "Минимум страниц (по умолчанию 20).",
                },
                "max_pages": {
                    "type": "integer",
                    "minimum": 20,
                    "maximum": 50,
                    "description": "Максимум страниц (по умолчанию 30).",
                },
                "max_rounds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4,
                    "description": "Сколько раундов поиска (по умолчанию 2).",
                },
                "include_notes": {
                    "type": "boolean",
                    "description": "Вернуть все заметки с цитатами (по умолчанию false).",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "collect",
        "description": (
            "Быстрый ОДИН вызов: локально найти страницы или взять готовые URL, "
            "скачать их, извлечь факты и вернуть сведённый JSON без HTML. "
            "Используй, когда уже знаешь официальные URL/домены. Для сложного "
            "поиска по расплывчатой задаче используй research."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Что именно нужно найти, на естественном языке.",
                },
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Необязательный список URL. Если задан, "
                        "локальный поиск не выполняется."
                    ),
                },
                "domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Необязательные домены для поиска, например "
                        "['sber.ru', 'vk.com']; результаты вне их отбрасываются."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Сколько ссылок взять из поиска (по умолчанию 6).",
                },
                "max_pages": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Сколько страниц обработать (по умолчанию 6).",
                },
                "include_evidence": {
                    "type": "boolean",
                    "description": "Добавить точные цитаты evidence (по умолчанию false).",
                },
                "include_pages": {
                    "type": "boolean",
                    "description": (
                        "Вернуть разбивку по страницам (по умолчанию false: "
                        "только сведённые факты, конфликты и ошибки)."
                    ),
                },
            },
            "required": ["task"],
        },
    },
]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _defaults() -> dict[str, Any]:
    return {
        "provider": _env("WEB2SPEC_PROVIDER", "searxng"),
        "searxng_url": _env("WEB2SPEC_SEARXNG_URL", "http://localhost:8080"),
        "llm_model": os.environ.get("WEB2SPEC_LLM_MODEL") or None,
        "llm_url": _env("WEB2SPEC_LLM_URL", "http://localhost:11434"),
        "llm_provider": _env("WEB2SPEC_LLM_PROVIDER", "ollama"),
        "language": _env("WEB2SPEC_LANGUAGE", "ru"),
        "planner": _env(
            "WEB2SPEC_PLANNER", str(search_cli.DEFAULT_PLANNER)
        ),
        "extractor": _env(
            "WEB2SPEC_EXTRACTOR",
            str(search_cli.DEFAULT_EXTRACTOR),
        ),
        "analyst": _env("WEB2SPEC_ANALYST", str(search_cli.DEFAULT_ANALYST)),
        "research_min_pages": int(_env("WEB2SPEC_RESEARCH_MIN_PAGES", "20")),
        "research_max_pages": int(_env("WEB2SPEC_RESEARCH_MAX_PAGES", "30")),
        "backend": _env("WEB2SPEC_BACKEND", "auto"),
        "llama_server": os.environ.get("WEB2SPEC_LLAMA_SERVER") or None,
        "device": _env("WEB2SPEC_LLAMA_DEVICE", search_cli.DEFAULT_DEVICE),
        "context": int(_env("WEB2SPEC_CONTEXT", "8192")),
        "startup_timeout": float(_env("WEB2SPEC_STARTUP_TIMEOUT", "240")),
        "request_timeout": float(_env("WEB2SPEC_REQUEST_TIMEOUT", "300")),
        "max_chars": int(_env("WEB2SPEC_MAX_CHARS", "12000")),
        "timeout": float(_env("WEB2SPEC_TIMEOUT", "20")),
        "delay": float(_env("WEB2SPEC_DELAY", "1.0")),
        "html_dir": os.environ.get("WEB2SPEC_HTML_DIR") or None,
    }


def _namespace(
    *,
    task: str,
    out: Path,
    max_pages: int,
    max_chars: int,
    timeout: float,
    delay: float,
    planner: str,
    extractor: str,
    html_dir: str | None,
    backend: str,
    llama_server: str | None,
    device: str | None,
    context: int,
    startup_timeout: float,
    request_timeout: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        task=task,
        out=out,
        max_pages=max_pages,
        max_chars=max_chars,
        timeout=timeout,
        delay=delay,
        planner=planner,
        extractor=extractor,
        html_dir=Path(html_dir) if html_dir else None,
        backend=backend,
        llama_server=llama_server,
        device=device,
        context=context,
        startup_timeout=startup_timeout,
        request_timeout=request_timeout,
        allow_partial=True,
        plan_max_new_tokens=320,
        fact_max_new_tokens=384,
        collector_timeout=3600.0,
    )


def _value_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _compact(
    report: dict[str, Any],
    include_evidence: bool,
    include_pages: bool = False,
) -> dict[str, Any]:
    """Merge per-page facts so the caller does not have to read every page.

    A field with one distinct value becomes ``facts[name]``.  Disagreements are
    never silently resolved: they become ``conflicts[name]`` with every value
    and source.  Failed pages are kept as short errors, not full page dumps.
    """

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    missing: dict[str, list[str]] = {}
    sources: list[str] = []
    errors: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    by_source: dict[str, dict[str, Any]] = {}

    for page in report.get("pages", []):
        url = page.get("url")
        status = page.get("status")
        facts = page.get("facts", [])
        if status != "ok":
            errors.append({"url": url, "status": status, "error": page.get("error")})
        else:
            if url and url not in sources:
                sources.append(url)
        for name in page.get("missing", []):
            missing.setdefault(name, [])
            if url and url not in missing[name]:
                missing[name].append(url)
        for fact in facts:
            name = fact.get("name")
            if not name:
                continue
            key = _value_key(fact.get("value"))
            bucket = grouped.setdefault(name, {})
            entry = bucket.setdefault(
                key,
                {
                    "value": fact.get("value"),
                    "sources": [],
                    "evidence": [],
                },
            )
            if url and url not in entry["sources"]:
                entry["sources"].append(url)
            quote = fact.get("evidence")
            if isinstance(quote, str) and quote:
                evidence_row = {"url": url, "quote": quote}
                if evidence_row not in entry["evidence"]:
                    entry["evidence"].append(evidence_row)
        if status == "ok" and facts and url:
            by_source[url] = {fact["name"]: fact["value"] for fact in facts}
        if include_pages:
            pages.append(
                {
                    "url": url,
                    "status": status,
                    "entity": page.get("entity"),
                    "facts": {fact["name"]: fact["value"] for fact in facts},
                    "missing": page.get("missing", []),
                    "error": page.get("error"),
                }
            )

    facts: dict[str, Any] = {}
    conflicts: dict[str, list[dict[str, Any]]] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    for name, values in grouped.items():
        rows = list(values.values())
        if len(rows) == 1:
            facts[name] = rows[0]["value"]
            if include_evidence:
                evidence[name] = rows[0]["evidence"]
        else:
            conflicts[name] = [
                {
                    "value": row["value"],
                    "sources": row["sources"],
                    **( {"evidence": row["evidence"]} if include_evidence else {} ),
                }
                for row in rows
            ]

    result: dict[str, Any] = {
        "task": report.get("task"),
        "facts": facts,
        "conflicts": conflicts,
        "missing": {
            name: urls
            for name, urls in missing.items()
            if name not in facts and name not in conflicts
        },
        "sources": sources,
        "errors": errors,
        "stats": report.get("stats", {}),
        "search": report.get("search", {}),
    }
    if include_evidence:
        result["evidence"] = evidence
    if conflicts and len(by_source) > 1:
        # Multi-entity tasks (compare A/B/C): keep compact per-source rows so
        # the caller can attribute each conflicting value without reading pages.
        result["by_source"] = by_source
    if include_pages:
        result["pages"] = pages
    return result


def tool_search_links(args: dict[str, Any]) -> dict[str, Any]:
    cfg = _defaults()
    task = str(args.get("task") or "").strip()
    if len(task) < 5:
        raise ValueError("task must contain at least 5 characters")
    max_results = int(args.get("max_results") or 10)
    domains = args.get("domains") or []
    block, urls, warnings = search_cli.search_urls(
        task,
        provider=cfg["provider"],
        searxng_url=cfg["searxng_url"],
        language=cfg["language"],
        max_results=max_results,
        llm_model=cfg["llm_model"],
        llm_url=cfg["llm_url"],
        llm_provider=cfg["llm_provider"],
        search_only=True,
        domains=domains,
    )
    return {
        "task": task,
        "queries": block.get("queries", []),
        "query_source": block.get("query_source"),
        "links": [
            {
                "url": row["url"],
                "title": row.get("title", ""),
                "snippet": row.get("snippet", ""),
                "score": row.get("score"),
            }
            for row in block.get("candidates", [])
        ],
        "warnings": warnings,
    }


def tool_research(args: dict[str, Any]) -> dict[str, Any]:
    cfg = _defaults()
    task = str(args.get("task") or "").strip()
    if len(task) < 5:
        raise ValueError("task must contain at least 5 characters")
    min_pages = max(20, min(50, int(args.get("min_pages") or cfg["research_min_pages"])))
    max_pages = max(
        min_pages, min(50, int(args.get("max_pages") or cfg["research_max_pages"]))
    )
    max_rounds = max(1, min(4, int(args.get("max_rounds") or 2)))
    include_notes = bool(args.get("include_notes", False))
    raw_urls = args.get("urls") or []
    raw_domains = args.get("domains") or []
    raw_hints = args.get("source_hints") or []
    for label, value in (
        ("urls", raw_urls),
        ("domains", raw_domains),
        ("source_hints", raw_hints),
    ):
        if not isinstance(value, list):
            raise ValueError(f"{label} must be an array of strings")

    out = REPO_ROOT / "data" / "reports" / f"mcp_research_{os.getpid()}.json"
    command = [
        sys.executable,
        str(RESEARCH_SCRIPT),
        "--task",
        task,
        "--min-pages",
        str(min_pages),
        "--max-pages",
        str(max_pages),
        "--max-rounds",
        str(max_rounds),
        "--provider",
        str(cfg["provider"]),
        "--searxng-url",
        str(cfg["searxng_url"]),
        "--language",
        str(cfg["language"]),
        "--analyst",
        str(cfg["analyst"]),
        "--device",
        str(cfg["device"]),
        "--context",
        str(cfg["context"]),
        "--startup-timeout",
        str(cfg["startup_timeout"]),
        "--request-timeout",
        str(cfg["request_timeout"]),
        "--fetch-timeout",
        str(cfg["timeout"]),
        "--delay",
        str(cfg["delay"]),
        "--out",
        str(out),
    ]
    if cfg["llama_server"]:
        command += ["--llama-server", str(cfg["llama_server"])]
    if cfg["html_dir"]:
        command += ["--html-dir", str(cfg["html_dir"])]
    for url in raw_urls:
        if str(url).strip():
            command += ["--url", str(url).strip()]
    for domain in raw_domains:
        if str(domain).strip():
            command += ["--domain", str(domain).strip()]
    for hint in raw_hints:
        if str(hint).strip():
            command += ["--source-hint", str(hint).strip()]

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(_env("WEB2SPEC_RESEARCH_TIMEOUT", "3600")),
        )
        if process.returncode != 0:
            tail = "\n".join(process.stderr.strip().splitlines()[-6:])
            raise RuntimeError(f"research failed ({process.returncode}): {tail}")
        if not out.is_file():
            raise RuntimeError("research produced no report")
        report = json.loads(out.read_text(encoding="utf-8"))
    finally:
        out.unlink(missing_ok=True)

    result: dict[str, Any] = {
        "goal": report.get("goal"),
        "answer": report.get("answer", ""),
        "brief": report.get("brief", {}),
        "entities": report.get("entities", []),
        "sources": report.get("sources", []),
        "errors": report.get("errors", [])[:20],
        "gaps": report.get("gaps", []),
        "stats": report.get("stats", {}),
        "speed": report.get("speed", {}),
        "config": report.get("config", {}),
    }
    if include_notes:
        result["notes"] = report.get("notes", [])
    return result


def tool_collect(args: dict[str, Any]) -> dict[str, Any]:
    cfg = _defaults()
    task = str(args.get("task") or "").strip()
    if len(task) < 5:
        raise ValueError("task must contain at least 5 characters")
    max_results = int(args.get("max_results") or 6)
    max_pages = int(args.get("max_pages") or 6)
    include_evidence = bool(args.get("include_evidence", False))
    include_pages = bool(args.get("include_pages", False))
    raw_urls = args.get("urls") or []
    domains = args.get("domains") or []
    if not isinstance(raw_urls, list):
        raise ValueError("urls must be an array of strings")
    if not isinstance(domains, list):
        raise ValueError("domains must be an array of strings")
    urls = [str(url).strip() for url in raw_urls if str(url).strip()]

    search_block: dict[str, Any] = {}
    warnings: list[str] = []
    if not urls:
        search_block, urls, warnings = search_cli.search_urls(
            task,
            provider=cfg["provider"],
            searxng_url=cfg["searxng_url"],
            language=cfg["language"],
            max_results=max_results,
            llm_model=cfg["llm_model"],
            llm_url=cfg["llm_url"],
            llm_provider=cfg["llm_provider"],
            search_only=False,
            domains=[str(domain) for domain in domains],
        )
    if not urls:
        return {
            "task": task,
            "facts": {},
            "conflicts": {},
            "missing": {},
            "sources": [],
            "errors": [],
            "stats": {"requested": 0, "ok": 0, "failed": 0, "by_status": {}},
            "search": {
                "provider": search_block.get("provider"),
                "query_source": search_block.get("query_source"),
                "queries": search_block.get("queries", []),
                "count": 0,
                "domains": search_block.get("domains", []),
                "warnings": warnings,
                "collector_error": "no URLs found",
            },
        }

    ns = _namespace(
        task=task,
        out=REPO_ROOT / "data/reports/mcp_last.json",
        max_pages=max_pages,
        max_chars=cfg["max_chars"],
        timeout=cfg["timeout"],
        delay=cfg["delay"],
        planner=cfg["planner"],
        extractor=cfg["extractor"],
        html_dir=cfg["html_dir"],
        backend=cfg["backend"],
        llama_server=cfg["llama_server"],
        device=cfg["device"],
        context=cfg["context"],
        startup_timeout=cfg["startup_timeout"],
        request_timeout=cfg["request_timeout"],
    )
    report, error = search_cli._run_collector(ns, urls)
    if error is not None:
        raise RuntimeError(error)
    report = report or {}
    report["task"] = task
    report["search"] = {
        "provider": search_block.get("provider"),
        "query_source": search_block.get("query_source"),
        "queries": search_block.get("queries", []),
        "count": search_block.get("count", len(urls)),
        "domains": search_block.get("domains", []),
        "warnings": warnings,
    }
    return _compact(report, include_evidence, include_pages=include_pages)


def _text_result(payload: Any, *, is_error: bool = False) -> dict[str, Any]:
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }


def handle_message(request: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC message; return response or None for notifications."""

    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}

    if request_id is None:
        return None  # notification: initialized, cancelled, ...

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": params.get("protocolVersion") or DEFAULT_PROTOCOL,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": (
                    "Для сложной или расплывчатой задачи вызывай ОДИН раз "
                    "research(task=...): локальный 4B-аналитик сам ищет и читает "
                    "20-50 страниц, возвращает отчёт с цитатами, конфликтами и "
                    "пробелами. Если уже есть официальные URL/домены, вызывай "
                    "быстрый collect(task=..., urls/domains=...). Не вызывай "
                    "поиск отдельными инструментами."
                ),
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        try:
            if name == "collect":
                result = tool_collect(arguments)
            elif name == "research":
                result = tool_research(arguments)
            elif name == "search_links":
                result = tool_search_links(arguments)
            else:
                raise ValueError(f"unknown tool: {name}")
        except Exception as exc:  # noqa: BLE001 - report tool errors to the client
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": _text_result(f"{type(exc).__name__}: {exc}", is_error=True),
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": _text_result(result)}

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def _read_message(stream: Any) -> dict[str, Any] | None:
    """Read one MCP stdio message: newline JSON, with Content-Length fallback."""

    while True:
        line = stream.readline()
        if not line:
            return None
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("content-length:"):
            try:
                length = int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
            headers = line
            while True:
                extra = stream.readline()
                if isinstance(extra, bytes):
                    extra = extra.decode("utf-8", errors="replace")
                headers += extra
                if extra in ("\r\n", "\n", ""):
                    break
            body = stream.read(length)
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="replace")
            return json.loads(body)
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            print(f"[mcp] ignored malformed line: {line[:200]}", file=sys.stderr)
            continue


def _write_message(stream: Any, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def serve(stdin: Any = None, stdout: Any = None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    print(
        f"[mcp] {SERVER_NAME} {SERVER_VERSION} ready (tools: research, collect)",
        file=sys.stderr,
        flush=True,
    )
    while True:
        request = _read_message(stdin)
        if request is None:
            return 0
        if not isinstance(request, dict):
            continue
        response = handle_message(request)
        if response is not None:
            _write_message(stdout, response)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCP server web2spec: task -> evidence")
    parser.add_argument("--provider", choices=("searxng", "ddg"), default=None)
    parser.add_argument("--searxng-url", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--llm-url", type=str, default=None)
    parser.add_argument("--llm-provider", choices=("ollama", "openai"), default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--planner", type=str, default=None)
    parser.add_argument("--extractor", type=str, default=None)
    parser.add_argument("--analyst", type=str, default=None)
    parser.add_argument("--min-pages", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--backend", choices=("auto", "gguf", "hf"), default=None)
    parser.add_argument("--llama-server", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--startup-timeout", type=float, default=None)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--delay", type=float, default=None)
    parser.add_argument("--html-dir", type=str, default=None)
    args = parser.parse_args(argv)

    overrides = {
        "WEB2SPEC_PROVIDER": args.provider,
        "WEB2SPEC_SEARXNG_URL": args.searxng_url,
        "WEB2SPEC_LLM_MODEL": args.llm_model,
        "WEB2SPEC_LLM_URL": args.llm_url,
        "WEB2SPEC_LLM_PROVIDER": args.llm_provider,
        "WEB2SPEC_LANGUAGE": args.language,
        "WEB2SPEC_PLANNER": args.planner,
        "WEB2SPEC_EXTRACTOR": args.extractor,
        "WEB2SPEC_ANALYST": args.analyst,
        "WEB2SPEC_RESEARCH_MIN_PAGES": (
            str(args.min_pages) if args.min_pages else None
        ),
        "WEB2SPEC_RESEARCH_MAX_PAGES": (
            str(args.max_pages) if args.max_pages else None
        ),
        "WEB2SPEC_BACKEND": args.backend,
        "WEB2SPEC_LLAMA_SERVER": args.llama_server,
        "WEB2SPEC_LLAMA_DEVICE": args.device,
        "WEB2SPEC_CONTEXT": str(args.context) if args.context else None,
        "WEB2SPEC_STARTUP_TIMEOUT": (
            str(args.startup_timeout) if args.startup_timeout else None
        ),
        "WEB2SPEC_REQUEST_TIMEOUT": (
            str(args.request_timeout) if args.request_timeout else None
        ),
        "WEB2SPEC_MAX_CHARS": str(args.max_chars) if args.max_chars else None,
        "WEB2SPEC_TIMEOUT": str(args.timeout) if args.timeout else None,
        "WEB2SPEC_DELAY": str(args.delay) if args.delay else None,
        "WEB2SPEC_HTML_DIR": args.html_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            os.environ[key] = value
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())