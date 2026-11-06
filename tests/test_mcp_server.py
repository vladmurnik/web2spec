from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "23_mcp_server.py"
_spec = importlib.util.spec_from_file_location("mcp_server", SCRIPT)
assert _spec is not None and _spec.loader is not None
mcp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcp)


def _report() -> dict:
    return {
        "task": "найди статьи",
        "pages": [
            {
                "url": "https://example.com/a",
                "status": "ok",
                "entity": "article",
                "facts": [
                    {"name": "title", "value": "Заголовок", "evidence": "Заголовок"},
                    {"name": "author", "value": "Иван", "evidence": "Иван"},
                ],
                "missing": ["published_at"],
                "error": None,
            }
        ],
        "aggregate": {},
        "stats": {"requested": 1, "ok": 1, "failed": 0, "by_status": {"ok": 1}},
    }


def test_tools_list_contract() -> None:
    response = mcp.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = response["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["research", "collect"]
    research = tools[0]
    assert research["inputSchema"]["required"] == ["task"]
    assert "source_hints" in research["inputSchema"]["properties"]
    assert "min_pages" in research["inputSchema"]["properties"]
    collect = tools[1]
    assert collect["inputSchema"]["required"] == ["task"]
    assert "urls" in collect["inputSchema"]["properties"]
    assert "domains" in collect["inputSchema"]["properties"]
    assert "include_pages" in collect["inputSchema"]["properties"]


def test_research_tool_clamps_pages_and_returns_report(monkeypatch) -> None:
    seen: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        out = Path(command[command.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "goal": "тарифы",
                    "answer": "ответ",
                    "brief": {"questions": ["q"]},
                    "entities": [{"entity": "yandex"}],
                    "sources": ["https://example.com"],
                    "errors": [],
                    "gaps": [],
                    "stats": {"pages_ok": 20},
                    "speed": {"calls": 3},
                    "config": {"max_pages": 50},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

        return Result()

    monkeypatch.setattr(mcp.subprocess, "run", fake_run)
    response = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "research",
                "arguments": {
                    "task": "сравни тарифы",
                    "min_pages": 1,
                    "max_pages": 99,
                },
            },
        }
    )
    assert response["result"]["isError"] is False
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["answer"] == "ответ"
    command = seen["command"]
    assert command[command.index("--min-pages") + 1] == "20"
    assert command[command.index("--max-pages") + 1] == "50"
    assert payload["stats"]["pages_ok"] == 20


def test_initialize_reports_tools_capability() -> None:
    response = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        }
    )
    assert response["result"]["protocolVersion"] == "2024-11-05"
    assert "tools" in response["result"]["capabilities"]
    assert response["result"]["serverInfo"]["name"] == "web2spec"


def test_notification_has_no_response() -> None:
    assert (
        mcp.handle_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        )
        is None
    )


def test_unknown_method_returns_jsonrpc_error() -> None:
    response = mcp.handle_message(
        {"jsonrpc": "2.0", "id": 2, "method": "resources/list"}
    )
    assert response["error"]["code"] == -32601


def test_search_links_tool(monkeypatch) -> None:
    def fake_search(task, **kwargs):
        assert kwargs["max_results"] == 3
        return (
            {
                "queries": ["q1"],
                "query_source": "deterministic",
                "candidates": [
                    {
                        "url": "https://a.example/1",
                        "title": "A",
                        "snippet": "s",
                        "score": 0.5,
                    }
                ],
            },
            ["https://a.example/1"],
            [],
        )

    monkeypatch.setattr(mcp.search_cli, "search_urls", fake_search)
    response = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_links",
                "arguments": {"task": "найди данные", "max_results": 3},
            },
        }
    )
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["queries"] == ["q1"]
    assert payload["links"][0]["url"] == "https://a.example/1"


def test_collect_with_urls_returns_compact_facts(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp.search_cli, "_run_collector", lambda ns, urls: (_report(), None)
    )
    response = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "collect",
                "arguments": {
                    "task": "найди статьи",
                    "urls": ["https://example.com/a"],
                },
            },
        }
    )
    assert response["result"]["isError"] is False
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["facts"] == {"title": "Заголовок", "author": "Иван"}
    assert payload["missing"] == {"published_at": ["https://example.com/a"]}
    assert payload["sources"] == ["https://example.com/a"]
    assert "evidence" not in response["result"]["content"][0]["text"]


def test_collect_include_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp.search_cli, "_run_collector", lambda ns, urls: (_report(), None)
    )
    response = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "collect",
                "arguments": {
                    "task": "найди статьи",
                    "urls": ["https://example.com/a"],
                    "include_evidence": True,
                },
            },
        }
    )
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["evidence"]["title"][0]["quote"] == "Заголовок"
    assert "pages" not in payload


def test_collect_include_pages_returns_per_page_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp.search_cli, "_run_collector", lambda ns, urls: (_report(), None)
    )
    response = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "collect",
                "arguments": {
                    "task": "найди статьи",
                    "urls": ["https://example.com/a"],
                    "include_pages": True,
                },
            },
        }
    )
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["pages"][0]["facts"] == {"title": "Заголовок", "author": "Иван"}


def test_collect_reports_conflicts_instead_of_choosing(monkeypatch) -> None:
    report = _report()
    report["pages"].append(
        {
            "url": "https://example.com/b",
            "status": "ok",
            "entity": "article",
            "facts": [
                {"name": "title", "value": "Другой", "evidence": "Другой"},
                {"name": "author", "value": "Иван", "evidence": "Иван"},
            ],
            "missing": [],
            "error": None,
        }
    )
    monkeypatch.setattr(
        mcp.search_cli, "_run_collector", lambda ns, urls: (report, None)
    )
    response = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "collect",
                "arguments": {
                    "task": "найди статьи",
                    "urls": ["https://example.com/a", "https://example.com/b"],
                },
            },
        }
    )
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["facts"] == {"author": "Иван"}
    assert "title" not in payload["facts"]
    assert {row["value"] for row in payload["conflicts"]["title"]} == {
        "Заголовок",
        "Другой",
    }
    assert payload["conflicts"]["title"][0]["sources"]
    assert payload["by_source"]["https://example.com/b"]["title"] == "Другой"


def test_collect_without_urls_runs_search(monkeypatch) -> None:
    called = {"search": 0, "domains": None}

    def fake_search(task, **kwargs):
        called["search"] += 1
        called["domains"] = kwargs.get("domains")
        return (
            {
                "queries": ["q"],
                "query_source": "deterministic",
                "candidates": [{"url": "https://a.example/1"}],
            },
            ["https://a.example/1"],
            [],
        )

    seen = {}

    def fake_run(ns, urls):
        seen["urls"] = urls
        return _report(), None

    monkeypatch.setattr(mcp.search_cli, "search_urls", fake_search)
    monkeypatch.setattr(mcp.search_cli, "_run_collector", fake_run)
    response = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "collect",
                "arguments": {"task": "найди статьи", "domains": ["example.com"]},
            },
        }
    )
    payload = json.loads(response["result"]["content"][0]["text"])
    assert called["search"] == 1
    assert called["domains"] == ["example.com"]
    assert seen["urls"] == ["https://a.example/1"]
    assert payload["search"]["queries"] == ["q"]


def test_collect_no_urls_found(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp.search_cli,
        "search_urls",
        lambda task, **kwargs: (
            {"queries": ["q"], "query_source": "deterministic", "candidates": []},
            [],
            ["nothing"],
        ),
    )
    response = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "collect", "arguments": {"task": "найди статьи"}},
        }
    )
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["stats"]["ok"] == 0
    assert payload["search"]["collector_error"] == "no URLs found"


def test_tool_exception_becomes_is_error(monkeypatch) -> None:
    def boom(ns, urls):
        return None, "collector failed"

    monkeypatch.setattr(mcp.search_cli, "_run_collector", boom)
    response = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "collect",
                "arguments": {"task": "найди статьи", "urls": ["https://a"]},
            },
        }
    )
    assert response["result"]["isError"] is True
    assert "collector failed" in response["result"]["content"][0]["text"]


def test_unknown_tool_is_error() -> None:
    response = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        }
    )
    assert response["result"]["isError"] is True
    assert "unknown tool" in response["result"]["content"][0]["text"]


def test_serve_reads_newline_framed_messages(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp,
        "tool_search_links",
        lambda args: {"links": [{"url": "https://a", "title": "A"}]},
    )
    stdin = io.StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        + "\n"
        + json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "search_links", "arguments": {"task": "найди"}},
            }
        )
        + "\n"
    )
    stdout = io.StringIO()
    assert mcp.serve(stdin, stdout) == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    assert [line["id"] for line in lines] == [1, 2]
    payload = json.loads(lines[1]["result"]["content"][0]["text"])
    assert payload["links"][0]["url"] == "https://a"


def test_serve_supports_content_length_framing() -> None:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    stdin = io.StringIO(f"Content-Length: {len(body)}\r\n\r\n{body}")
    stdout = io.StringIO()
    assert mcp.serve(stdin, stdout) == 0
    response = json.loads(stdout.getvalue().strip())
    assert response["result"] == {}