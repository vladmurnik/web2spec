from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "22_search_and_collect.py"
_spec = importlib.util.spec_from_file_location("search_and_collect", SCRIPT)
assert _spec is not None and _spec.loader is not None
search_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(search_cli)


def test_normalize_url_strips_tracking_and_fragment() -> None:
    url = search_cli.normalize_url(
        "HTTPS://Example.COM/path/?utm_source=x&b=2&a=1#frag"
    )
    assert url == "https://example.com/path?b=2&a=1"
    assert search_cli.normalize_url("mailto:user@example.com") == ""


def test_deterministic_queries_uses_task_and_short_variant() -> None:
    queries = search_cli.deterministic_queries(
        "Найди статьи про рынок недвижимости в Москве"
    )
    assert queries[0] == "Найди статьи про рынок недвижимости в Москве"
    assert len(queries) == 2
    assert queries[1].casefold() != queries[0].casefold()


def test_parse_queries_accepts_wrapped_and_plain_json() -> None:
    assert search_cli._parse_queries('{"queries":["a","b"]}') == ["a", "b"]
    assert search_cli._parse_queries('["a","b"]') == ["a", "b"]
    assert search_cli._parse_queries("not json") == []


def test_llm_queries_ollama(monkeypatch) -> None:
    def fake_http_json(url, *, data=None, timeout=60.0):
        assert url == "http://localhost:11434/api/generate"
        assert data["model"] == "qwen2.5:3b"
        return {"response": '{"queries":["купить RTX 5060 Ti","цена RTX 5060 Ti"]}'}, None

    monkeypatch.setattr(search_cli, "_http_json", fake_http_json)
    queries, error = search_cli.llm_queries(
        "сравни цены", model="qwen2.5:3b", base_url="http://localhost:11434"
    )
    assert error is None
    assert queries == ["купить RTX 5060 Ti", "цена RTX 5060 Ti"]


def test_llm_queries_openai(monkeypatch) -> None:
    def fake_http_json(url, *, data=None, timeout=60.0):
        assert url == "http://localhost:1234/v1/chat/completions"
        return {
            "choices": [{"message": {"content": '{"queries":["ноутбук цена"]}'}}]
        }, None

    monkeypatch.setattr(search_cli, "_http_json", fake_http_json)
    queries, error = search_cli.llm_queries(
        "найди ноутбуки",
        model="local",
        base_url="http://localhost:1234",
        provider="openai",
    )
    assert error is None
    assert queries == ["ноутбук цена"]


def test_llm_queries_reports_server_error(monkeypatch) -> None:
    monkeypatch.setattr(
        search_cli, "_http_json", lambda *a, **k: (None, "connection refused")
    )
    queries, error = search_cli.llm_queries(
        "найди ноутбуки", model="local", base_url="http://localhost:11434"
    )
    assert queries == []
    assert "connection refused" in error


def test_search_searxng_parses_results(monkeypatch) -> None:
    def fake_http_json(url, *, data=None, timeout=60.0):
        assert "format=json" in url
        return {
            "results": [
                {"url": "https://a.example/1", "title": "A", "content": "alpha"},
                {"url": 42, "title": "bad"},
                {"title": "missing url"},
            ]
        }, None

    monkeypatch.setattr(search_cli, "_http_json", fake_http_json)
    results, error = search_cli.search_searxng(
        "test", base_url="http://localhost:8080"
    )
    assert error is None
    assert results == [
        {"url": "https://a.example/1", "title": "A", "snippet": "alpha"}
    ]


DDG_HTML = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">Example</a>
  <a class="result__snippet">Useful snippet</a>
</div>
<div class="result">
  <a class="result__a" href="https://second.example/x">Second</a>
</div>
"""


def test_search_ddg_parses_and_decodes_redirects(monkeypatch) -> None:
    monkeypatch.setattr(search_cli, "_http", lambda *a, **k: (DDG_HTML, None))
    results, error = search_cli.search_ddg("test")
    assert error is None
    assert results[0]["url"] == "https://example.com/page"
    assert results[0]["title"] == "Example"
    assert results[1]["url"] == "https://second.example/x"


def test_rank_candidates_dedupes_and_limits() -> None:
    per_query = [
        (
            "q1",
            [
                {"url": "https://a.example/1?utm_source=x", "title": "RTX price", "snippet": ""},
                {"url": "https://b.example/2", "title": "other", "snippet": ""},
            ],
        ),
        (
            "q2",
            [
                {"url": "https://a.example/1", "title": "RTX price", "snippet": "cheap"},
            ],
        ),
    ]
    ranked = search_cli.rank_candidates("RTX price", per_query, limit=1)
    assert len(ranked) == 1
    assert ranked[0]["url"] == "https://a.example/1"
    assert ranked[0]["queries"] == ["q1", "q2"]


def test_search_urls_falls_back_when_llm_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        search_cli,
        "llm_queries",
        lambda *a, **k: ([], "connection refused"),
    )
    monkeypatch.setattr(
        search_cli,
        "search_searxng",
        lambda query, **k: (
            [{"url": "https://a.example/1", "title": "A", "snippet": ""}],
            None,
        ),
    )
    block, urls, warnings = search_cli.search_urls(
        "найди статьи",
        provider="searxng",
        searxng_url="http://localhost:8080",
        language="ru",
        max_results=5,
        llm_model="qwen2.5:3b",
    )
    assert block["query_source"] == "deterministic"
    assert urls == ["https://a.example/1"]
    assert warnings and "connection refused" in warnings[0]


def test_search_urls_records_provider_errors(monkeypatch) -> None:
    monkeypatch.setattr(search_cli, "deterministic_queries", lambda task: ["one"])
    monkeypatch.setattr(
        search_cli, "search_searxng", lambda query, **k: ([], "http 429")
    )
    block, urls, warnings = search_cli.search_urls(
        "задача",
        provider="searxng",
        searxng_url="http://localhost:8080",
        language="ru",
        max_results=5,
    )
    assert urls == []
    assert block["count"] == 0
    assert warnings and "http 429" in warnings[0]


def test_search_urls_filters_and_scopes_domains(monkeypatch) -> None:
    queries_seen: list[str] = []

    def fake_search(query, **kwargs):
        queries_seen.append(query)
        return (
            [
                {"url": "https://docs.example.com/a", "title": "A", "snippet": ""},
                {"url": "https://blog.other.com/b", "title": "B", "snippet": ""},
            ],
            None,
        )

    monkeypatch.setattr(search_cli, "deterministic_queries", lambda task: ["base query"])
    monkeypatch.setattr(search_cli, "search_searxng", fake_search)
    block, urls, warnings = search_cli.search_urls(
        "тарифы",
        provider="searxng",
        searxng_url="http://localhost:8080",
        language="ru",
        max_results=5,
        domains=["https://docs.example.com/path"],
    )
    assert queries_seen[0] == "тарифы site:docs.example.com"
    assert urls == ["https://docs.example.com/a"]
    assert block["domains"] == ["docs.example.com"]
    assert warnings == []


def test_clean_domains_accepts_urls_paths_and_comma_lists() -> None:
    assert search_cli._clean_domains(
        ["https://docs.example.com/path", "Sber.ru, cloud.vk.com", "sber.ru"]
    ) == ["docs.example.com", "sber.ru", "cloud.vk.com"]


def test_search_urls_warns_when_domain_has_no_matches(monkeypatch) -> None:
    monkeypatch.setattr(search_cli, "deterministic_queries", lambda task: ["q"])
    monkeypatch.setattr(
        search_cli,
        "search_searxng",
        lambda query, **kwargs: (
            [{"url": "https://other.com/a", "title": "A", "snippet": ""}],
            None,
        ),
    )
    block, urls, warnings = search_cli.search_urls(
        "тарифы",
        provider="searxng",
        searxng_url="http://localhost:8080",
        language="ru",
        max_results=5,
        domains=["docs.example.com"],
    )
    assert urls == []
    assert block["count"] == 0
    assert any("no search results matched domains" in item for item in warnings)


def test_build_output_merges_search_and_collector() -> None:
    collector = {
        "task": "t",
        "generated_at": "old",
        "pages": [{"url": "u", "status": "ok"}],
        "aggregate": {"u": {}},
        "stats": {"requested": 1, "ok": 1, "failed": 0, "by_status": {"ok": 1}},
    }
    block = {"provider": "ddg", "queries": ["q"], "candidates": [], "count": 0}
    output = search_cli.build_output("t", block, collector, None)
    assert output["pages"] == collector["pages"]
    assert output["search"]["provider"] == "ddg"
    assert output["generated_at"] != "old"


def test_build_output_keeps_error_without_collector() -> None:
    block = {"provider": "ddg", "queries": [], "candidates": [], "count": 0}
    output = search_cli.build_output("t", block, None, "boom")
    assert output["stats"]["ok"] == 0
    assert output["search"]["collector_error"] == "boom"


def test_cli_search_only(monkeypatch, tmp_path) -> None:
    def fake_search(task, **kwargs):
        block = {
            "provider": "ddg",
            "query_source": "deterministic",
            "queries": [task],
            "candidates": [
                {"url": "https://a.example/1", "title": "A", "snippet": "", "score": 1.0}
            ],
            "count": 1,
            "search_only": True,
            "warnings": [],
        }
        return block, ["https://a.example/1"], []

    monkeypatch.setattr(search_cli, "search_urls", fake_search)
    out = tmp_path / "out.json"
    code = search_cli.main(
        ["--task", "найди данные", "--search-only", "--out", str(out)]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["search"]["count"] == 1
    assert payload["pages"] == []


def test_cli_no_urls_returns_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        search_cli,
        "search_urls",
        lambda task, **kwargs: (
            {"provider": "ddg", "queries": [], "candidates": [], "count": 0},
            [],
            [],
        ),
    )
    out = tmp_path / "none.json"
    code = search_cli.main(["--task", "найди данные", "--out", str(out)])
    assert code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["search"]["collector_error"] == "no candidate URLs found"


def test_collector_argv_passes_urls_and_offline_dir(tmp_path) -> None:
    args = search_cli.argparse.Namespace(
        task="t",
        out=tmp_path / "o.json",
        max_pages=8,
        max_chars=12000,
        timeout=20.0,
        delay=1.0,
        planner="p",
        extractor="e",
        plan_max_new_tokens=320,
        fact_max_new_tokens=384,
        html_dir=tmp_path,
    )
    argv = search_cli._collector_argv(args, ["https://a", "https://b"], tmp_path / "o.json")
    assert argv[1].endswith("21_collect_urls.py")
    assert "--html-dir" in argv
    assert "--allow-partial" in argv
    assert argv.count("--url") == 2
    assert str(tmp_path) in argv


def _args(tmp_path, **overrides):
    values = dict(
        task="t",
        out=tmp_path / "final.json",
        max_pages=8,
        max_chars=12000,
        timeout=20.0,
        delay=1.0,
        planner="p",
        extractor="e",
        plan_max_new_tokens=320,
        fact_max_new_tokens=384,
        html_dir=None,
        collector_timeout=60.0,
    )
    values.update(overrides)
    return search_cli.argparse.Namespace(**values)


def test_run_collector_reads_temp_report(monkeypatch, tmp_path) -> None:
    report = {
        "task": "t",
        "pages": [{"url": "https://a", "status": "ok"}],
        "aggregate": {"https://a": {}},
        "stats": {"requested": 1, "ok": 1, "failed": 0, "by_status": {"ok": 1}},
    }

    def fake_run(argv, **kwargs):
        out = Path(argv[argv.index("--out") + 1])
        out.write_text(json.dumps(report), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(search_cli.subprocess, "run", fake_run)
    loaded, error = search_cli._run_collector(_args(tmp_path), ["https://a"])
    assert error is None
    assert loaded["stats"]["ok"] == 1


def test_cli_full_flow_merges_collector(monkeypatch, tmp_path) -> None:
    block = {
        "provider": "ddg",
        "query_source": "deterministic",
        "queries": ["q"],
        "candidates": [{"url": "https://a", "title": "A", "snippet": "", "score": 1.0}],
        "count": 1,
        "search_only": False,
        "warnings": [],
    }
    monkeypatch.setattr(
        search_cli,
        "search_urls",
        lambda task, **kwargs: (block, ["https://a"], []),
    )
    monkeypatch.setattr(
        search_cli,
        "_run_collector",
        lambda args, urls: (
            {
                "task": "t",
                "pages": [{"url": "https://a", "status": "ok", "facts": []}],
                "aggregate": {"https://a": {"status": "ok", "facts": {}}},
                "stats": {"requested": 1, "ok": 1, "failed": 0, "by_status": {"ok": 1}},
            },
            None,
        ),
    )
    out = tmp_path / "full.json"
    code = search_cli.main(["--task", "найди данные", "--out", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["search"]["count"] == 1
    assert payload["pages"][0]["url"] == "https://a"


def test_cli_collector_error_returns_one(monkeypatch, tmp_path) -> None:
    block = {
        "provider": "ddg",
        "query_source": "deterministic",
        "queries": ["q"],
        "candidates": [{"url": "https://a", "title": "A", "snippet": "", "score": 1.0}],
        "count": 1,
        "search_only": False,
        "warnings": [],
    }
    monkeypatch.setattr(
        search_cli, "search_urls", lambda task, **kwargs: (block, ["https://a"], [])
    )
    monkeypatch.setattr(
        search_cli, "_run_collector", lambda args, urls: (None, "collector boom")
    )
    out = tmp_path / "err.json"
    code = search_cli.main(["--task", "найди данные", "--out", str(out)])
    assert code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["search"]["collector_error"] == "collector boom"
