from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "21_collect_urls.py"
_spec = importlib.util.spec_from_file_location("collect_urls", SCRIPT)
assert _spec is not None and _spec.loader is not None
collect_urls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect_urls)


PLAN = {
    "version": 1,
    "goal": "Extract product title and price",
    "entity": "product",
    "entry": "detail",
    "fields": [
        {"name": "title", "why": "product name", "required": True, "kind": "text"},
        {"name": "price", "why": "product price", "required": True, "kind": "number"},
    ],
    "filters": {},
    "pagination": False,
    "detail_pages": False,
    "dedup_key": "url",
    "stop_conditions": [],
}

PLAN_WITH_STOCK = {
    **PLAN,
    "fields": PLAN["fields"]
    + [{"name": "stock", "why": "availability text", "kind": "text"}],
}

HTML = (
    '<html><body><h1>Чайник</h1><span class="price">990 ₽</span>'
    '<p class="brand">Дом</p></body></html>'
)


def _plan_raw(plan: dict = PLAN) -> str:
    return json.dumps(plan, ensure_ascii=False)


def _facts_raw(facts: list[dict], missing: list[str] | None = None) -> str:
    return json.dumps({"facts": facts, "missing": missing or []}, ensure_ascii=False)


def test_collect_urls_reads_files_and_dedups(tmp_path: Path) -> None:
    path = tmp_path / "urls.txt"
    path.write_text(
        "# comment\nhttps://a.example/1\n\nhttps://b.example/2 extra\nhttps://a.example/1\n",
        encoding="utf-8",
    )
    urls = collect_urls.collect_urls([path], ["https://c.example/3", "https://a.example/1"])
    assert urls == ["https://a.example/1", "https://b.example/2", "https://c.example/3"]


def test_local_html_path_sha1_and_segment(tmp_path: Path) -> None:
    url = "https://shop.example/product/mug?utm=1"
    by_sha = tmp_path / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.html"
    by_sha.write_text("<html></html>", encoding="utf-8")
    assert collect_urls.local_html_path(tmp_path, url) == by_sha.resolve()

    by_sha.unlink()
    by_seg = tmp_path / "mug.html"
    by_seg.write_text("<html></html>", encoding="utf-8")
    assert collect_urls.local_html_path(tmp_path, url) == by_seg.resolve()

    assert collect_urls.local_html_path(tmp_path, "https://n.example/missing") is None


def test_assemble_page_happy_path() -> None:
    page = collect_urls.assemble_page(
        "https://shop.example/product/mug",
        HTML,
        _plan_raw(),
        _facts_raw(
            [
                {"name": "title", "value": "Чайник", "evidence": "Чайник"},
                {"name": "price", "value": 990, "evidence": "990 ₽"},
            ]
        ),
    )
    assert page["status"] == "ok"
    assert page["entity"] == "product"
    assert {fact["name"] for fact in page["facts"]} == {"title", "price"}
    assert page["missing"] == []
    assert page["error"] is None
    assert page["repairs"] == {"evidence": [], "dropped": []}
    assert "990" in HTML


def test_assemble_page_drops_foreign_facts() -> None:
    page = collect_urls.assemble_page(
        "https://shop.example/product/mug",
        HTML,
        _plan_raw(),
        _facts_raw(
            [
                {"name": "title", "value": "Чайник", "evidence": "Чайник"},
                {"name": "price", "value": 990, "evidence": "990 ₽"},
                {"name": "brand", "value": "Дом", "evidence": "Дом"},
            ]
        ),
    )
    assert page["status"] == "ok"
    assert {fact["name"] for fact in page["facts"]} == {"title", "price"}
    assert [row["name"] for row in page["dropped_foreign"]] == ["brand"]


def test_assemble_page_repairs_evidence() -> None:
    page = collect_urls.assemble_page(
        "https://shop.example/product/mug",
        HTML,
        _plan_raw(),
        _facts_raw(
            [
                {"name": "title", "value": "Чайник", "evidence": "Чайник"},
                {"name": "price", "value": 990, "evidence": "цена: 990 руб."},
            ]
        ),
    )
    assert page["status"] == "ok"
    assert page["repairs"]["evidence"][0]["name"] == "price"
    assert page["repairs"]["evidence"][0]["new"] in HTML


def test_assemble_page_ungrounded_fact_goes_to_missing() -> None:
    page = collect_urls.assemble_page(
        "https://shop.example/product/mug",
        HTML,
        _plan_raw(),
        _facts_raw(
            [
                {"name": "title", "value": "Чайник", "evidence": "Чайник"},
                {"name": "price", "value": 55555, "evidence": "не на странице"},
            ]
        ),
    )
    assert page["status"] == "ok"
    assert {fact["name"] for fact in page["facts"]} == {"title"}
    assert page["missing"] == ["price"]
    assert [row["name"] for row in page["repairs"]["dropped"]] == ["price"]


def test_assemble_page_uncovered_field_is_extract_error() -> None:
    page = collect_urls.assemble_page(
        "https://shop.example/product/mug",
        HTML,
        _plan_raw(PLAN_WITH_STOCK),
        _facts_raw([{"name": "title", "value": "Чайник", "evidence": "Чайник"}]),
    )
    assert page["status"] == "extract_error"
    assert page["facts"] == []
    assert page["missing"] == []
    assert "without a fact or missing" in page["error"]


def test_assemble_page_invalid_plan() -> None:
    page = collect_urls.assemble_page(
        "https://shop.example/product/mug", HTML, "not json", ""
    )
    assert page["status"] == "plan_error"
    assert page["facts"] == []
    assert page["error"]


def test_assemble_page_empty_plan_is_ok() -> None:
    empty = {**PLAN, "entity": "none", "fields": []}
    page = collect_urls.assemble_page(
        "https://shop.example/product/mug", HTML, _plan_raw(empty), ""
    )
    assert page["status"] == "ok"
    assert page["entity"] == "none"
    assert page["facts"] == []
    assert page["missing"] == []


def test_assemble_page_recovers_lenient_plan() -> None:
    plan = {
        "goal": "Extract product title and price",
        "extra_key": "ignored",
        "entity": "none",
        "fields": [
            {"name": "title", "kind": "string"},
            {"name": "price", "kind": "currency", "why": "cost"},
        ],
    }
    page = collect_urls.assemble_page(
        "https://shop.example/product/mug",
        HTML,
        _plan_raw(plan),
        _facts_raw(
            [
                {"name": "title", "value": "Чайник", "evidence": "Чайник"},
                {"name": "price", "value": 990, "evidence": "990 ₽"},
            ]
        ),
        task="extract product title and price",
    )
    assert page["status"] == "ok"
    assert page["entity"] == "page"
    assert {fact["name"] for fact in page["facts"]} == {"title", "price"}


def test_assemble_page_skips_second_compression(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("already compressed HTML must not be compressed again")

    monkeypatch.setattr(collect_urls, "compress_html", fail)
    page = collect_urls.assemble_page(
        "https://shop.example/product/mug",
        HTML,
        _plan_raw(),
        _facts_raw(
            [
                {"name": "title", "value": "Чайник", "evidence": "Чайник"},
                {"name": "price", "value": 990, "evidence": "990 ₽"},
            ]
        ),
        html_is_compressed=True,
    )
    assert page["status"] == "ok"


def test_build_report_shape_and_stats() -> None:
    ok = collect_urls.assemble_page(
        "https://shop.example/product/mug",
        HTML,
        _plan_raw(),
        _facts_raw(
            [
                {"name": "title", "value": "Чайник", "evidence": "Чайник"},
                {"name": "price", "value": 990, "evidence": "990 ₽"},
            ]
        ),
    )
    failed = collect_urls._empty_page("https://shop.example/product/gone")
    report = collect_urls.build_report("Extract product", [ok, failed])
    assert report["task"] == "Extract product"
    assert report["generated_at"].endswith("+00:00")
    assert report["stats"] == {
        "requested": 2,
        "ok": 1,
        "failed": 1,
        "by_status": {"ok": 1, "fetch_error": 1},
    }
    assert set(report["aggregate"]) == {ok["url"], failed["url"]}
    assert report["aggregate"][ok["url"]]["facts"] == {"title": "Чайник", "price": 990}
    assert report["aggregate"][failed["url"]]["status"] == "fetch_error"


def test_cli_offline_end_to_end(tmp_path: Path, monkeypatch) -> None:
    url = "https://shop.example/product/mug"
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    (html_dir / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.html").write_text(
        HTML, encoding="utf-8"
    )
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text(f"# list\n{url}\n", encoding="utf-8")
    out = tmp_path / "collected.json"

    plan_raw = _plan_raw()
    facts_raw = _facts_raw(
        [
            {"name": "title", "value": "Чайник", "evidence": "Чайник"},
            {"name": "price", "value": 990, "evidence": "990 ₽"},
        ]
    )
    monkeypatch.setattr(
        collect_urls,
        "generate_plans",
        lambda items, model_dir, max_new_tokens, task, **kwargs: {
            item_url: plan_raw for item_url, _ in items
        },
    )
    monkeypatch.setattr(
        collect_urls,
        "generate_facts",
        lambda items, model_dir, max_new_tokens, task, **kwargs: {
            item_url: facts_raw for item_url, _, _ in items
        },
    )

    rc = collect_urls.main(
        [
            "--task",
            "Extract product title and price",
            "--urls",
            str(urls_file),
            "--html-dir",
            str(html_dir),
            "--out",
            str(out),
            "--planner",
            "unused",
            "--extractor",
            "unused",
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["pages"][0]["status"] == "ok"
    assert report["pages"][0]["url"] == url
    assert report["aggregate"][url]["facts"] == {"title": "Чайник", "price": 990}
    assert "input_html" not in json.dumps(report)


def test_cli_offline_all_failed_returns_one(tmp_path: Path) -> None:
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://missing.example/page\n", encoding="utf-8")
    out = tmp_path / "collected.json"
    rc = collect_urls.main(
        [
            "--task",
            "Extract product",
            "--urls",
            str(urls_file),
            "--html-dir",
            str(tmp_path / "empty"),
            "--out",
            str(out),
        ]
    )
    assert rc == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["stats"]["ok"] == 0
    assert report["pages"][0]["status"] == "fetch_error"