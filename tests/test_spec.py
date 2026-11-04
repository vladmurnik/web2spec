from __future__ import annotations

import json

import pytest

from web2spec.spec import (
    ScrapingSpec,
    SpecError,
    extract,
    load_spec,
    next_page_url,
    render_python,
)

LIST_HTML = """
<html><body>
<div class="card"><a class="title" href="/p/1">Первый товар</a>
<span class="price">100 ₽</span></div>
<div class="card"><a class="title" href="/p/2">Второй товар</a>
<span class="price">200 ₽</span></div>
<a class="next" href="/page/2">next</a>
</body></html>
"""

LIST_SPEC = {
    "version": 1,
    "item_selector": "div.card",
    "fields": {
        "title": {"selector": "a.title", "required": True},
        "url": {"selector": "a.title", "attr": "href", "required": True},
        "price": {"selector": "span.price"},
    },
    "pagination": {"selector": "a.next", "attr": "href", "max_pages": 3},
}

DETAIL_HTML = """
<html><body><main><h1 class="title">Товар</h1><span class="price">300 ₽</span></main></body></html>
"""

DETAIL_SPEC = {
    "version": 1,
    "fields": {
        "title": {"selector": "h1.title", "required": True},
        "price": {"selector": "span.price"},
        "brand": {"selector": "span.brand"},
    },
}


def test_load_spec_and_execute_list() -> None:
    spec = load_spec(LIST_SPEC)
    rows = extract(spec, LIST_HTML, base_url="https://example.ru/catalog")
    assert [row["title"] for row in rows] == ["Первый товар", "Второй товар"]
    assert rows[0]["url"] == "https://example.ru/p/1"
    assert rows[1]["price"] == "200 ₽"


def test_detail_optional_field_is_omitted() -> None:
    spec = load_spec(DETAIL_SPEC)
    row = extract(spec, DETAIL_HTML)
    assert row == {"title": "Товар", "price": "300 ₽"}


def test_required_missing_is_none() -> None:
    spec = load_spec(DETAIL_SPEC)
    row = extract(spec, "<html><body><span class='price'>1</span></body></html>")
    assert row["title"] is None


def test_pagination_url_is_resolved() -> None:
    spec = load_spec(LIST_SPEC)
    url = next_page_url(spec, LIST_HTML, base_url="https://example.ru/catalog")
    assert url == "https://example.ru/page/2"


def test_next_page_absent() -> None:
    spec = load_spec(LIST_SPEC)
    assert next_page_url(spec, "<html><body></body></html>") is None


def test_render_python_is_executable() -> None:
    spec = load_spec(LIST_SPEC)
    code = render_python(spec)
    compile(code, "<generated>", "exec")
    namespace: dict[str, object] = {}
    exec(code, namespace)  # noqa: S102 - generated code under test
    rows = namespace["scrape"](LIST_HTML, "https://example.ru/catalog")  # type: ignore[operator]
    assert rows[0]["url"] == "https://example.ru/p/1"
    assert namespace["next_page"](LIST_HTML, "https://example.ru/") == "https://example.ru/page/2"


def test_render_python_detail() -> None:
    spec = load_spec(DETAIL_SPEC)
    code = render_python(spec)
    namespace: dict[str, object] = {}
    exec(code, namespace)  # noqa: S102
    assert namespace["scrape"](DETAIL_HTML)["title"] == "Товар"  # type: ignore[operator]


def test_invalid_spec_rejected() -> None:
    with pytest.raises(SpecError):
        load_spec({"version": 2, "fields": {"title": {"selector": "h1"}}})
    with pytest.raises(SpecError):
        load_spec({"version": 1, "fields": {}})
    with pytest.raises(SpecError):
        load_spec("not json")


def test_spec_json_roundtrip() -> None:
    spec = ScrapingSpec.model_validate(LIST_SPEC)
    assert json.loads(spec.model_dump_json())["item_selector"] == "div.card"


def test_empty_html_rejected() -> None:
    spec = load_spec(DETAIL_SPEC)
    with pytest.raises(SpecError):
        extract(spec, "")
