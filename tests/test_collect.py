from __future__ import annotations

import json

import pytest

from web2spec.collect import (
    CollectError,
    _availability,
    _number,
    build_record,
    discover_links,
    extract_jsonld,
    load_seeds,
    pick_primary,
)

PRODUCT_LD = {
    "@context": "https://schema.org",
    "@type": "Product",
    "name": "Кружка керамическая",
    "brand": {"@type": "Brand", "name": "Дом"},
    "sku": "MUG-1",
    "aggregateRating": {"@type": "AggregateRating", "ratingValue": "4.8", "reviewCount": 12},
    "offers": {
        "@type": "Offer",
        "price": "1290",
        "priceCurrency": "RUB",
        "availability": "https://schema.org/InStock",
    },
}


def product_html(ld: dict | None = None) -> str:
    payload = json.dumps(ld or PRODUCT_LD, ensure_ascii=False)
    return f"""<!doctype html><html lang="ru"><head>
    <title>Кружка керамическая — магазин</title>
    <script type="application/ld+json">{payload}</script>
    </head><body><h1>Кружка керамическая</h1>
    <div class="price">1 290 ₽</div><div>В наличии</div>
    <p>Простая керамическая кружка объёмом 300 мл.</p></body></html>"""


def test_extract_jsonld_handles_graph() -> None:
    html = """<script type="application/ld+json">
    {"@context":"https://schema.org","@graph":[
      {"@type":"WebSite","name":"Shop"},
      {"@type":"Product","name":"Чайник","offers":{"price":999,"priceCurrency":"RUB"}}
    ]}</script>"""
    objects = extract_jsonld(html)
    assert len(objects) == 2
    picked = pick_primary(objects)
    assert picked is not None
    assert picked[0] == "product"
    assert picked[1]["name"] == "Чайник"


def test_build_product_record() -> None:
    record = build_record("https://shop.example/product/mug", product_html())
    assert record["task"] == "extract"
    assert record["group_id"] == "shop.example"
    assert record["target"] == {
        "title": "Кружка керамическая",
        "price": 1290,
        "currency": "RUB",
        "availability": "in_stock",
        "brand": "Дом",
        "sku": "MUG-1",
        "rating": 4.8,
        "review_count": 12,
    }
    assert "application/ld+json" not in record["input_html"]
    assert "1 290" in record["input_html"] or "1290" in record["input_html"]
    assert record["meta"]["lang"] == "ru"
    assert 1 <= record["meta"]["complexity"] <= 3


def test_build_article_record() -> None:
    ld = {
        "@type": "NewsArticle",
        "headline": "Рынок вырос",
        "author": {"@type": "Person", "name": "Иван Петров"},
        "datePublished": "2026-10-01T10:00:00+03:00",
        "articleSection": "Экономика",
        "description": "Краткое описание события.",
    }
    html = f"""<html><head><script type="application/ld+json">
    {json.dumps(ld, ensure_ascii=False)}</script></head>
    <body><h1>Рынок вырос</h1><p>Иван Петров, 1 октября</p>
    <p>Краткое описание события.</p></body></html>"""
    record = build_record("https://news.example/article/1", html)
    assert record["target"]["title"] == "Рынок вырос"
    assert record["target"]["author"] == "Иван Петров"
    assert record["target"]["published_at"].startswith("2026-10-01")
    assert record["target"]["section"] == "Экономика"


def test_build_listing_record() -> None:
    ld = {
        "@type": "ItemList",
        "itemListElement": [
            {
                "@type": "ListItem",
                "item": {
                    "@type": "Product",
                    "name": "Стол",
                    "offers": {"price": 5000, "priceCurrency": "RUB"},
                },
            },
            {
                "@type": "ListItem",
                "item": {
                    "@type": "Product",
                    "name": "Стул",
                    "offers": {"price": 2000, "priceCurrency": "RUB"},
                },
            },
        ],
    }
    html = f"""<html><head><script type="application/ld+json">
    {json.dumps(ld, ensure_ascii=False)}</script></head>
    <body><h1>Мебель</h1><p>Стол</p><p>Стул</p></body></html>"""
    record = build_record("https://shop.example/catalog/furniture", html, category="listings")
    assert record["target"]["count"] == 2
    assert record["target"]["items"][0]["title"] == "Стол"
    assert record["target"]["items"][0]["price"] == 5000


def test_title_must_be_visible() -> None:
    ld = dict(PRODUCT_LD)
    ld["name"] = "Товара нет на странице"
    with pytest.raises(CollectError) as exc:
        build_record("https://shop.example/product/x", product_html(ld))
    assert exc.value.code == "E_TITLE_NOT_VISIBLE"


def test_no_jsonld_is_rejected() -> None:
    with pytest.raises(CollectError) as exc:
        build_record("https://shop.example/product/x", "<html><body>hello</body></html>")
    assert exc.value.code == "E_NO_JSONLD"


def test_wrong_category_is_rejected() -> None:
    with pytest.raises(CollectError) as exc:
        build_record(
            "https://shop.example/product/mug", product_html(), category="articles"
        )
    assert exc.value.code == "E_WRONG_CATEGORY"


def test_discover_links_filters_and_dedups() -> None:
    base = "https://shop.example/catalog/"
    html = """
    <a href="/product/a">A</a>
    <a href="/product/a#reviews">A again</a>
    <a href="/product/b?utm_source=x">B</a>
    <a href="https://other.example/product/c">C</a>
    <a href="/images/p.png">img</a>
    <a href="/cart">cart</a>
    <a href="mailto:x@example.com">mail</a>
    """
    links = discover_links(html, base)
    assert links == ["https://shop.example/product/a", "https://shop.example/product/b"]


def test_number_and_availability_normalization() -> None:
    assert _number("1 290,50") == 1290.5
    assert _number("12 990 ₽") == 12990
    assert _number(0) is None
    assert _number(True) is None
    assert _availability("https://schema.org/OutOfStock") == "out_of_stock"
    assert _availability("https://schema.org/PreOrder") == "preorder"
    assert _availability("https://schema.org/InStock") == "in_stock"
    assert _availability(None) is None


def test_load_seeds_skips_comments(tmp_path) -> None:
    path = tmp_path / "seeds.txt"
    path.write_text("# comment\nhttps://a.example\n\nhttps://b.example extra\n", encoding="utf-8")
    assert load_seeds(path) == ["https://a.example", "https://b.example"]
