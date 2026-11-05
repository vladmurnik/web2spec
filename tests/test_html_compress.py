from __future__ import annotations

import pytest

from web2spec.html_compress import TRUNCATION_MARKER, compress_html

HTML = """
<html><head><title>Test</title><style>.x{color:red}</style></head>
<body>
  <script>alert('x')</script>
  <div class="card" style="color:red" onclick="evil()" data-foo="1">
    <h1>Товар</h1>
    <p>Описание <b>товара</b></p>
    <a href="/p/1" class="link">Купить</a>
    <img src="/i.png" alt="Фото">
  </div>
</body></html>
"""


def test_removes_scripts_and_styles() -> None:
    result = compress_html(HTML)
    assert "<script" not in result
    assert "<style" not in result
    assert "alert" not in result
    assert "color:red" not in result


def test_keeps_text_and_semantic_tags() -> None:
    result = compress_html(HTML)
    assert "<h1>Товар</h1>" in result
    assert "<b>товара</b>" in result
    assert 'href="/p/1"' in result
    assert 'alt="Фото"' in result


def test_strips_event_handlers_and_unknown_attributes() -> None:
    result = compress_html(HTML)
    assert "onclick" not in result
    assert "data-foo" not in result
    assert 'class="card"' in result


def test_truncates_on_tag_boundary() -> None:
    html = "<div>" + "".join(f"<p>строка {i}</p>" for i in range(200)) + "</div>"
    result = compress_html(html, max_chars=300)
    assert len(result) <= 300
    assert result.endswith(TRUNCATION_MARKER)
    assert result.count("<") == result.count(">")


def test_idempotent() -> None:
    once = compress_html(HTML)
    assert compress_html(once) == once


def test_empty_input() -> None:
    assert compress_html("") == ""
    assert compress_html("   ") == ""


def test_rejects_tiny_budget() -> None:
    with pytest.raises(ValueError):
        compress_html("<p>x</p>", max_chars=10)


def test_plain_text_without_tags() -> None:
    result = compress_html("просто текст без разметки")
    assert "просто текст" in result


META_HTML = """
<html><head>
  <title>Заголовок статьи</title>
  <meta property="article:published_time" content="2024-01-02T10:00:00+00:00">
  <meta property="article:modified_time" content="2024-01-03T11:00:00+00:00">
  <meta name="author" content="Иван Петров">
  <meta name="description" content="Короткое описание статьи">
  <link rel="canonical" href="https://example.com/article">
</head><body><article>
  <h1>Заголовок статьи</h1>
  <p>Опубликовано <time datetime="2024-01-02">2 января 2024</time></p>
</article></body></html>
"""


def test_preserves_metadata_before_content() -> None:
    result = compress_html(META_HTML)
    assert 'data-web2spec-meta="1"' in result
    assert 'data-field="title"' in result
    assert 'data-field="published_at"' in result
    assert 'data-field="modified_at"' in result
    assert 'data-field="author"' in result
    assert 'data-field="canonical"' in result
    assert 'datetime="2024-01-02T10:00:00+00:00"' in result
    assert "Иван Петров" in result
    assert result.startswith('<header data-web2spec-meta="1">')


def test_keeps_datetime_attribute_in_content() -> None:
    result = compress_html(META_HTML)
    assert 'datetime="2024-01-02"' in result


def test_metadata_survives_truncation() -> None:
    html = META_HTML.replace(
        "</article>", "<p>" + "очень длинный текст " * 100 + "</p></article>"
    )
    result = compress_html(html, max_chars=500)
    assert len(result) <= 500
    assert 'data-field="published_at"' in result
    assert result.endswith(TRUNCATION_MARKER)


def test_metadata_idempotent() -> None:
    once = compress_html(META_HTML)
    assert compress_html(once) == once


def test_jsonld_metadata() -> None:
    html = """
    <html><body><script type="application/ld+json">
    {"@type":"Article","headline":"JSON заголовок","author":{"@type":"Person","name":"Пётр"},
     "datePublished":"2024-02-03","dateModified":"2024-02-04"}
    </script><article><p>Текст</p></article></body></html>
    """
    result = compress_html(html)
    assert "JSON заголовок" in result
    assert "Пётр" in result
    assert 'datetime="2024-02-03"' in result
    assert 'datetime="2024-02-04"' in result
