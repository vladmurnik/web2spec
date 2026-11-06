from __future__ import annotations

from web2spec.semantic import (
    chunks_text,
    extract_chunks,
    page_text,
    score_chunks,
    select_chunks,
)

HTML = """
<html><head>
<title>Тарифы распознавания</title>
<meta name="description" content="Цены на речь">
<meta property="article:published_time" content="2026-11-01T10:00:00Z">
<link rel="canonical" href="https://example.com/pricing">
</head><body>
<nav>меню меню меню</nav>
<main>
<h1>Тарифы</h1>
<p>Распознавание речи стоит 0,50 рубля за минуту. Это основной тариф.</p>
<h2>Дополнительно</h2>
<table>
<tr><th>Услуга</th><th>Цена</th></tr>
<tr><td>Распознавание</td><td>0,50 руб/мин</td></tr>
<tr><td>Синтез</td><td>1 руб/1000 символов</td></tr>
</table>
<p>Диаризация спикеров не описана в этом разделе.</p>
</main>
<footer>подвал</footer>
</body></html>
"""


def test_extract_chunks_keeps_metadata_and_tables() -> None:
    chunks = extract_chunks(HTML)
    assert chunks[0].kind == "metadata"
    assert "Тарифы распознавания" in chunks[0].text
    assert "2026-11-01" in chunks[0].text
    tables = [chunk for chunk in chunks if chunk.kind == "table"]
    assert len(tables) == 1
    assert "Услуга: Распознавание; Цена: 0,50 руб/мин" in tables[0].text
    all_text = page_text(chunks)
    assert "меню" not in all_text
    assert "подвал" not in all_text


def test_select_chunks_prefers_question_and_metadata() -> None:
    chunks = extract_chunks(HTML)
    selected = select_chunks(chunks, ["цена распознавания за минуту"], limit=2)
    assert selected[0].kind == "metadata"
    assert any("0,50" in chunk.text for chunk in selected[1:])
    scores = score_chunks(chunks, ["цена распознавания"])
    assert max(scores.values()) > 0


def test_select_chunks_respects_limits() -> None:
    chunks = extract_chunks(HTML)
    selected = select_chunks(chunks, ["тариф"], limit=99, max_chars=800)
    assert selected
    assert sum(len(chunk.text) for chunk in selected) <= 900 or len(selected) == 1
    assert "\n\n" in chunks_text(selected)


def test_extract_chunks_rejects_tiny_limit_and_handles_empty() -> None:
    try:
        extract_chunks(HTML, max_chars=100)
    except ValueError:
        pass
    else:
        raise AssertionError("max_chars < 300 must fail")
    assert extract_chunks("") == []
