from __future__ import annotations

from web2spec.research import (
    Note,
    Page,
    ResearchBrief,
    ResearchConfig,
    _quote_for_value,
    _strip_metadata_prefix,
    aggregate_notes,
    canonical_entity,
    clamp_pages,
    detect_blocked,
    ground_quote,
    normalize_value,
    parse_notes,
    run_research,
    value_supported,
)

HTML = """
<html><head><title>Тарифы</title></head><body><main>
<h1>Распознавание речи</h1>
<p>Стоимость распознавания — 0,50 рубля за минуту аудио.</p>
<p>Диаризация спикеров поддерживается в модели general.</p>
</main></body></html>
"""


class FakeLLM:
    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        return (
            '{"entities":["vendor"],"questions":["цена за час"],'
            '"queries":["тарифы"],"gaps":[],"answer":"Цена 30 рублей в час.",'
            '"notes":[{"entity":"vendor","field":"price_per_minute",'
            '"value":0.5,"quote":"Стоимость распознавания — 0,50 рубля за минуту аудио.",'
            '"confidence":"high"}]}'
        )


def _page(text: str = HTML) -> Page:
    return Page(
        url="https://example.com/pricing",
        title="Тарифы",
        status="ok",
        error=None,
        chunks=[],
        selected=[],
        text=text,
    )


def test_clamp_pages_enforces_20_to_50() -> None:
    assert clamp_pages(1, 99) == (20, 50)
    assert clamp_pages(25, 10) == (25, 25)
    assert clamp_pages(30, 40) == (30, 40)


def test_ground_quote_and_value_supported() -> None:
    assert ground_quote(HTML, "0,50 рубля за минуту") is not None
    assert ground_quote(HTML, "нет такой фразы") is None
    assert value_supported(0.5, "0,50 рубля")
    assert not value_supported(0.7, "0,50 рубля")


def test_normalize_minute_price_to_hour() -> None:
    value, unit, changed = normalize_value(
        "price_per_minute", 0.5, "0,50 рубля за минуту"
    )
    assert (value, unit, changed) == (30.0, "RUB/hour", True)
    assert normalize_value("price_per_hour", 30, "30 рублей за час") == (
        30,
        "RUB/hour",
        False,
    )


def test_parse_notes_drops_ungrounded_quote() -> None:
    brief = ResearchBrief(goal="цена", entities=["vendor"], questions=["цена"])
    notes, dropped, _ = parse_notes(
        '{"notes":[{"entity":"vendor","field":"price","value":1,"quote":"нет на странице"}]}',
        page=_page(),
        brief=brief,
    )
    assert notes == []
    assert "quote not found" in dropped[0]


def test_parse_notes_grounds_and_normalizes() -> None:
    brief = ResearchBrief(goal="цена", entities=["vendor"], questions=["цена"])
    notes, dropped, _ = parse_notes(
        '{"notes":[{"entity":"vendor","field":"price_per_minute","value":0.5,'
        '"quote":"Стоимость распознавания — 0,50 рубля за минуту аудио.","confidence":"high"}]}',
        page=_page(),
        brief=brief,
    )
    assert not dropped
    assert notes[0].value == 30.0
    assert notes[0].unit == "RUB/hour"
    assert notes[0].quote.startswith("Стоимость распознавания")


def test_aggregate_notes_surfaces_conflict() -> None:
    base = dict(
        entity="vendor",
        field="price",
        quote="цена",
        url="https://example.com",
    )
    notes = [
        Note(id="n1", value=10, **base),
        Note(id="n2", value=20, **base),
        Note(id="n3", value=True, entity="vendor", field="russian", quote="ru", url="u"),
    ]
    entities = aggregate_notes(notes)
    vendor = entities[0]
    assert "price" not in vendor["facts"]
    assert {row["value"] for row in vendor["conflicts"]["price"]} == {10, 20}
    assert vendor["facts"]["russian"] is True


def test_detect_blocked_markers_and_empty() -> None:
    assert detect_blocked("") is not None
    assert detect_blocked("Checking your browser before redirecting to site") is not None
    assert detect_blocked("Обычный текст страницы с тарифами и ценами") is None


def test_run_research_reports_honest_shortfall() -> None:
    calls = {"fetch": 0}

    def search_fn(query: str):
        return (
            [
                {"url": "https://example.com/a", "title": "A", "snippet": "s"},
                {"url": "https://example.com/b", "title": "B", "snippet": "s"},
            ],
            None,
        )

    def fetch_fn(url: str):
        calls["fetch"] += 1
        return HTML, None

    result = run_research(
        "сравни тарифы",
        llm=FakeLLM(),
        search_fn=search_fn,
        fetch_fn=fetch_fn,
        config=ResearchConfig(min_pages=1, max_pages=3, max_rounds=1),
        urls=["https://example.com/a", "https://example.com/b"],
    )
    assert result["stats"]["min_pages"] == 20
    assert result["stats"]["max_pages"] == 20
    assert result["stats"]["pages_ok"] == 2
    assert result["stats"]["pages_shortfall"] == 18
    assert calls["fetch"] == 2
    assert result["entities"][0]["facts"]["price_per_minute"] == 30.0
    assert result["answer"]


def test_parse_notes_canonicalizes_entity_and_field() -> None:
    text = "Цена распознавания — 0,50 рубля / минута."
    brief = ResearchBrief(
        goal="сравни тарифы",
        entities=["Yandex SpeechKit"],
        questions=["цена за час"],
    )
    notes, dropped, _ = parse_notes(
        '{"notes":[{"entity":"Yandex","field":"Цена","value":0.5,'
        '"quote":"Цена распознавания — 0,50 рубля / минута.","confidence":"high"}]}',
        page=_page(text),
        brief=brief,
    )
    assert not dropped
    assert notes[0].entity == "Yandex SpeechKit"
    assert notes[0].field == "price"
    assert notes[0].value == 30.0
    assert notes[0].unit == "RUB/hour"


def test_normalize_minute_price_with_slash() -> None:
    assert normalize_value("price", 0.5, "0,50 рубля / минута") == (
        30.0,
        "RUB/hour",
        True,
    )
    assert normalize_value("price", "0,50 ₽/мин", "0,50 ₽/мин") == (
        30.0,
        "RUB/hour",
        True,
    )


def test_canonical_entity_uses_page_title_instead_of_field_name() -> None:
    assert canonical_entity("заголовок", ["Habr"], "Статья про LLM") == "Статья про LLM"
    assert canonical_entity("заголовок", ["заголовок"], "Статья про LLM") == "Статья про LLM"
    assert canonical_entity("title", ["Habr"], "Статья про LLM") == "Статья про LLM"
    assert (
        canonical_entity("заголовок страницы", ["Habr"], "Статья про LLM")
        == "Статья про LLM"
    )
    assert (
        canonical_entity("дата публикации", ["Habr"], "Статья про LLM")
        == "Статья про LLM"
    )
    assert canonical_entity("статья/страница на Habr", ["Habr"], "Статья") == "Статья"
    assert (
        canonical_entity(
            "статья/страница на Habr",
            ["Habr"],
            "",
            "https://habr.com/ru/articles/1",
        )
        == "https://habr.com/ru/articles/1"
    )
    assert canonical_entity("другой сервис", ["Habr"], "Статья") == "другой сервис"


def test_quote_for_value_recovers_numeric_sentence() -> None:
    text = (
        "Распознавание речи стоит 0,50 рубля за минуту. "
        "Синтез стоит 1 рубль за 1000 символов."
    )
    quote = _quote_for_value(text, 0.5)
    assert quote is not None
    assert "0,50" in quote
    assert "1 рубль" not in quote


def test_parse_notes_recovers_ungrounded_quote_by_value() -> None:
    text = "Распознавание речи стоит 0,50 рубля за минуту."
    brief = ResearchBrief(goal="тарифы", entities=["VK Cloud"], questions=["цена"])
    notes, dropped, _ = parse_notes(
        '{"notes":[{"entity":"VK Cloud","field":"price","value":0.5,'
        '"quote":"выдуманная цитата","confidence":"high"}]}',
        page=_page(text),
        brief=brief,
    )
    assert not dropped
    assert notes[0].quote == text
    assert notes[0].value == 30.0
    assert notes[0].unit == "RUB/hour"


def test_strip_metadata_prefix_keeps_grounded_value() -> None:
    text = "Заголовок статьи\n6 часов назад"
    assert _strip_metadata_prefix(text, "published_at: 6 часов назад") == "6 часов назад"
    assert _strip_metadata_prefix(text, "title: Заголовок статьи") == "Заголовок статьи"
    assert _strip_metadata_prefix(text, "author: Иван") == "author: Иван"



