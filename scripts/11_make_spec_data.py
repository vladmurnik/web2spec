"""Generate a deterministic dataset for the scraping-spec task.

Each sample contains a compressed HTML page, an instruction, and a JSON spec
that the deterministic renderer can execute. The generator knows the markup it
emits, so every selector in the target is guaranteed to match the page.

Usage:
    python scripts/11_make_spec_data.py --out data/spec --count 600
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from web2spec.html_compress import compress_html  # noqa: E402
from web2spec.schema import Sample, load_task_schemas, make_id, validate_sample  # noqa: E402
from web2spec.spec import ScrapingSpec, extract  # noqa: E402

SEED = 42
BASE_URL = "https://example.ru"
MAX_CHARS = 3500
PIPELINE_VERSION = "spec-synthetic-0.1.0"

CATEGORIES = ("product", "article", "job", "realestate")
LAYOUTS = ("v1", "v2", "v3")

TITLES = {
    "product": (
        "Смартфон Aurora X5",
        "Ноутбук Vega Pro 14",
        "Наушники Pulse Air",
        "Кофемашина Terra",
    ),
    "article": ("Как выбрать ноутбук", "Рынок недвижимости в 2026", "Пять ошибок при переезде"),
    "job": ("Python-разработчик", "Аналитик данных", "Инженер по тестированию"),
    "realestate": ("2-комнатная квартира, 54 м²", "Дом 120 м² в пригороде", "Студия у метро"),
}
EXTRA = {
    "product": ("Aurora", "Vega", "Pulse", "Terra"),
    "article": ("Иван Петров", "Мария Соколова", "Алексей Орлов"),
    "job": ("ООО Вектор", "АО Север", "ТехноПлюс"),
    "realestate": ("Москва, ул. Лесная, 12", "Казань, ул. Мира, 4", "Сочи, ул. Морская, 8"),
}

INSTRUCTIONS_RU = (
    "Составь spec для сбора данных с этой страницы. Верни только JSON.",
    "Опиши селекторы и поля страницы в виде JSON-спеки.",
    "Преобразуй страницу в исполнимую spec: item_selector, fields, pagination.",
    "Сгенерируй JSON-спеку для скрапера: какие элементы и поля извлекать.",
)
INSTRUCTIONS_EN = (
    "Build a scraping spec for this page. Return JSON only.",
    "Describe the page selectors and fields as a JSON spec.",
    "Convert the page into an executable spec: item_selector, fields, pagination.",
    "Generate a JSON spec for a scraper: which elements and fields to extract.",
)


def _pick(rng: random.Random, seq: tuple[str, ...]) -> str:
    return seq[rng.randrange(len(seq))]


def _noise(rng: random.Random, category: str) -> str:
    return f"""
<header class="site-header"><nav><a href="/">Главная</a>
<a href="/catalog">Каталог</a></nav></header>
<div class="breadcrumbs"><span>Главная</span> / <span>{category}</span></div>
<script>window.__DATA__ = {{"tracking": "none"}};</script>
"""


def _footer() -> str:
    return """
<footer class="site-footer"><p>© 2026 Example. Все права защищены.</p>
<a href="/privacy">Политика</a></footer>
"""


def _item_values(rng: random.Random, category: str, index: int) -> dict[str, str]:
    title = _pick(rng, TITLES[category])
    if category == "product":
        return {
            "title": f"{title} {index + 1}",
            "url": f"/product/{1000 + index}",
            "price": f"{rng.randrange(3, 120) * 1000:,} ₽".replace(",", " "),
        }
    if category == "article":
        return {
            "title": f"{title}: часть {index + 1}",
            "url": f"/news/{2000 + index}",
            "author": _pick(rng, EXTRA["article"]),
        }
    if category == "job":
        return {
            "title": f"{title} #{index + 1}",
            "url": f"/vacancy/{3000 + index}",
            "company": _pick(rng, EXTRA["job"]),
            "salary": f"от {rng.randrange(60, 250)} 000 ₽",
        }
    return {
        "title": f"{title}, вариант {index + 1}",
        "url": f"/flat/{4000 + index}",
        "price": f"{rng.randrange(4, 30) * 1_000_000:,} ₽".replace(",", " "),
        "area": f"{rng.randrange(28, 140)} м²",
    }


def _field_classes(category: str) -> dict[str, tuple[str, str | None]]:
    """field name -> (selector, attr) for a listing item."""

    if category == "product":
        return {
            "title": ("a.title", None),
            "url": ("a.title", "href"),
            "price": ("span.price", None),
        }
    if category == "article":
        return {
            "title": ("a.title", None),
            "url": ("a.title", "href"),
            "author": ("span.author", None),
        }
    if category == "job":
        return {
            "title": ("a.title", None),
            "url": ("a.title", "href"),
            "company": ("span.company", None),
            "salary": ("span.salary", None),
        }
    return {
        "title": ("a.title", None),
        "url": ("a.title", "href"),
        "price": ("span.price", None),
        "area": ("span.area", None),
    }


def _render_listing(
    rng: random.Random, category: str, layout: str, count: int
) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    tag = {"v1": "div", "v2": "article", "v3": "li"}[layout]
    item_class = {"v1": "card", "v2": "listing-item", "v3": "row"}[layout]
    rows = [_item_values(rng, category, i) for i in range(count)]
    fields = _field_classes(category)
    parts = [
        _noise(rng, category),
        f'<main class="content {layout}"><h1>Список: {category}</h1><div class="items">',
    ]
    for row in rows:
        cells = []
        for name, (selector, attr) in fields.items():
            if name == "url":
                continue
            tag_name, _, cls = selector.partition(".")
            cls = cls or ""
            if name == "title":
                cells.append(
                    f'<{tag_name} class="{cls}" href="{html.escape(row["url"])}">'
                    f"{html.escape(row[name])}</{tag_name}>"
                )
            elif attr:
                cells.append(
                    f'<{tag_name} class="{cls}" href="{html.escape(row[name])}">'
                    f'{html.escape(row["title"])}</{tag_name}>'
                )
            else:
                cells.append(f'<{tag_name} class="{cls}">{html.escape(row[name])}</{tag_name}>')
        parts.append(f'<{tag} class="{item_class}">' + "".join(cells) + f"</{tag}>")
    parts.append('</div><a class="next" href="/page/2">Следующая</a></main>')
    parts.append(_footer())
    spec = {
        "version": 1,
        "item_selector": f"{tag}.{item_class}",
        "fields": {
            name: {"selector": selector, **({"attr": attr} if attr else {}), "required": True}
            for name, (selector, attr) in fields.items()
        },
        "pagination": {"selector": "a.next", "attr": "href", "max_pages": 3},
    }
    expected = [{key: row[key] for key in fields} for row in rows]
    for item in expected:
        item["url"] = urljoin(BASE_URL, item["url"])
    return "".join(parts), spec, expected


def _render_detail(
    rng: random.Random, category: str, layout: str
) -> tuple[str, dict[str, Any], dict[str, str]]:
    row = _item_values(rng, category, 0)
    fields = _field_classes(category)
    # A detail page has one entity: title, price/author/company, etc.
    detail_fields = {k: v for k, v in fields.items() if k != "url"}
    cells = []
    for name, (selector, _attr) in detail_fields.items():
        tag_name, _, cls = selector.partition(".")
        cells.append(f'<{tag_name} class="{cls}">{html.escape(row[name])}</{tag_name}>')
    html_text = "".join(
        [
            _noise(rng, category),
            f'<main class="detail {layout}">',
            "".join(cells),
            '<a class="back" href="/">Назад</a></main>',
            _footer(),
        ]
    )
    spec = {
        "version": 1,
        "fields": {
            name: {"selector": selector, **({"attr": attr} if attr else {}), "required": True}
            for name, (selector, attr) in detail_fields.items()
        },
    }
    expected = {key: row[key] for key in detail_fields}
    return html_text, spec, expected


def _instruction(rng: random.Random, lang: str, category: str, mode: str) -> str:
    source = INSTRUCTIONS_RU if lang == "ru" else INSTRUCTIONS_EN
    base = _pick(rng, source)
    suffix = " Собери список карточек." if mode == "listing" and lang == "ru" else ""
    if mode == "listing" and lang == "en":
        suffix = " Collect the list of cards."
    return (
        f"{base}{suffix} Категория: {category}."
        if lang == "ru"
        else f"{base}{suffix} Category: {category}."
    )


def build(count: int, out_dir: Path) -> dict[str, Any]:
    rng = random.Random(SEED)
    schemas = load_task_schemas(REPO_ROOT / "configs/tasks.yaml")
    samples: list[Sample] = []
    gold: dict[str, Any] = {}
    category_counts: Counter[str] = Counter()
    per_category = max(1, count // len(CATEGORIES))
    for category in CATEGORIES:
        for index in range(per_category):
            mode = "listing" if index % 10 < 7 else "detail"
            layout = LAYOUTS[index % len(LAYOUTS)]
            if mode == "listing":
                raw_html, spec, expected = _render_listing(
                    rng, category, layout, rng.randrange(4, 9)
                )
            else:
                raw_html, spec, expected = _render_detail(rng, category, layout)
            compressed = compress_html(raw_html, max_chars=MAX_CHARS)
            lang = "ru" if index % 2 == 0 else "en"
            instruction = _instruction(rng, lang, category, mode)
            sample_id = make_id("spec", category, layout, mode, index)
            raw = {
                "id": sample_id,
                "group_id": f"spec.{category}.{layout}.{mode}",
                "task": "spec",
                "instruction": instruction,
                "input_html": compressed,
                "target": spec,
                "meta": {
                    "lang": lang,
                    "complexity": 1 if mode == "listing" else 2,
                    "source_url": f"{BASE_URL}/{category}/{index}",
                    "n_tokens": None,
                    "pipeline_version": PIPELINE_VERSION,
                },
            }
            sample = validate_sample(raw, schemas)
            # The spec must execute against the same compressed HTML.
            got = extract(
                ScrapingSpec.model_validate(sample.target), sample.input_html, base_url=BASE_URL
            )
            if got != expected:
                raise ValueError(f"spec mismatch for {sample_id}: {got!r} != {expected!r}")
            samples.append(sample)
            gold[sample.id] = expected
            category_counts[category] += 1

    rng.shuffle(samples)
    n = len(samples)
    n_val = max(1, round(n * 0.1))
    n_test = max(1, round(n * 0.1))
    splits = {
        "train": samples[: n - n_val - n_test],
        "val": samples[n - n_val - n_test : n - n_test],
        "test": samples[n - n_test :],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gold").mkdir(exist_ok=True)
    stats: Counter[str] = Counter()
    for name, items in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in items:
                handle.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")
        gold_path = out_dir / "gold" / f"{name}.jsonl"
        with gold_path.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in items:
                handle.write(
                    json.dumps({"id": sample.id, "expected": gold[sample.id]}, ensure_ascii=False)
                    + "\n"
                )
        stats[name] = len(items)
        stats[f"{name}_chars_min"] = min(len(s.input_html) for s in items)
        stats[f"{name}_chars_max"] = max(len(s.input_html) for s in items)

    report = {
        "seed": SEED,
        "count": n,
        "splits": dict(stats),
        "by_category": dict(category_counts),
        "by_lang": dict(Counter(sample.meta.lang for sample in samples)),
    }
    report_path = out_dir / "stats.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Синтетический датасет scraping-spec")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/spec")
    parser.add_argument("--count", type=int, default=600)
    args = parser.parse_args(argv)
    report = build(args.count, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
