"""Build a deterministic bootstrap dataset for the task-conditioned planner.

One page gets several tasks: relevant intents, a broader intent, and a negative
intent that must produce an empty plan. The gold plan is derived from the fields
that already exist in the extraction label, so no teacher API is needed for the
first iteration.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from web2spec.html_compress import compress_html  # noqa: E402
from web2spec.plan import CollectionPlan, empty_plan, plan_target  # noqa: E402
from web2spec.schema import Sample, load_task_schemas, make_id, validate_sample  # noqa: E402

PIPELINE_VERSION = "plan-bootstrap-0.1.0"
PAGE_LIMIT = 20000
SCALAR_KEYS = {
    "title",
    "author",
    "description",
    "published_at",
    "modified_at",
    "section",
    "company",
    "location",
    "salary",
    "posted_at",
    "employment_type",
    "price",
    "currency",
    "availability",
    "brand",
    "sku",
    "rating",
    "review_count",
    "address",
    "area",
    "rooms",
    "count",
}

INTENTS: dict[str, tuple[str, ...]] = {
    "product": ("compare", "enrich", "filter", "negative"),
    "article": ("monitor", "research", "enrich", "negative"),
    "job": ("enrich", "filter", "monitor", "negative"),
    "realestate": ("compare", "filter", "enrich", "negative"),
}

TASKS: dict[str, dict[str, tuple[str, str]]] = {
    "product": {
        "compare": (
            "Сравни цены и характеристики товаров на этой странице.",
            "Compare prices and specifications of the products on this page.",
        ),
        "filter": (
            "Найди товары в наличии и собери их названия и цены.",
            "Find products that are in stock and collect their names and prices.",
        ),
        "enrich": (
            "Собери карточки товаров: название, бренд, цену и артикул.",
            "Collect product cards: name, brand, price, and SKU.",
        ),
        "monitor": (
            "Отслеживай изменения цен и наличия товаров.",
            "Monitor price and availability changes for products.",
        ),
        "research": (
            "Изучи ассортимент и собери ключевые характеристики товаров.",
            "Study the assortment and collect key product specifications.",
        ),
    },
    "article": {
        "monitor": (
            "Отслеживай новые статьи, их авторов и даты публикации.",
            "Track new articles, their authors, and publication dates.",
        ),
        "research": (
            "Найди статьи по теме и собери авторов, разделы и описания.",
            "Find articles on the topic and collect authors, sections, and descriptions.",
        ),
        "enrich": (
            "Собери подробные данные о статьях на странице.",
            "Collect detailed information about the articles on this page.",
        ),
        "compare": (
            "Сравни материалы по датам и тематическим разделам.",
            "Compare the materials by dates and topical sections.",
        ),
        "filter": (
            "Отбери только свежие материалы и собери их метаданные.",
            "Select only recent materials and collect their metadata.",
        ),
    },
    "job": {
        "enrich": (
            "Собери данные о вакансиях: компания, локация, зарплата.",
            "Collect job data: company, location, and salary.",
        ),
        "filter": (
            "Найди вакансии с указанной зарплатой и локацией.",
            "Find vacancies with the specified salary and location.",
        ),
        "monitor": (
            "Отслеживай новые вакансии и даты публикации.",
            "Track new vacancies and publication dates.",
        ),
        "compare": (
            "Сравни вакансии по компаниям, локациям и зарплатам.",
            "Compare vacancies by company, location, and salary.",
        ),
        "research": (
            "Изучи рынок вакансий и собери ключевые условия.",
            "Study the job market and collect key conditions.",
        ),
    },
    "realestate": {
        "compare": (
            "Сравни объекты недвижимости по цене, площади и комнатам.",
            "Compare properties by price, area, and number of rooms.",
        ),
        "filter": (
            "Найди объекты в заданном диапазоне цены и площади.",
            "Find properties in the specified price and area range.",
        ),
        "enrich": (
            "Собери подробные данные об объектах недвижимости.",
            "Collect detailed data about the real-estate properties.",
        ),
        "research": (
            "Изучи предложения и собери адреса, цены и площади.",
            "Study the listings and collect addresses, prices, and areas.",
        ),
    },
}

NEGATIVE = {
    "product": (
        "Собери данные о вакансиях с этой страницы.",
        "Collect job vacancy data from this page.",
    ),
    "article": (
        "Собери данные о квартирах с этой страницы.",
        "Collect apartment data from this page.",
    ),
    "job": (
        "Собери цены на видеокарты с этой страницы.",
        "Collect graphics-card prices from this page.",
    ),
    "realestate": (
        "Собери данные о статьях и их авторах с этой страницы.",
        "Collect article and author data from this page.",
    ),
}

GOALS = {
    "compare": ("Сравнить записи по ключевым полям", "Compare records by key fields"),
    "filter": ("Отобрать записи по условию", "Filter records by a condition"),
    "enrich": ("Собрать подробные данные о записях", "Collect detailed record data"),
    "monitor": ("Отслеживать новые записи и даты", "Monitor new records and dates"),
    "research": ("Собрать материал по теме", "Collect material on the topic"),
}

FIELD_WHY = {
    "title": "идентификация записи",
    "author": "автор материала",
    "description": "краткое содержание",
    "published_at": "дата публикации",
    "modified_at": "дата обновления",
    "section": "тематический раздел",
    "company": "работодатель",
    "location": "место работы",
    "salary": "уровень оплаты",
    "posted_at": "дата публикации вакансии",
    "employment_type": "тип занятости",
    "price": "стоимость",
    "currency": "валюта",
    "availability": "наличие",
    "brand": "бренд",
    "sku": "артикул",
    "rating": "рейтинг",
    "review_count": "количество отзывов",
    "address": "адрес объекта",
    "area": "площадь",
    "rooms": "количество комнат",
    "count": "количество записей",
}

SUBSETS = {
    "compare": {
        "title",
        "price",
        "currency",
        "availability",
        "brand",
        "rating",
        "review_count",
        "area",
        "rooms",
        "address",
    },
    "filter": {"title", "price", "availability", "location", "salary", "rooms", "area"},
    "enrich": SCALAR_KEYS,
    "monitor": {"title", "published_at", "modified_at", "posted_at", "price", "availability"},
    "research": {"title", "author", "section", "description", "published_at"},
    "negative": set(),
}


def infer_category(sample: Sample) -> str | None:
    target = sample.target
    if not isinstance(target, dict) or "items" in target:
        return None
    keys = set(target)
    if {"address", "area", "rooms"} & keys:
        return "realestate"
    if {"company", "salary", "posted_at"} & keys:
        return "job"
    if {"price", "availability", "brand", "sku"} & keys:
        return "product"
    if {"author", "description", "section"} & keys or "title" in keys:
        return "article"
    return None


def infer_kind(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if key.endswith("_at") or key in {"date"}:
        return "date"
    if key in {"url"}:
        return "url"
    return "text"


def field_entry(key: str, value: Any, required: bool) -> dict[str, Any]:
    return {
        "name": key,
        "why": FIELD_WHY.get(key, "поле для выполнения задачи"),
        "required": required,
        "kind": infer_kind(key, value),
    }


def build_plan(
    sample: Sample,
    category: str,
    intent: str,
    lang: str,
    *,
    empty: bool = False,
) -> CollectionPlan:
    target = sample.target
    assert isinstance(target, dict)
    if empty:
        return empty_plan(
            "Собрать релевантные данные" if lang == "ru" else "Collect relevant data"
        )

    available = {
        key: value
        for key, value in target.items()
        if key in SCALAR_KEYS and isinstance(value, (str, int, float, bool))
    }
    wanted = SUBSETS.get(intent, SCALAR_KEYS)
    keys = [key for key in available if key in wanted]
    if len(keys) < 2:
        keys = list(available)
    keys = keys[:12]
    if not keys:
        return empty_plan("Нет подходящих данных" if lang == "ru" else "No relevant data")

    fields = [
        field_entry(key, available[key], required=(key == "title" or index == 0))
        for index, key in enumerate(keys)
    ]
    filters: dict[str, str] = {}
    if intent == "filter" and "availability" in available:
        filters["availability"] = "in_stock"
    if intent == "filter" and category == "job" and "location" in available:
        filters["location"] = "удалённо"

    dedup = "sku" if "sku" in available else "url" if "url" in available else "title"
    if dedup not in {"url", "id", "sku", "title"}:
        dedup = "title"
    goal = GOALS[intent][0 if lang == "ru" else 1]
    plan = CollectionPlan(
        goal=goal,
        entity=category,
        entry="detail",
        fields=fields,
        filters=filters,
        pagination=False,
        detail_pages=False,
        dedup_key=dedup,
        stop_conditions=[
            "достигнут лимит страниц" if lang == "ru" else "page limit reached",
            "нет новых записей" if lang == "ru" else "no new records",
        ],
    )
    return plan


def make_record(
    sample: Sample,
    category: str,
    intent: str,
    lang: str,
    index: int,
    *,
    empty: bool = False,
) -> dict[str, Any]:
    task_pair = NEGATIVE[category] if empty else TASKS[category][intent]
    instruction = task_pair[0 if lang == "ru" else 1]
    plan = build_plan(sample, category, intent, lang, empty=empty)
    record = {
        "id": make_id("plan", sample.id, sample.group_id, category, intent),
        "group_id": sample.group_id,
        "task": "plan",
        "instruction": instruction,
        "input_html": compress_html(sample.input_html, max_chars=2800),
        "target": plan_target(plan),
        "meta": {
            "lang": lang,
            "complexity": 1 if empty else 2,
            "source_url": sample.meta.source_url,
            "n_tokens": None,
            "pipeline_version": PIPELINE_VERSION,
        },
    }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap dataset для planner")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/processed/train.jsonl")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/plan")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    schemas = load_task_schemas(REPO_ROOT / "configs/tasks.yaml")
    rng = random.Random(args.seed)

    samples: list[Sample] = []
    pages = 0
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = Sample.model_validate(json.loads(line))
        category = infer_category(sample)
        if category is None:
            continue
        pages += 1
        intents = INTENTS[category]
        for index, intent in enumerate(intents):
            lang = "ru" if (index + pages) % 2 == 0 else "en"
            empty = intent == "negative"
            raw = make_record(sample, category, intent, lang, index, empty=empty)
            samples.append(validate_sample(raw, schemas))

    # Hold out Habr entirely: unseen host for the final test.
    test = [sample for sample in samples if sample.group_id == "habr.com"]
    rest = [sample for sample in samples if sample.group_id != "habr.com"]
    rng.shuffle(rest)
    n_val = max(1, round(len(rest) * 0.1))
    val = rest[:n_val]
    train = rest[n_val:]
    splits = {"train": train, "val": val, "test": test}

    args.out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in items:
                handle.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")

    stats = {
        "pages": pages,
        "raw_plans": len(samples),
        "splits": {name: len(items) for name, items in splits.items()},
        "by_category": dict(Counter(sample.target["plan"]["entity"] for sample in samples)),
        "by_group": dict(Counter(sample.group_id for sample in samples)),
        "pipeline_version": PIPELINE_VERSION,
    }
    (args.out / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
