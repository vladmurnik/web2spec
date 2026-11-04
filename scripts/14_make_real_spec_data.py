"""Build real-page scraping specs from existing extract labels.

For each real page we know the target values (from JSON-LD). This script locates
the visible element that carries each value, derives a CSS selector, and writes a
spec + expected output. It is a bootstrap for real-world spec training, not a
hand-labelled gold set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from web2spec.html_compress import compress_html  # noqa: E402
from web2spec.schema import Sample, load_task_schemas, validate_sample  # noqa: E402
from web2spec.spec import ScrapingSpec, extract  # noqa: E402

MAX_CHARS = 3500
INSTRUCTION = "Составь spec для сбора данных с этой страницы. Верни только JSON."
SOURCES = (
    ("data/raw/articles/lenta.ru.jsonl", 90),
    ("data/raw/articles/habr.com.jsonl", 40),
    ("data/raw/listings/kaluga.hh.ru.jsonl", 60),
    ("data/raw/products/www.regard.ru.jsonl", 60),
)


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def _matches(text: str, target: Any) -> bool:
    if isinstance(target, bool) or target is None:
        return False
    if isinstance(target, (int, float)):
        digits = _digits(str(target))
        return bool(digits) and digits in _digits(text)
    target_text = _norm(str(target))
    if len(target_text) < 3:
        return False
    return target_text in _norm(text)


def _selector(element: Any, soup: BeautifulSoup) -> str | None:
    if element.get("id"):
        selector = f"#{element['id']}"
        if len(soup.select(selector)) == 1:
            return selector
    classes = [c for c in element.get("class", []) if c]
    base = element.name
    if classes:
        base += "." + ".".join(classes)
    candidates = [base]
    for cls in classes:
        candidates.append(f"{element.name}.{cls}")
    index = 1
    for sibling in element.previous_siblings:
        if getattr(sibling, "name", None) == element.name:
            index += 1
    candidates.append(f"{base}:nth-of-type({index})")
    for candidate in candidates:
        try:
            if len(soup.select(candidate)) == 1:
                return candidate
        except Exception:
            continue
    return None


PRIORITY = {
    "title": ("h1", "[itemprop=name]", "h2"),
    "price": ("[itemprop=price]", ".price", "[class*=price]"),
    "author": ("[itemprop=author]", "[rel=author]", "a[class*=author]"),
    "brand": ("[itemprop=brand]", "[class*=brand]"),
    "sku": ("[itemprop=sku]", "[class*=sku]", "[class*=article]"),
    "published_at": ("time[datetime]", "time"),
    "modified_at": ("time[datetime]", "time"),
    "company": ("[itemprop=hiringOrganization]", "[class*=company]"),
    "location": ("[itemprop=jobLocation]", "[class*=address]", "[class*=location]"),
    "salary": ("[class*=salary]", "[class*=compensation]"),
    "address": ("[itemprop=address]", "[class*=address]"),
}


def _find(soup: BeautifulSoup, target: Any, key: str = "") -> Any:
    for selector in PRIORITY.get(key, ()):
        for element in soup.select(selector):
            if _matches(element.get_text(" ", strip=True), target):
                return element
    best = None
    best_len = 10**9
    for element in soup.find_all(True):
        text = element.get_text(" ", strip=True)
        attr_match = False
        for attr in ("datetime", "content", "href", "src"):
            value = element.get(attr)
            if value and _matches(str(value), target):
                attr_match = True
                break
        if not text and not attr_match:
            continue
        if not _matches(text or str(target), target):
            continue
        length = len(text)
        if length < best_len:
            best, best_len = element, length
    return best


def build_page(
    raw: dict[str, Any], schemas: dict[str, Any]
) -> tuple[Sample, dict[str, Any]] | None:
    source = Sample.model_validate(raw)
    html = compress_html(source.input_html, max_chars=MAX_CHARS)
    soup = BeautifulSoup(html, "lxml")
    target = source.target
    if not isinstance(target, dict):
        return None
    fields: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    for key, value in target.items():
        if isinstance(value, (dict, list)) or value is None:
            continue
        element = _find(soup, value, key)
        if element is None:
            continue
        selector = _selector(element, soup)
        if selector is None:
            continue
        field: dict[str, Any] = {"selector": selector, "required": True}
        attr = None
        if key in {"published_at", "modified_at"} and element.get("datetime"):
            attr = "datetime"
        if (
            isinstance(value, str)
            and value.startswith("http")
            and element.get("href")
        ):
            attr = "href"
        if attr:
            field["attr"] = attr
        candidate = {"version": 1, "fields": {key: field}}
        got = extract(ScrapingSpec.model_validate(candidate), html)
        if not isinstance(got, dict) or not _matches(str(got.get(key, "")), value):
            continue
        fields[key] = field
        expected[key] = got[key]
    if len(fields) < 2:
        return None
    spec = {"version": 1, "fields": fields}
    if extract(ScrapingSpec.model_validate(spec), html) != expected:
        return None
    record = {
        "id": source.id,
        "group_id": source.group_id,
        "task": "spec",
        "instruction": INSTRUCTION,
        "input_html": html,
        "target": spec,
        "meta": {
            "lang": source.meta.lang,
            "complexity": 2,
            "source_url": source.meta.source_url,
            "n_tokens": None,
            "pipeline_version": "real-spec-0.1.0",
        },
    }
    return validate_sample(record, schemas), expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Spec-датасет из реальных страниц")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/spec_real")
    args = parser.parse_args(argv)
    schemas = load_task_schemas(REPO_ROOT / "configs/tasks.yaml")
    samples: list[Sample] = []
    gold: dict[str, Any] = {}
    for path, limit in SOURCES:
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for raw in rows[:limit]:
            built = build_page(raw, schemas)
            if built is None:
                continue
            sample, expected = built
            samples.append(sample)
            gold[sample.id] = expected
    rng = random.Random(42)
    rng.shuffle(samples)
    n = len(samples)
    n_test = max(1, round(n * 0.1))
    n_val = max(1, round(n * 0.1))
    splits = {
        "train": samples[: n - n_val - n_test],
        "val": samples[n - n_val - n_test : n - n_test],
        "test": samples[n - n_test :],
    }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "gold").mkdir(exist_ok=True)
    for name, items in splits.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for sample in items:
                handle.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")
        with (out / "gold" / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for sample in items:
                handle.write(
                    json.dumps({"id": sample.id, "expected": gold[sample.id]}, ensure_ascii=False)
                    + "\n"
                )
    print(json.dumps({"total": n, **{k: len(v) for k, v in splits.items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
