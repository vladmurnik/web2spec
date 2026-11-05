"""Locate exact evidence spans for gold values in compressed HTML.

Fact extraction requires every fact to carry an exact substring of the page.
Gold extraction labels are often canonicalised (ISO dates, bare numbers,
``RUB``), while the page contains human formatting (``3 ноября 2026``,
``89 560 ₽``).  This module maps the canonical value back to the formatted
span so the evidence stays a verbatim substring of the input.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

_MONTHS_RU = {
    1: r"январ[а-я]*",
    2: r"феврал[а-я]*",
    3: r"март[а-я]*",
    4: r"апрел[а-я]*",
    5: r"ма[яй]",
    6: r"июн[а-я]*",
    7: r"июл[а-я]*",
    8: r"август[а-я]*",
    9: r"сентябр[а-я]*",
    10: r"октябр[а-я]*",
    11: r"ноябр[а-я]*",
    12: r"декабр[а-я]*",
}

_MONTHS_EN = {
    1: "january",
    2: "february",
    3: "march",
    4: "april",
    5: "may",
    6: "june",
    7: "july",
    8: "august",
    9: "september",
    10: "october",
    11: "november",
    12: "december",
}

_CURRENCY = {
    "RUB": (r"₽", r"руб(?:\.|лей|ля|ль)?", r"\bр\.", r"\brub\b"),
    "USD": (r"\$", r"долл(?:ар[а-я]*)?", r"\busd\b"),
    "EUR": (r"€", r"евро", r"\beur\b"),
    "KZT": (r"₸", r"тенге", r"\bkzt\b"),
    "BYN": (r"Br", r"бел\.?\s*руб", r"\bbyn\b"),
}

_AVAILABILITY = {
    "in_stock": (r"в наличии", r"есть в наличии", r"in stock", r"\bavailable\b"),
    "out_of_stock": (
        r"нет в наличии",
        r"под заказ",
        r"out of stock",
        r"not available",
        r"ожидается",
    ),
    "preorder": (r"предзаказ", r"pre-?order"),
}

_NUMERIC_KEYS = {
    "price",
    "salary",
    "count",
    "review_count",
    "area",
    "rooms",
    "rating",
}

_DATE_KEYS = {"published_at", "modified_at", "posted_at", "date"}

_COMPANY_SUFFIX = re.compile(
    r"\b(ООО|ОАО|ЗАО|ПАО|АО|ИП|LLC|Ltd\.?|Inc\.?|Corp\.?|GmbH|S\.A\.)\b",
    re.IGNORECASE,
)

_SEP = r"(?:[\s\u00a0.,]|&nbsp;|&#160;|&#xa0;)?"
_TEXT_SEP = r"(?:\s|&nbsp;|&#160;|&#xa0;)+"


def find_evidence(html: str, value: Any, *, key: str | None = None) -> str | None:
    """Return a verbatim HTML span for ``value`` or ``None`` when absent."""

    if not html or value is None:
        return None
    if isinstance(value, bool):
        return _find_bool(html, value)
    if isinstance(value, (int, float)):
        return _find_number(html, value)

    text = str(value).strip()
    if not text:
        return None

    # Fast path: the canonical value is already present verbatim.
    if text in html:
        return text

    if key in _NUMERIC_KEYS and _looks_numeric(text):
        found = _find_number(html, float(text))
        if found:
            return found
    if key in _DATE_KEYS or _looks_iso_date(text):
        found = _find_date(html, text)
        if found:
            return found
    if key == "currency":
        found = _find_currency(html, text)
        if found:
            return found
    if key == "availability":
        found = _find_availability(html, text)
        if found:
            return found
    if key in {"location", "address"} and "," in text:
        for part in text.split(","):
            part = part.strip()
            if len(part) < 3:
                continue
            found = _find_text(html, part)
            if found:
                return found
    if key == "company":
        cleaned = _COMPANY_SUFFIX.sub(" ", text).strip(" \"'«».,")
        if cleaned and cleaned != text:
            found = _find_text(html, cleaned)
            if found:
                return found

    found = _find_text(html, text)
    if found:
        return found

    # Descriptions are frequently truncated in the compressed sample.
    if len(text) >= 40:
        for cut in (40, 60, 80, 120):
            prefix = text[:cut].rstrip(" .,;:!?-…")
            if len(prefix) < 20:
                continue
            found = _find_text(html, prefix)
            if found:
                return found

    # Currency codes sometimes appear after a formatted amount.
    found = _find_currency(html, text)
    if found:
        return found
    found = _find_availability(html, text)
    if found:
        return found
    return None


def _find_text(html: str, text: str) -> str | None:
    parts = [part for part in re.split(r"\s+", text.strip()) if part]
    if not parts:
        return None
    pattern = _TEXT_SEP.join(re.escape(part) for part in parts)
    match = re.search(pattern, html, re.IGNORECASE)
    return match.group(0) if match else None


def _find_bool(html: str, value: bool) -> str | None:
    words = (r"\btrue\b", r"\bда\b", r"\byes\b") if value else (r"\bfalse\b", r"\bнет\b", r"\bno\b")
    for pattern in words:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def _find_number(html: str, value: float) -> str | None:
    if isinstance(value, float) and not value.is_integer():
        variants = [f"{value:.2f}", f"{value:.2f}".replace(".", ",")]
    else:
        whole = int(value)
        variants = [
            str(whole),
            f"{whole:,}",
            f"{whole:,}".replace(",", " "),
            f"{whole:,}".replace(",", "\u00a0"),
            f"{whole:,}".replace(",", "."),
            f"{whole:,}".replace(",", "'"),
        ]
    for variant in dict.fromkeys(variants):
        pattern = re.escape(variant)
        pattern = pattern.replace(r"\ ", _SEP).replace(r"\,", _SEP)
        match = re.search(pattern, html)
        if match:
            return match.group(0)

    whole = str(int(value)) if float(value).is_integer() else None
    if whole and len(whole) > 3:
        pattern = r"(?<!\d)" + _SEP.join(whole) + r"(?!\d)"
        match = re.search(pattern, html)
        if match:
            return match.group(0)
    return None


def _looks_numeric(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:[.,]\d+)?", text.strip()))


def _looks_iso_date(text: str) -> bool:
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}(?:[T ].*)?$", text.strip()))


def _find_date(html: str, text: str) -> str | None:
    parsed = _parse_date(text)
    if parsed is None:
        return None
    day, month, year = parsed.day, parsed.month, parsed.year
    patterns = [
        rf"{day:02d}\.{month:02d}\.{year}",
        rf"{day}\.{month}\.{year}",
        rf"{year}-{month:02d}-{day:02d}",
        rf"{day:02d}/{month:02d}/{year}",
        rf"{day}\s+{_MONTHS_RU[month]}\s+{year}",
        rf"{day}\s+{_MONTHS_EN[month]}\s+{year}",
        rf"{_MONTHS_RU[month]}\s+{year}",
        rf"{_MONTHS_EN[month]}\s+{year}",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def _parse_date(text: str) -> dt.datetime | None:
    candidate = text.strip()
    if not candidate:
        return None
    try:
        return dt.datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        pass
    for pattern in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return dt.datetime.strptime(candidate[:10], pattern)
        except ValueError:
            continue
    return None


def _find_currency(html: str, text: str) -> str | None:
    patterns = _CURRENCY.get(text.strip().upper())
    if not patterns:
        return None
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def _find_availability(html: str, text: str) -> str | None:
    key = text.strip().lower().replace("-", "_").replace(" ", "_")
    patterns = _AVAILABILITY.get(key)
    if not patterns:
        return None
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


__all__ = ["find_evidence"]
