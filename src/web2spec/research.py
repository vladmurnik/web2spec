"""Question-first research: brief -> sources -> evidence notes -> report.

The big API model supplies a goal.  A medium local model clarifies the goal,
plans diverse queries, reads semantically selected chunks and writes short
extractive notes.  The final structure is assembled deterministically from
those notes, so a fluent but careless model cannot silently invent a fact:
every note must carry a quote that exists on the page.

The module is model-agnostic.  ``run_research`` accepts any object with a
``generate(prompt, max_new_tokens=...)`` method, which keeps tests offline.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from web2spec.eval import extract_json
from web2spec.semantic import Chunk, chunks_text, extract_chunks, page_text, select_chunks
from web2spec.tokens import _manual_chatml

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[\w\u0430-\u044f\u0451-]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_TEXT_NUMBER_RE = re.compile(r"(?<![\w<>/])-?\d+(?:[.,]\d+)?(?![\w<>])")
_BLOCKED = (
    "подтвердите, что запросы отправляли вы",
    "checking your browser",
    "verify you are human",
    "enable javascript and cookies",
    "smartcaptcha",
    "access denied",
    "403 forbidden",
    "слишком много запросов",
    "проверка браузера",
)


class LLM(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str: ...


@dataclass(slots=True)
class ResearchBrief:
    goal: str
    entities: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    source_hints: list[str] = field(default_factory=list)
    output: str = ""


@dataclass(slots=True)
class Note:
    id: str
    entity: str
    field: str
    value: Any
    quote: str
    url: str
    confidence: str = "medium"
    source_value: Any = None
    unit: str | None = None
    normalized: bool = False


@dataclass(slots=True)
class Page:
    url: str
    title: str
    status: str
    error: str | None
    chunks: list[Chunk]
    selected: list[Chunk]
    text: str
    notes: list[Note] = field(default_factory=list)
    note_error: str | None = None


@dataclass(slots=True)
class ResearchConfig:
    min_pages: int = 20
    max_pages: int = 30
    max_rounds: int = 2
    queries_per_round: int = 6
    max_chunks: int = 6
    chunk_chars: int = 1400
    selected_chars: int = 7000
    fetch_timeout: float = 20.0
    delay: float = 1.0
    domains: list[str] = field(default_factory=list)
    human_only: bool = True


def clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 2 and not token.isdigit()
    }


def clamp_pages(min_pages: int, max_pages: int) -> tuple[int, int]:
    """The user's hard range: at least 20, at most 50 pages."""

    minimum = max(20, min(50, int(min_pages)))
    maximum = max(minimum, min(50, int(max_pages)))
    return minimum, maximum


def _prompt(system: str, user: str) -> str:
    return _manual_chatml(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )


def _json(llm: LLM, prompt: str, *, max_new_tokens: int) -> dict[str, Any] | None:
    raw = llm.generate(prompt, max_new_tokens=max_new_tokens)
    parsed = extract_json(raw)
    return parsed if isinstance(parsed, dict) else None


def fallback_brief(goal: str, source_hints: list[str] | None = None) -> ResearchBrief:
    entities = [part.strip() for part in re.split(r"[,;]", goal) if part.strip()][:4]
    if not entities:
        entities = ["general"]
    return ResearchBrief(
        goal=goal,
        entities=entities,
        questions=[goal, f"{goal}: официальные данные и цифры"],
        source_hints=source_hints or [],
        output="Сравнить найденные значения, показать источники и расхождения.",
    )


def build_brief(
    goal: str,
    llm: LLM,
    *,
    source_hints: list[str] | None = None,
) -> ResearchBrief:
    hints = source_hints or []
    prompt = _prompt(
        "Ты исследовательский аналитик. Разбери задачу на проверяемые вопросы. "
        "Не отвечай на задачу и не выдумывай факты. Верни только JSON.",
        f"""Цель: {goal}
Подсказки об источниках: {', '.join(hints) or 'нет'}

Верни JSON:
{{
  "entities": ["сущности, которые надо сравнить"],
  "questions": ["3-8 конкретных проверяемых вопросов"],
  "source_hints": ["официальные домены или типы страниц"],
  "output": "что должно быть в итоговом отчёте"
}}
Правила: entities — объекты сравнения (компании, товары, страницы), а НЕ поля
вроде «цена», «заголовок», «дата»; для списка однотипных страниц используй одну
сущность «статьи/страницы». questions — про факты, цифры, условия; source_hints —
только домены/типы источников, не выдумывай URL.""",
    )
    payload = _json(llm, prompt, max_new_tokens=420)
    if not payload:
        return fallback_brief(goal, hints)
    entities = [clean(str(item))[:120] for item in payload.get("entities", []) if str(item).strip()]
    questions = [
        clean(str(item))[:300]
        for item in payload.get("questions", [])
        if str(item).strip()
    ]
    if not entities:
        entities = fallback_brief(goal, hints).entities
    if not questions:
        questions = [goal]
    return ResearchBrief(
        goal=goal,
        entities=entities[:8],
        questions=questions[:8],
        source_hints=hints
        + [clean(str(item))[:120] for item in payload.get("source_hints", []) if str(item).strip()],
        output=clean(str(payload.get("output") or ""))[:500],
    )


def plan_queries(
    brief: ResearchBrief,
    llm: LLM,
    *,
    previous: list[str] | None = None,
    gaps: list[str] | None = None,
    limit: int = 6,
) -> list[str]:
    previous = previous or []
    prompt = _prompt(
        "Ты планируешь поиск. Составь разнообразные поисковые запросы, "
        "которые найдут официальные страницы с проверяемыми данными. "
        "Верни только JSON.",
        f"""Цель: {brief.goal}
Сущности: {', '.join(brief.entities)}
Вопросы: {'; '.join(brief.questions)}
Уже искали: {'; '.join(previous) or 'нет'}
Пробелы: {'; '.join(gaps or []) or 'нет'}
Источники: {', '.join(brief.source_hints) or 'не заданы'}

Верни JSON {{"queries": ["..."]}}. Нужно 4-{limit} запросов.
Добавляй site: для известных официальных доменов. Не повторяй уже искавшееся.""",
    )
    payload = _json(llm, prompt, max_new_tokens=320)
    queries: list[str] = []
    if payload and isinstance(payload.get("queries"), list):
        queries = [
            clean(str(item))[:240]
            for item in payload["queries"]
            if isinstance(item, str) and item.strip()
        ]
    if not queries:
        queries = [brief.goal]
        queries += [f"{entity} официальные тарифы" for entity in brief.entities[:3]]
        queries += [f"{question} официальный сайт" for question in brief.questions[:2]]
    result: list[str] = []
    lowered = {item.casefold() for item in previous}
    for query in queries:
        if query.casefold() in lowered:
            continue
        lowered.add(query.casefold())
        result.append(query)
    return result[:limit]


def _normalize_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return ""
    query = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_")
    ]
    return urllib.parse.urlunsplit(
        (
            parts.scheme.casefold(),
            parts.netloc.casefold(),
            parts.path.rstrip("/") or "/",
            urllib.parse.urlencode(query),
            "",
        )
    )


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.casefold().split(":", 1)[0]


def _domain_hint(value: str) -> str:
    """Accept ``example.com``, ``https://example.com/path`` or ``www.example.com``."""

    text = str(value or "").strip().casefold().rstrip("/")
    text = (
        urllib.parse.urlsplit(text).netloc
        if "//" in text
        else text.split("/", 1)[0]
    )
    text = text.split(":", 1)[0]
    return text[4:] if text.startswith("www.") else text


def _clean_domains(domains: list[str] | None) -> list[str]:
    result: list[str] = []
    for value in domains or []:
        domain = _domain_hint(value)
        if domain and domain not in result:
            result.append(domain)
    return result


def _domain_allowed(url: str, domains: list[str]) -> bool:
    if not domains:
        return True
    host = _host(url)
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def _rank_sources(
    candidates: list[dict[str, Any]],
    brief: ResearchBrief,
    *,
    limit: int,
    domains: list[str] | None = None,
    per_domain: int = 7,
) -> list[dict[str, Any]]:
    query = " ".join([brief.goal, *brief.entities, *brief.questions])
    wanted = tokens(query)
    hints = {_domain_hint(hint) for hint in brief.source_hints if _domain_hint(hint)}
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        url = _normalize_url(str(candidate.get("url") or ""))
        if not url or url in seen:
            continue
        if domains and not _domain_allowed(url, domains):
            continue
        seen.add(url)
        text = f"{candidate.get('title') or ''} {candidate.get('snippet') or ''}"
        overlap = len(wanted & tokens(text)) / (len(wanted) or 1)
        score = float(candidate.get("score") or 0.0) + 1.2 * overlap
        host = _host(url)
        if host in hints or any(host.endswith("." + hint) for hint in hints):
            score += 0.8
        path = urllib.parse.urlsplit(url).path.casefold()
        if any(word in path for word in ("tarif", "pricing", "price", "docs", "official")):
            score += 0.25
        if any(word in path for word in ("blog", "market", "s3", "on-premises")):
            score -= 0.35
        ranked.append({**candidate, "url": url, "score": round(score, 6)})
    ranked.sort(key=lambda row: (-float(row["score"]), row["url"]))
    quota: dict[str, int] = {}
    result: list[dict[str, Any]] = []
    for row in ranked:
        host = _host(row["url"])
        if quota.get(host, 0) >= per_domain:
            continue
        quota[host] = quota.get(host, 0) + 1
        result.append(row)
        if len(result) >= limit:
            break
    return result


def detect_blocked(text: str, html: str = "") -> str | None:
    lowered = text.casefold()
    for phrase in _BLOCKED:
        if phrase in lowered:
            return f"blocked page marker: {phrase}"
    if len(text) < 320 and "javascript" in lowered and "browser" in lowered:
        return "blocked or JavaScript-only page"
    if not text.strip():
        return "empty page text"
    return None


def ground_quote(text: str, quote: Any) -> str | None:
    """Return the exact page substring for a model quote, or ``None``."""

    if not isinstance(quote, str):
        return None
    needle = clean(quote)
    if len(needle) < 3:
        return None
    if needle in text:
        return needle
    parts = [part for part in needle.split() if part]
    if not parts:
        return None
    pattern = r"\s+".join(re.escape(part) for part in parts)
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0) if match else None


def _quote_for_value(text: str, value: Any, limit: int = 320) -> str | None:
    """Find an exact sentence/line that contains a scalar value.

    Used only when the model's own quote cannot be grounded; the returned text
    is still a verbatim page substring, so evidence stays auditable.
    """

    if isinstance(value, bool) or value is None:
        return None
    needle = clean(str(value))
    match = (
        re.search(re.escape(needle), text, re.IGNORECASE)
        if len(needle) >= 4
        else None
    )
    if match is None:
        try:
            target = float(needle.replace(" ", "").replace(",", "."))
        except ValueError:
            return None
        for number in _TEXT_NUMBER_RE.finditer(text):
            try:
                found = float(number.group(0).replace(",", "."))
            except ValueError:
                continue
            if abs(found - target) <= 1e-9:
                match = number
                break
    if match is None:
        return None
    start_positions = [
        index
        for index in (text.rfind("\n", 0, match.start()), text.rfind(". ", 0, match.start()))
        if index >= 0
    ]
    start = min(start_positions) if start_positions else 0
    end_candidates = [
        index
        for index in (text.find("\n", match.end()), text.find(". ", match.end()))
        if index >= 0
    ]
    end = (min(end_candidates) + 1) if end_candidates else len(text)
    quote = clean(text[start:end])
    return quote[:limit] if len(quote) >= 3 else None


def _numbers(text: str) -> list[float]:
    result: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        try:
            result.append(float(match.group(0).replace(",", ".")))
        except ValueError:
            continue
    return result


def value_supported(value: Any, quote: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    text = clean(str(value))
    if not text:
        return False
    if text.casefold() in quote.casefold():
        return True
    value_numbers = _numbers(text)
    quote_numbers = _numbers(quote)
    return bool(
        value_numbers
        and quote_numbers
        and any(
            any(abs(left - right) <= 1e-6 for right in quote_numbers)
            for left in value_numbers
        )
    )


_FIELD_ALIASES = {
    "цена": "price",
    "стоимость": "price",
    "тариф": "price",
    "цена_за_час": "price_per_hour",
    "стоимость_за_час": "price_per_hour",
    "price_rub": "price",
    "price_per_hour_rub": "price_per_hour_rub",
    "бесплатный_лимит": "free_tier",
    "бесплатно": "free_tier",
    "free": "free_tier",
    "free_tier_rub": "free_tier",
    "русский_язык": "russian",
    "поддержка_русского": "russian",
    "russian_support": "russian",
    "диаризация": "diarization",
    "разделение_спикеров": "diarization",
    "speaker_diarization": "diarization",
    "тарифы_url": "pricing_url",
    "url_тарифов": "pricing_url",
}


_ENTITY_FIELD_WORDS = {
    "title",
    "author",
    "published_at",
    "date",
    "publication_date",
    "заголовок",
    "название",
    "автор",
    "дата",
    "дата_публикации",
}

_ENTITY_FIELD_TOKENS = {
    "title",
    "author",
    "date",
    "published",
    "заголовок",
    "название",
    "автор",
    "дата",
}

_GENERIC_ENTITY_TOKENS = {
    "article",
    "page",
    "habr",
    "статья",
    "статьи",
    "страница",
    "страницы",
    "публикация",
    "материал",
}

_METADATA_KEYS = {
    "title",
    "description",
    "author",
    "published_at",
    "modified_at",
    "canonical",
}


def _strip_metadata_prefix(text: str, quote: str) -> str:
    """Drop a synthetic ``field:`` prefix from a metadata evidence quote."""

    match = re.match(r"^([a-z_]{2,40}):\s+(.+)$", quote, re.IGNORECASE)
    if match and match.group(1).casefold() in _METADATA_KEYS:
        candidate = clean(match.group(2))
        grounded = ground_quote(text, candidate)
        if grounded is not None:
            return grounded
    return quote


def normalize_field(value: Any, fallback: str) -> str:
    text = re.sub(r"[^\w]+", "_", str(value or "").casefold()).strip("_")
    text = re.sub(r"_+", "_", text)[:48]
    if len(text) < 2:
        return fallback
    return _FIELD_ALIASES.get(text, text)


def canonical_entity(
    value: Any, known: list[str], page_title: str = "", page_url: str = ""
) -> str:
    """Map a model's entity label to one of the brief entities when possible.

    Small models often put the field name (``title``/``цена``) into ``entity``.
    That would merge unrelated documents, so such labels fall back to the page
    title; genuine unknown entity names are kept.
    """

    text = clean(str(value or ""))
    page = clean(page_title)[:120] or clean(page_url)[:200]
    if not text:
        return page or (known[0] if known else "general")
    folded = text.casefold()
    field_like = re.sub(r"[^\w]+", "_", folded).strip("_")
    field_tokens = {part for part in field_like.split("_") if part}
    if (
        field_like in _FIELD_ALIASES
        or field_like in _ENTITY_FIELD_WORDS
        or (len(field_tokens) <= 3 and field_tokens & _ENTITY_FIELD_TOKENS)
        or (page and len(field_tokens) <= 5 and field_tokens & _GENERIC_ENTITY_TOKENS)
    ):
        return page or (known[0] if known else "general")
    for name in known:
        if folded == clean(name).casefold():
            return name
    for name in known:
        candidate = clean(name).casefold()
        if candidate and (candidate in folded or folded in candidate):
            return name
    return text[:120]



def normalize_value(field: str, value: Any, quote: str) -> tuple[Any, str | None, bool]:
    """Derive an hourly price when the quote explicitly says per minute.

    This is deliberately narrow: only ruble prices with a minute unit are
    converted.  The original model value is kept in ``source_value`` by the
    caller, so nothing is hidden.
    """

    if not isinstance(value, (int, float, str)):
        return value, None, False
    raw = str(value).replace(" ", "").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not match:
        return value, None, False
    number = float(match.group(0))
    lowered = quote.casefold()
    ruble = "₽" in quote or "руб" in lowered
    per_minute = bool(
        re.search(r"(?:/|за|в)\s*1?\s*мин(?:ут[а-я]*|\.)?", lowered)
    )
    per_hour = bool(re.search(r"(?:/|за|в)\s*1?\s*час(?:а|ов)?", lowered))
    if ruble and per_minute and "price" in field.casefold():
        return round(number * 60, 4), "RUB/hour", True
    if ruble and per_hour:
        return number, "RUB/hour", False
    return value, None, False


def parse_notes(
    raw: str,
    *,
    page: Page,
    brief: ResearchBrief,
) -> tuple[list[Note], list[str], list[str]]:
    payload = extract_json(raw)
    if not isinstance(payload, dict):
        return [], ["model did not return a JSON object"], []
    rows = payload.get("notes")
    if not isinstance(rows, list):
        return [], ["model returned no notes array"], []
    notes: list[Note] = []
    dropped: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            dropped.append(f"note {index}: not an object")
            continue
        value = row.get("value")
        if isinstance(value, (dict, list)):
            dropped.append(f"note {index}: value is not a scalar")
            continue
        quote_raw = row.get("quote")
        quote = ground_quote(page.text, quote_raw)
        if quote is None or not value_supported(value, quote):
            alt = _quote_for_value(page.text, value)
            if alt is not None and value_supported(value, alt):
                quote = alt
        if quote is None:
            dropped.append(f"note {index}: quote not found on page")
            continue
        quote = _strip_metadata_prefix(page.text, quote)
        if not value_supported(value, quote):
            dropped.append(f"note {index}: value not supported by quote")
            continue
        entity = canonical_entity(
            row.get("entity"), brief.entities, page.title, page.url
        )
        field_name = normalize_field(row.get("field"), f"field_{index}")
        source_value = value
        value, unit, normalized = normalize_value(field_name, value, quote)
        confidence = str(row.get("confidence") or "medium").casefold()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        digest = hashlib.sha1(page.url.encode("utf-8")).hexdigest()[:6]
        notes.append(
            Note(
                id=f"n{digest}-{len(notes) + 1:03d}",
                entity=entity,
                field=field_name,
                value=value,
                quote=quote,
                url=page.url,
                confidence=confidence,
                source_value=source_value,
                unit=unit,
                normalized=normalized,
            )
        )
    gaps = [
        clean(str(item))[:300]
        for item in payload.get("gaps", [])
        if isinstance(item, str) and item.strip()
    ]
    return notes, dropped + ([] if notes else ["no valid notes"]), gaps


def notes_prompt(brief: ResearchBrief, page: Page) -> str:
    return _prompt(
        "Ты исследователь. Прочитай фрагменты страницы и выпиши только факты, "
        "которые прямо подтверждены цитатой. Верни только JSON. Не пересказывай, "
        "не вычисляй, не переводи и не добавляй того, чего нет в тексте.",
        f"""Цель: {brief.goal}
Вопросы: {'; '.join(brief.questions)}
Сущности: {', '.join(brief.entities)}
URL: {page.url}

Фрагменты:
{chunks_text(page.selected)}

Верни JSON:
{{
  "notes": [
    {{"entity": "сущность из списка или заголовок страницы",
      "field": "snake_case: title, author, price, free_tier, russian...",
      "value": "значение как на странице", "quote": "точная цитата со страницы",
      "confidence": "high|medium|low"}}
  ],
  "gaps": ["чего не хватает на странице"]
}}
Правила: entity — дословно одна из перечисленных сущностей, заголовок страницы
или general; никогда не пиши в entity название поля. quote — дословная подстрока
фрагмента; value — число/цена/дата/фраза как написано; если цена за минуту, не
пересчитывай; не выдумывай поля.
Если фактов нет, верни {{"notes": [], "gaps": ["..."]}}.""",
    )


def analyze_gaps(
    brief: ResearchBrief,
    notes: list[Note],
    llm: LLM,
    *,
    pages: int,
    min_pages: int,
) -> tuple[list[str], list[str]]:
    compact = [
        {
            "entity": note.entity,
            "field": note.field,
            "value": note.value,
            "url": note.url,
        }
        for note in notes[:50]
    ]
    prompt = _prompt(
        "Ты проверяешь полноту исследования. Найди пробелы и предложи "
        "новые запросы. Верни только JSON.",
        f"""Цель: {brief.goal}
Вопросы: {'; '.join(brief.questions)}
Сущности: {', '.join(brief.entities)}
Страниц обработано: {pages} (минимум {min_pages})
Уже найденные заметки: {json.dumps(compact, ensure_ascii=False)}

Верни JSON {{"gaps": ["..."], "queries": ["4-6 новых запросов"]}}.
Не повторяй уже найденное; предлагай официальные страницы и другие аспекты.""",
    )
    payload = _json(llm, prompt, max_new_tokens=380)
    if not payload:
        return [], []
    gaps = [
        clean(str(item))[:300]
        for item in payload.get("gaps", [])
        if isinstance(item, str) and item.strip()
    ]
    queries = [
        clean(str(item))[:240]
        for item in payload.get("queries", [])
        if isinstance(item, str) and item.strip()
    ]
    return gaps, queries


def aggregate_notes(notes: list[Note]) -> list[dict[str, Any]]:
    """Deterministically turn notes into facts and explicit conflicts."""

    by_entity: dict[str, dict[str, list[Note]]] = {}
    for note in notes:
        by_entity.setdefault(note.entity, {}).setdefault(note.field, []).append(note)
    entities: list[dict[str, Any]] = []
    for entity, fields in by_entity.items():
        facts: dict[str, Any] = {}
        conflicts: dict[str, list[dict[str, Any]]] = {}
        evidence: dict[str, list[dict[str, Any]]] = {}
        for field_name, rows in fields.items():
            grouped: dict[str, list[Note]] = {}
            for row in rows:
                key = json.dumps(row.value, ensure_ascii=False, sort_keys=True)
                grouped.setdefault(key, []).append(row)
            if len(grouped) == 1:
                rows_group = next(iter(grouped.values()))
                row = rows_group[0]
                facts[field_name] = row.value
                evidence[field_name] = [
                    {
                        "quote": item.quote,
                        "url": item.url,
                        "note_id": item.id,
                        "confidence": item.confidence,
                    }
                    for item in rows_group
                ]
            else:
                conflicts[field_name] = [
                    {
                        "value": rows_group[0].value,
                        "unit": rows_group[0].unit,
                        "sources": sorted({item.url for item in rows_group}),
                        "note_ids": [item.id for item in rows_group],
                        "evidence": [
                            {
                                "quote": item.quote,
                                "url": item.url,
                                "note_id": item.id,
                            }
                            for item in rows_group
                        ],
                    }
                    for rows_group in grouped.values()
                ]
        entities.append(
            {
                "name": entity,
                "facts": facts,
                "conflicts": conflicts,
                "evidence": evidence,
            }
        )
    return sorted(entities, key=lambda item: item["name"].casefold())


def synthesize(
    brief: ResearchBrief,
    notes: list[Note],
    llm: LLM,
    *,
    pages: int,
) -> str:
    compact = [
        {
            "id": note.id,
            "entity": note.entity,
            "field": note.field,
            "value": note.value,
            "quote": note.quote[:300],
            "url": note.url,
        }
        for note in notes[:60]
    ]
    prompt = _prompt(
        "Ты старший аналитик. Сделай короткий отчёт только по заметкам. "
        "Не добавляй новых цифр и фактов, не выдавай догадку за факт. "
        "Явно назови расхождения и пробелы. Верни только JSON.",
        f"""Цель: {brief.goal}
Вопросы: {'; '.join(brief.questions)}
Страниц: {pages}
Заметки: {json.dumps(compact, ensure_ascii=False)}

Верни JSON {{"answer": "2-5 абзацев", "gaps": ["..."], "conflicts": ["..."]}}.""",
    )
    payload = _json(llm, prompt, max_new_tokens=700)
    if not payload:
        return ""
    return clean(str(payload.get("answer") or ""))[:5000]


def _merge_candidates(
    per_query: list[tuple[str, list[dict[str, Any]]]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for _query, rows in per_query:
        for rank, row in enumerate(rows, 1):
            url = _normalize_url(str(row.get("url") or ""))
            if not url:
                continue
            entry = merged.setdefault(
                url,
                {
                    "url": url,
                    "title": str(row.get("title") or ""),
                    "snippet": str(row.get("snippet") or ""),
                    "score": 0.0,
                },
            )
            entry["score"] += 1.0 / (rank + 10)
            if not entry["title"] and row.get("title"):
                entry["title"] = str(row["title"])
            if not entry["snippet"] and row.get("snippet"):
                entry["snippet"] = str(row["snippet"])
    return list(merged.values())


def run_research(
    goal: str,
    *,
    llm: LLM,
    search_fn: Callable[[str], tuple[list[dict[str, Any]], str | None]],
    config: ResearchConfig,
    fetch_fn: Callable[[str], tuple[str | None, str | None]],
    urls: list[str] | None = None,
    source_hints: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run a bounded question-first loop and return a compact report."""

    config.min_pages, config.max_pages = clamp_pages(config.min_pages, config.max_pages)
    say = progress or (lambda _message: None)
    brief = build_brief(goal, llm, source_hints=source_hints)
    say(
        f"[brief] entities={len(brief.entities)} questions={len(brief.questions)} "
        f"pages={config.min_pages}..{config.max_pages}"
    )

    pages: list[Page] = []
    seen_pages: dict[str, Page] = {}
    queries_seen: list[str] = []
    gaps: list[str] = []
    next_queries: list[str] = []
    note_drops: list[str] = []
    stats = {
        "queries": 0,
        "candidates": 0,
        "requested": 0,
        "ok": 0,
        "blocked": 0,
        "failed": 0,
        "notes": 0,
        "rounds": 0,
    }

    initial_urls = [_normalize_url(url) for url in (urls or []) if _normalize_url(url)]
    rounds = max(1, config.max_rounds)
    for round_index in range(rounds):
        stats["rounds"] = round_index + 1
        if initial_urls:
            selected_urls = initial_urls[: config.max_pages]
            initial_urls = []
        else:
            queries = (
                plan_queries(
                    brief,
                    llm,
                    previous=queries_seen,
                    gaps=gaps,
                    limit=config.queries_per_round,
                )
                if round_index == 0
                else next_queries
            )
            if not queries:
                break
            queries_seen.extend(queries)
            stats["queries"] += len(queries)
            say(f"[round {round_index + 1}] queries={len(queries)}")
            per_query: list[tuple[str, list[dict[str, Any]]]] = []
            for query in queries:
                rows, error = search_fn(query)
                if error:
                    say(f"[search] {query!r}: {error}")
                per_query.append((query, rows))
            candidates = _merge_candidates(per_query)
            stats["candidates"] += len(candidates)
            ranked = _rank_sources(
                candidates,
                brief,
                limit=config.max_pages * 2,
                domains=config.domains,
            )
            selected_urls = [row["url"] for row in ranked]

        for url in selected_urls:
            if len(pages) >= config.max_pages:
                break
            if url in seen_pages:
                continue
            stats["requested"] += 1
            html, error = fetch_fn(url)
            if error or not html:
                page = Page(url, "", "fetch_error", error or "empty response", [], [], "")
                pages.append(page)
                seen_pages[url] = page
                stats["failed"] += 1
                say(f"[fetch] {url}: {page.error}")
                continue
            chunks = extract_chunks(html, max_chars=config.chunk_chars)
            text = page_text(chunks)
            blocked = detect_blocked(text, html)
            title = next(
                (chunk.metadata.get("title", "") for chunk in chunks if chunk.metadata),
                "",
            )
            if blocked:
                page = Page(url, title, "blocked", blocked, chunks, [], text)
                pages.append(page)
                seen_pages[url] = page
                stats["blocked"] += 1
                say(f"[blocked] {url}: {blocked}")
                continue
            has_metadata = any(chunk.metadata for chunk in chunks)
            if not chunks or (len(text) < 100 and not has_metadata):
                page = Page(url, title, "empty", "no readable text", chunks, [], text)
                pages.append(page)
                seen_pages[url] = page
                stats["failed"] += 1
                continue
            selected = select_chunks(
                chunks,
                [brief.goal, *brief.questions],
                limit=config.max_chunks,
                max_chars=config.selected_chars,
            )
            page = Page(url, title, "ok", None, chunks, selected, text)
            try:
                raw = llm.generate(notes_prompt(brief, page), max_new_tokens=640)
                page.notes, drops, page_gaps = parse_notes(
                    raw, page=page, brief=brief
                )
                note_drops.extend(f"{url}: {drop}" for drop in drops)
                gaps.extend(page_gaps)
            except Exception as exc:  # noqa: BLE001 - one page must not kill the run
                page.note_error = str(exc)[:300]
            pages.append(page)
            seen_pages[url] = page
            stats["ok"] += 1
            stats["notes"] += len(page.notes)
            say(f"[page] {url}: {len(page.notes)} notes")

        if round_index + 1 >= rounds:
            break
        ok_pages = sum(1 for page in pages if page.status == "ok")
        gaps, next_queries = analyze_gaps(
            brief,
            [note for page in pages for note in page.notes],
            llm,
            pages=ok_pages,
            min_pages=config.min_pages,
        )
        if ok_pages < config.min_pages and not next_queries:
            fallback_queries = [brief.goal] + [
                f"{entity} официальные данные" for entity in brief.entities[:4]
            ]
            next_queries = [
                query
                for query in fallback_queries
                if query.casefold() not in {item.casefold() for item in queries_seen}
            ]
        if next_queries and len(pages) < config.max_pages:
            say(f"[gaps] new queries={len(next_queries)}")
            continue
        break

    notes = [note for page in pages for note in page.notes]
    entities = aggregate_notes(notes)
    answer = synthesize(brief, notes, llm, pages=len(pages)) if notes else ""
    sources = [page.url for page in pages if page.status == "ok"]
    errors = [
        {"url": page.url, "status": page.status, "error": page.error}
        for page in pages
        if page.status != "ok"
    ]
    result = {
        "goal": goal,
        "brief": {
            "entities": brief.entities,
            "questions": brief.questions,
            "source_hints": brief.source_hints,
            "output": brief.output,
        },
        "answer": answer,
        "entities": entities,
        "notes": [
            {
                "id": note.id,
                "entity": note.entity,
                "field": note.field,
                "value": note.value,
                "source_value": note.source_value,
                "unit": note.unit,
                "quote": note.quote,
                "url": note.url,
                "confidence": note.confidence,
            }
            for note in notes
        ],
        "sources": sources,
        "errors": errors,
        "gaps": gaps,
        "stats": {
            **stats,
            "pages_requested": len(pages),
            "pages_ok": len(sources),
            "pages_shortfall": max(0, config.min_pages - len(sources)),
            "min_pages": config.min_pages,
            "max_pages": config.max_pages,
            "note_drops": note_drops[:50],
        },
    }
    return result
