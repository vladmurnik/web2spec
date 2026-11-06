"""Semantic page chunks for a medium-model research pass.

The 0.5B collector works on a compressed HTML skeleton.  A 4B analyst needs
larger, readable units, but it still cannot read a whole site.  This module
turns one page into heading-aware text chunks, keeps tables as row-oriented
records (where prices usually live), and selects the chunks most relevant to a
research question with a small BM25-style scorer.

No model and no network access live here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[\w\u0430-\u044f\u0451-]+", re.IGNORECASE)
_BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "dt", "dd", "blockquote")
_SKIP_TAGS = (
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "template",
    "canvas",
    "nav",
    "footer",
    "aside",
)
_MIN_TEXT = 40


@dataclass(slots=True)
class Chunk:
    """One retrievable page fragment."""

    id: str
    kind: str
    heading: str
    text: str
    order: int
    metadata: dict[str, str] = field(default_factory=dict)

    def prompt_text(self) -> str:
        prefix = f"[{self.id}]"
        if self.heading:
            prefix += f" {self.heading}"
        return f"{prefix}\n{self.text}".strip()


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 2 and not token.isdigit()
    }


def _metadata(soup: BeautifulSoup) -> dict[str, str]:
    values: dict[str, str] = {}
    for node in soup.find_all("meta"):
        key = str(node.get("property") or node.get("name") or "").casefold()
        content = _clean(str(node.get("content") or ""))
        if not key or not content:
            continue
        if key in {"og:title", "twitter:title", "title"}:
            values.setdefault("title", content)
        elif key in {"description", "og:description"}:
            values.setdefault("description", content)
        elif key in {"author", "article:author"}:
            values.setdefault("author", content)
        elif key in {
            "article:published_time",
            "og:published_time",
            "datepublished",
            "date",
            "pubdate",
        }:
            values.setdefault("published_at", content)
        elif key in {"article:modified_time", "datemodified"}:
            values.setdefault("modified_at", content)
    if soup.title and soup.title.get_text(strip=True):
        values.setdefault("title", _clean(soup.title.get_text(" ", strip=True)))
    h1 = soup.find("h1")
    if h1:
        values.setdefault("title", _clean(h1.get_text(" ", strip=True)))
    link = soup.find("link", rel="canonical")
    if link and link.get("href"):
        values.setdefault("canonical", _clean(str(link["href"])))
    for node in soup.find_all("time"):
        value = _clean(str(node.get("datetime") or node.get_text(" ", strip=True)))
        if value:
            values.setdefault("published_at", value)
            break
    return {key: value[:500] for key, value in values.items() if value}


def _table_text(table: Any) -> str:
    headers: list[str] = []
    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        values = [_clean(cell.get_text(" ", strip=True)) for cell in cells]
        values = [value for value in values if value]
        if not values:
            continue
        if not headers and all(cell.name == "th" for cell in cells):
            headers = values
            continue
        if headers and len(headers) == len(values):
            rows.append(
                "; ".join(
                    f"{name}: {value}"
                    for name, value in zip(headers, values, strict=False)
                )
            )
        else:
            rows.append(" | ".join(values))
    caption = table.find("caption")
    prefix = _clean(caption.get_text(" ", strip=True)) if caption else ""
    body = "\n".join(rows)
    return (prefix + "\n" + body).strip() if prefix else body


def _content_root(soup: BeautifulSoup) -> Any:
    candidates = soup.find_all(["main", "article"])
    if candidates:
        return max(candidates, key=lambda node: len(node.get_text(" ", strip=True)))
    return soup.body or soup


def extract_chunks(html: str, *, max_chars: int = 1400) -> list[Chunk]:
    """Turn raw HTML into ordered metadata/text/table chunks.

    The function is deterministic and tolerant of malformed markup.  Tables are
    never merged with surrounding prose, because a price row loses its meaning
    when the header row is detached.
    """

    if max_chars < 300:
        raise ValueError("max_chars must be at least 300")
    if not html or not html.strip():
        return []
    soup = BeautifulSoup(html, "lxml")
    metadata = _metadata(soup)
    for node in soup.find_all(_SKIP_TAGS):
        node.decompose()

    chunks: list[Chunk] = []
    if metadata:
        summary = " | ".join(f"{key}: {value}" for key, value in metadata.items())
        chunks.append(
            Chunk(
                id="c000",
                kind="metadata",
                heading="page metadata",
                text=summary[:max_chars],
                order=0,
                metadata=metadata,
            )
        )

    root = _content_root(soup)
    heading = ""
    buffer: list[str] = []
    buffer_heading = ""
    order = len(chunks)

    def flush() -> None:
        nonlocal buffer, buffer_heading, order
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if text:
            chunks.append(
                Chunk(
                    id=f"c{order:03d}",
                    kind="text",
                    heading=buffer_heading,
                    text=text[:max_chars],
                    order=order,
                )
            )
            order += 1
        buffer = []
        buffer_heading = ""

    for node in root.find_all(list(_BLOCK_TAGS) + ["table"]):
        if node.name == "table":
            if node.find_parent("table") is not None:
                continue
            text = _table_text(node)
            if len(text) < 20:
                continue
            flush()
            chunks.append(
                Chunk(
                    id=f"c{order:03d}",
                    kind="table",
                    heading=heading,
                    text=text[:max_chars],
                    order=order,
                )
            )
            order += 1
            continue
        if node.name.startswith("h"):
            level = int(node.name[1])
            if level <= 3:
                text = _clean(node.get_text(" ", strip=True))
                if text:
                    flush()
                    heading = text[:300]
            continue
        if node.find_parent("table") is not None:
            continue
        if node.find(["p", "li", "table", "blockquote"]) is not None:
            continue
        text = _clean(node.get_text(" ", strip=True))
        if len(text) < _MIN_TEXT:
            continue
        if buffer and buffer_heading != heading:
            flush()
        if buffer and sum(len(item) for item in buffer) + len(text) > max_chars:
            flush()
        buffer_heading = heading
        buffer.append(text)
    flush()

    if not chunks:
        text = _clean(soup.get_text(" ", strip=True))
        if text:
            chunks.append(
                Chunk(id="c000", kind="text", heading="", text=text[:max_chars], order=0)
            )
    return chunks


def score_chunks(chunks: list[Chunk], questions: list[str]) -> dict[str, float]:
    """BM25-ish relevance with heading boosts and a small table prior."""

    query_tokens: set[str] = set()
    for question in questions:
        query_tokens |= _tokens(question)
    if not query_tokens:
        return {chunk.id: 0.0 for chunk in chunks}
    documents = {chunk.id: _tokens(f"{chunk.heading} {chunk.text}") for chunk in chunks}
    total = len(documents) or 1
    scores: dict[str, float] = {}
    for chunk in chunks:
        document = documents[chunk.id]
        score = 0.0
        for token in query_tokens:
            if token not in document:
                continue
            df = sum(1 for words in documents.values() if token in words)
            idf = math.log((total + 1) / (df + 0.5)) + 1.0
            score += idf
            if token in _tokens(chunk.heading):
                score += idf
        if chunk.kind == "table":
            score += 0.4
        if chunk.kind == "metadata":
            score += 0.1
        scores[chunk.id] = round(score, 4)
    return scores


def select_chunks(
    chunks: list[Chunk],
    questions: list[str],
    *,
    limit: int = 6,
    max_chars: int = 7000,
) -> list[Chunk]:
    """Pick the most relevant chunks, always keeping metadata first."""

    if limit < 1:
        raise ValueError("limit must be >= 1")
    scores = score_chunks(chunks, questions)
    metadata = [chunk for chunk in chunks if chunk.kind == "metadata"]
    ranked = sorted(
        (chunk for chunk in chunks if chunk.kind != "metadata"),
        key=lambda chunk: (-scores.get(chunk.id, 0.0), chunk.order),
    )
    selected: list[Chunk] = []
    used = 0
    for chunk in metadata[:1] + ranked:
        if len(selected) >= limit:
            break
        if selected and used + len(chunk.text) > max_chars:
            continue
        selected.append(chunk)
        used += len(chunk.text)
    if not selected and chunks:
        selected = chunks[:1]
    return selected


def chunks_text(chunks: list[Chunk]) -> str:
    return "\n\n".join(chunk.prompt_text() for chunk in chunks)


def page_text(chunks: list[Chunk]) -> str:
    """Plain text used to verify that a note quote really exists on the page."""

    return "\n".join(chunk.text for chunk in chunks)
