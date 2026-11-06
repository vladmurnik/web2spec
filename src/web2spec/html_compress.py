"""HTML compression: turn a real page into a compact semantic skeleton.

The goal is not readability but token efficiency: keep text and structure,
drop scripts, styles, inline handlers and decorative attributes.
"""

from __future__ import annotations

import html as html_lib
import json
import re

from bs4 import BeautifulSoup, Comment

REMOVE_TAGS = (
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "template",
    "canvas",
    "object",
    "embed",
    "link",
    "meta",
    "input",
    "select",
    "textarea",
    "button",
)

KEEP_TAGS = (
    "html",
    "body",
    "main",
    "article",
    "section",
    "header",
    "footer",
    "nav",
    "div",
    "ul",
    "ol",
    "li",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "a",
    "img",
    "span",
    "strong",
    "em",
    "b",
    "i",
    "br",
    "dl",
    "dt",
    "dd",
    "time",
)

KEEP_ATTRS = {
    "class",
    "id",
    "href",
    "src",
    "alt",
    "aria-label",
    "itemprop",
    "role",
    "type",
    "name",
    "value",
    "placeholder",
    "data-testid",
    "data-field",
    "data-web2spec-meta",
    "datetime",
}

TRUNCATION_MARKER = "<!-- truncated -->"
_WS_RE = re.compile(r"\s+")
_TAG_WS_RE = re.compile(r">\s+<")
_ATTR_LIMIT = 300
_HEADER_LIMIT = 800
_META_LIMIT = 240


def compress_html(html: str, *, max_chars: int = 12_000) -> str:
    """Return a compact, deterministic HTML skeleton no longer than ``max_chars``."""

    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    if not html or not html.strip():
        return ""

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:  # pragma: no cover - lxml is a hard dependency, keep a fallback
        soup = BeautifulSoup(html, "html.parser")

    existing_meta = _existing_metadata(soup)
    metadata = _extract_metadata(soup)
    metadata.update({key: value for key, value in existing_meta.items() if value})

    for node in soup.find_all(REMOVE_TAGS):
        node.decompose()

    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        if comment.strip() == "truncated":
            continue
        comment.extract()

    for tag in list(soup.find_all(True)):
        if tag.decomposed:
            continue
        if tag.name not in KEEP_TAGS:
            tag.unwrap()
            continue
        clean: dict[str, str] = {}
        for name, value in tag.attrs.items():
            lowered = name.lower()
            if lowered not in KEEP_ATTRS and not (
                lowered.startswith("data-") and lowered == "data-testid"
            ):
                continue
            if isinstance(value, list):
                value = " ".join(str(item) for item in value)
            text_value = str(value).strip()
            if text_value:
                clean[lowered] = text_value[:_ATTR_LIMIT]
        tag.attrs = clean

    body = _content_root(soup)
    text = str(body)
    text = _WS_RE.sub(" ", text)
    text = _TAG_WS_RE.sub("><", text).strip()
    # BOM characters occur inside page text (not just at the start); the sample
    # schema rejects them and they carry no meaning in a compressed skeleton.
    text = text.replace("\ufeff", "")
    header = _metadata_header(metadata)
    if header:
        text = header + text

    if not text:
        return _fallback_text(html, max_chars)
    if len(text) > max_chars:
        return _truncate(text, max_chars)
    return text


def _existing_metadata(soup: BeautifulSoup) -> dict[str, str]:
    """Read and remove a header produced by an earlier compression pass."""

    result: dict[str, str] = {}
    for header in soup.find_all("header", attrs={"data-web2spec-meta": "1"}):
        for node in header.find_all(attrs={"data-field": True}):
            name = str(node.get("data-field") or "").strip()
            value = node.get_text(" ", strip=True)
            if name and value and name not in result:
                result[name] = value
        header.decompose()
    return result


def _extract_metadata(soup: BeautifulSoup) -> dict[str, str]:
    """Keep high-value page metadata that is otherwise deleted with tags."""

    meta_map: dict[str, str] = {}
    for node in soup.find_all("meta"):
        key = node.get("property") or node.get("name") or node.get("itemprop")
        content = node.get("content")
        if not key or not content:
            continue
        meta_map.setdefault(str(key).lower(), _clean_value(content))

    jsonld = _jsonld_metadata(soup)
    title = (
        _first(meta_map, "og:title", "twitter:title", "title")
        or (soup.title.get_text(" ", strip=True) if soup.title else "")
        or (soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else "")
        or jsonld.get("title", "")
    )
    author = (
        _first(meta_map, "author", "article:author", "og:article:author")
        or jsonld.get("author", "")
    )
    if author.startswith(("http://", "https://")):
        author = jsonld.get("author", "")
    published = (
        _first(
            meta_map,
            "article:published_time",
            "og:published_time",
            "datepublished",
            "date",
            "pubdate",
            "publish-date",
        )
        or jsonld.get("published_at", "")
    )
    modified = (
        _first(meta_map, "article:modified_time", "datemodified", "last-modified")
        or jsonld.get("modified_at", "")
    )

    times = []
    for node in soup.find_all("time"):
        value = node.get("datetime") or ""
        if value:
            times.append(_clean_value(value))
    if not published and times:
        published = times[0]
    if not modified and len(times) > 1:
        modified = times[1]

    canonical = ""
    link = soup.find("link", rel="canonical")
    if link and link.get("href"):
        canonical = _clean_value(link["href"])

    metadata = {
        "title": title,
        "description": _first(meta_map, "description", "og:description"),
        "author": author,
        "published_at": published,
        "modified_at": modified,
        "canonical": canonical,
    }
    return {key: value for key, value in metadata.items() if value}


def _jsonld_metadata(soup: BeautifulSoup) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in soup.find_all("script"):
        if "ld+json" not in str(node.get("type") or "").lower():
            continue
        raw = node.string or node.get_text()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            continue
        _walk_jsonld(data, result)
    return result


def _walk_jsonld(value, result: dict[str, str]) -> None:
    if isinstance(value, list):
        for item in value:
            _walk_jsonld(item, result)
        return
    if not isinstance(value, dict):
        return
    if "title" not in result and value.get("headline"):
        result["title"] = _clean_value(value["headline"])
    if "author" not in result and value.get("author"):
        author = _author_name(value["author"])
        if author:
            result["author"] = _clean_value(author)
    if "published_at" not in result and value.get("datePublished"):
        result["published_at"] = _clean_value(value["datePublished"])
    if "modified_at" not in result and value.get("dateModified"):
        result["modified_at"] = _clean_value(value["dateModified"])
    for nested in value.values():
        if isinstance(nested, (dict, list)):
            _walk_jsonld(nested, result)


def _author_name(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        name = value.get("name")
        return str(name) if name else ""
    if isinstance(value, list):
        for item in value:
            name = _author_name(item)
            if name:
                return name
    return ""


def _first(mapping: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key, "")
        if value:
            return value
    return ""


def _clean_value(value) -> str:
    return _WS_RE.sub(" ", str(value)).strip()[:_META_LIMIT]


def _metadata_header(metadata: dict[str, str]) -> str:
    """Render preserved metadata before the page body so truncation keeps it."""

    fields = (
        ("title", 180),
        ("published_at", 48),
        ("modified_at", 48),
        ("author", 80),
        ("description", 180),
        ("canonical", 240),
    )
    parts: list[str] = []
    for name, limit in fields:
        value = metadata.get(name)
        if not value:
            continue
        escaped = html_lib.escape(value[:limit], quote=True)
        if name.endswith("_at"):
            parts.append(
                f'<time data-field="{name}" datetime="{escaped}">{escaped}</time>'
            )
        else:
            parts.append(f'<span data-field="{name}">{escaped}</span>')
    if not parts:
        return ""
    header = '<header data-web2spec-meta="1">' + "".join(parts) + "</header>"
    if len(header) > _HEADER_LIMIT:
        header = '<header data-web2spec-meta="1">' + "".join(parts[:4]) + "</header>"
    return header


def _content_root(soup: BeautifulSoup):
    """Pick the main content region so truncation does not cut off the product card."""
    candidates = []
    for name in ("main", "article"):
        candidates.extend(soup.find_all(name))
    if not candidates:
        candidates = [
            node
            for node in soup.find_all(attrs={"role": "main"})
            if getattr(node, "name", None)
        ]
    if candidates:
        best = max(candidates, key=lambda node: len(node.get_text(" ", strip=True)))
        if len(best.get_text(" ", strip=True)) >= 200:
            return best
    for node in soup.find_all(["nav", "header", "footer", "aside"]):
        node.decompose()
    return soup.body or soup


def _truncate(text: str, max_chars: int) -> str:
    budget = max_chars - len(TRUNCATION_MARKER)
    if budget <= 0:
        return TRUNCATION_MARKER[:max_chars]
    cut = text.rfind(">", 0, budget)
    if cut == -1:
        return text[:budget] + TRUNCATION_MARKER
    return text[: cut + 1] + TRUNCATION_MARKER


def _fallback_text(html: str, max_chars: int) -> str:
    stripped = re.sub(r"<[^>]+>", " ", html)
    stripped = html_lib.unescape(stripped)
    stripped = _WS_RE.sub(" ", stripped).strip()
    return stripped[:max_chars]
