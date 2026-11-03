"""HTML compression: turn a real page into a compact semantic skeleton.

The goal is not readability but token efficiency: keep text and structure,
drop scripts, styles, inline handlers and decorative attributes.
"""

from __future__ import annotations

import html as html_lib
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
}

TRUNCATION_MARKER = "<!-- truncated -->"
_WS_RE = re.compile(r"\s+")
_TAG_WS_RE = re.compile(r">\s+<")
_ATTR_LIMIT = 300


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

    body = soup.body or soup
    text = str(body)
    text = _WS_RE.sub(" ", text)
    text = _TAG_WS_RE.sub("><", text).strip()

    if not text:
        return _fallback_text(html, max_chars)
    if len(text) > max_chars:
        return _truncate(text, max_chars)
    return text


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
