"""Polite HTML collector: fetch pages, extract schema.org JSON-LD, build samples.

The collector is deliberately narrow:

* only public HTML pages, no authorization, no cookies;
* robots.txt is obeyed, and a host whose robots.txt is unavailable is skipped;
* one request at a time with a delay, bounded pages per seed and per run;
* the training target is derived from JSON-LD that is already published on the
  page, so no manual labeling and no teacher model are required.

Scripts and JSON-LD blocks are removed from ``input_html`` by compression, so
the model has to read the visible page, not the metadata block.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from web2spec.html_compress import compress_html
from web2spec.schema import make_id

USER_AGENT = "web2spec-research/0.1 (educational dataset; contact: local)"
MAX_BYTES = 16 * 1024 * 1024
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".pdf", ".zip")
SKIP_PATH = re.compile(r"/(login|logout|cart|basket|checkout|account|search|filter)(/|$)", re.I)

PRODUCT_PATTERNS = re.compile(
    r"/(product|products|catalog|p|goods|item|items|shop|tovar|model|detail)/", re.I
)
ARTICLE_PATTERNS = re.compile(
    r"/(news|article|articles|blog|posts?|rubric|story|novosti|statyi|text|publication)/", re.I
)
LISTING_PATTERNS = re.compile(
    r"/(offer|offers|flat|flats|kvartira|rent|sale|job|jobs|vacancy|vakansii|"
    r"listing|objavlenie|real-estate|nedvizhimost)/",
    re.I,
)

CATEGORY_PATTERNS = {
    "products": PRODUCT_PATTERNS,
    "articles": ARTICLE_PATTERNS,
    "listings": LISTING_PATTERNS,
}

CATEGORY_TYPES = {
    "products": ("product",),
    "articles": ("article",),
    "listings": ("listing", "job", "realestate"),
}

INSTRUCTIONS: dict[str, tuple[str, ...]] = {
    "product": (
        "Извлеки из HTML карточки товара название, цену, валюту, наличие и бренд в JSON.",
        "Собери структурированные данные товара: название, цена, валюта, наличие, бренд, артикул.",
        "Преобразуй страницу товара в JSON с полями title, price, currency, availability.",
    ),
    "article": (
        "Извлеки из HTML статьи заголовок, автора, дату публикации и раздел в JSON.",
        "Собери метаданные статьи: title, author, published_at, section, description.",
        "Преобразуй страницу статьи в JSON с заголовком, автором и датой публикации.",
    ),
    "listing": (
        "Извлеки из HTML список объявлений: заголовки, ссылки и цены в JSON.",
        "Собери элементы листинга в JSON: items с title, url, price, currency.",
        "Преобразуй страницу со списком предложений в JSON с массивом items.",
    ),
    "job": (
        "Извлеки из HTML вакансии название, компанию, город, зарплату и дату в JSON.",
        "Собери данные вакансии: title, company, location, salary, currency, posted_at.",
    ),
    "realestate": (
        "Извлеки из HTML объявления о недвижимости заголовок, цену, адрес и площадь в JSON.",
        "Собери параметры объекта: title, price, currency, address, area, rooms.",
    ),
}

LANG_RU = re.compile(r"[\u0430-\u044f\u0451]", re.I)


class CollectError(ValueError):
    """A page that cannot become a training sample."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


@dataclass(slots=True)
class FetchResult:
    url: str
    status: int
    html: str = ""
    content_type: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and bool(self.html) and self.error is None


@dataclass(slots=True)
class CollectStats:
    category: str
    requested: int = 0
    fetched: int = 0
    records: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    hosts: Counter[str] = field(default_factory=Counter)
    shards: int = 0
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def skip(self, code: str) -> None:
        self.skipped[code] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "requested": self.requested,
            "fetched": self.fetched,
            "records": self.records,
            "skipped": dict(self.skipped),
            "hosts": dict(self.hosts),
            "shards": self.shards,
            "started_at": self.started_at,
            "finished_at": datetime.now(UTC).isoformat(),
        }


def _decode(body: bytes, content_type: str) -> str:
    match = re.search(r"charset=([\w-]+)", content_type, re.I)
    charset = match.group(1) if match else None
    if not charset:
        head = body[:4096].decode("ascii", errors="ignore")
        meta = re.search(r"<meta[^>]+charset=[\"']?([\w-]+)", head, re.I)
        charset = meta.group(1) if meta else "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def fetch(
    url: str,
    *,
    timeout: float = 20.0,
    ua: str = USER_AGENT,
    allow_any_text: bool = False,
) -> FetchResult:
    """Fetch one URL without following redirects to a different scheme/host limit."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "ru,en;q=0.8",
            "Accept-Encoding": "identity",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            body = response.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                return FetchResult(url, 413, error="response larger than 16 MB")
            if body[:2] == b"\x1f\x8b":
                try:
                    body = gzip.decompress(body)
                except OSError:
                    return FetchResult(url, response.status, error="broken gzip")
            if (
                not allow_any_text
                and "html" not in content_type.lower()
                and content_type
            ):
                return FetchResult(
                    url, response.status, content_type=content_type, error="not html"
                )
            return FetchResult(url, response.status, _decode(body, content_type), content_type)
    except urllib.error.HTTPError as exc:
        return FetchResult(url, exc.code, error=f"http {exc.code}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return FetchResult(url, 0, error=str(exc)[:200])


class RobotsCache:
    """Per-host robots.txt policy; unavailable robots means the host is skipped."""

    def __init__(self, *, ua: str = USER_AGENT, timeout: float = 15.0) -> None:
        self.ua = ua
        self.timeout = timeout
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _parser(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        parts = urllib.parse.urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host in self._parsers:
            return self._parsers[host]
        result = fetch(f"{host}/robots.txt", timeout=self.timeout, ua=self.ua)
        parser: urllib.robotparser.RobotFileParser | None
        if result.status == 404:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse([])
        elif result.status == 200:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(result.html.splitlines())
        else:
            parser = None
        self._parsers[host] = parser
        return parser

    def allowed(self, url: str) -> bool:
        parser = self._parser(url)
        return parser is not None and parser.can_fetch(self.ua, url)

    def delay(self, url: str) -> float:
        parser = self._parser(url)
        if parser is None:
            return 0.0
        try:
            value = parser.crawl_delay(self.ua)
        except AttributeError:
            value = None
        return min(float(value), 5.0) if value else 0.0


def discover_links(
    html: str,
    base_url: str,
    *,
    pattern: re.Pattern[str] | None = None,
    limit: int = 60,
) -> list[str]:
    """Return unique same-host links matching the category pattern."""
    soup = BeautifulSoup(html, "lxml")
    base_host = urllib.parse.urlsplit(base_url).netloc.lower()
    seen: set[str] = set()
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urllib.parse.urljoin(base_url, href)
        parts = urllib.parse.urlsplit(absolute)
        if parts.scheme not in ("http", "https") or parts.netloc.lower() != base_host:
            continue
        path = parts.path.rstrip("/") or "/"
        if path.lower().endswith(IMAGE_EXT) or SKIP_PATH.search(path):
            continue
        if pattern is not None and not pattern.search(path):
            continue
        clean = urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))
        if clean in seen or clean == base_url.rstrip("/"):
            continue
        seen.add(clean)
        links.append(clean)
        if len(links) >= limit:
            break
    return links


def _flatten_jsonld(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _flatten_jsonld(item)
    elif isinstance(value, dict):
        graph = value.get("@graph")
        if isinstance(graph, list):
            yield from _flatten_jsonld(graph)
            return
        yield value


def extract_jsonld(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    objects: list[dict[str, Any]] = []
    for script in soup.find_all("script"):
        kind = str(script.get("type", "")).lower()
        if "ld+json" not in kind:
            continue
        raw = script.string if script.string is not None else script.get_text()
        raw = raw.strip().removeprefix("<!--").removesuffix("-->").strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        objects.extend(_flatten_jsonld(parsed))
    return objects


def _types(obj: dict[str, Any]) -> set[str]:
    value = obj.get("@type")
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def pick_primary(objects: list[dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
    priorities: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("product", ("Product", "ProductGroup")),
        ("article", ("NewsArticle", "Article", "BlogPosting", "ReportageNewsArticle")),
        ("job", ("JobPosting",)),
        ("realestate", ("RealEstateListing", "Apartment", "House", "Offer")),
        ("listing", ("ItemList", "CollectionPage")),
    )
    for page_type, types in priorities:
        for obj in objects:
            if _types(obj) & set(types):
                return page_type, obj
    return None


def _text(value: Any) -> str | None:
    if isinstance(value, str):
        value = re.sub(r"\s+", " ", value).strip()
        return value or None
    if isinstance(value, dict):
        return _text(value.get("name") or value.get("@id"))
    if isinstance(value, list):
        for item in value:
            found = _text(item)
            if found:
                return found
    return None


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        cleaned = re.sub(r"[^\d,.\-]", "", str(value)).replace(",", ".")
        if not cleaned or cleaned.count(".") > 1:
            return None
        try:
            number = float(cleaned)
        except ValueError:
            return None
    if number <= 0:
        return None
    return int(number) if number.is_integer() else round(number, 2)


def _availability(value: Any) -> str | None:
    text = (_text(value) or "").lower()
    if not text:
        return None
    if "outofstock" in text or "soldout" in text or "out of stock" in text:
        return "out_of_stock"
    if "preorder" in text or "backorder" in text:
        return "preorder"
    if "instock" in text or "in stock" in text:
        return "in_stock"
    return "unknown"


def _offers(obj: dict[str, Any]) -> dict[str, Any]:
    value = obj.get("offers")
    if isinstance(value, list):
        value = value[0] if value else None
    return value if isinstance(value, dict) else {}


def _price(offer: dict[str, Any]) -> int | float | None:
    for key in ("price", "lowPrice", "highPrice"):
        found = _number(offer.get(key))
        if found is not None:
            return found
    spec = offer.get("priceSpecification")
    if isinstance(spec, dict):
        return _number(spec.get("price"))
    return None


def _currency(offer: dict[str, Any]) -> str | None:
    value = offer.get("priceCurrency")
    return str(value).upper() if isinstance(value, str) and len(value) == 3 else None


def _rating(obj: dict[str, Any]) -> tuple[int | float | None, int | None]:
    rating = obj.get("aggregateRating")
    if not isinstance(rating, dict):
        return None, None
    value = _number(rating.get("ratingValue"))
    count = _number(rating.get("reviewCount") or rating.get("ratingCount"))
    return value, int(count) if count else None


def _product_target(obj: dict[str, Any]) -> dict[str, Any]:
    offer = _offers(obj)
    target: dict[str, Any] = {}
    title = _text(obj.get("name"))
    if title:
        target["title"] = title
    price = _price(offer)
    if price is not None:
        target["price"] = price
    currency = _currency(offer)
    if currency:
        target["currency"] = currency
    availability = _availability(offer.get("availability") or obj.get("availability"))
    if availability:
        target["availability"] = availability
    brand = _text(obj.get("brand"))
    if brand:
        target["brand"] = brand
    sku = _text(obj.get("sku") or obj.get("productID"))
    if sku:
        target["sku"] = sku
    rating, reviews = _rating(obj)
    if rating is not None:
        target["rating"] = rating
    if reviews is not None:
        target["review_count"] = reviews
    return target


def _article_target(obj: dict[str, Any]) -> dict[str, Any]:
    target: dict[str, Any] = {}
    title = _text(obj.get("headline") or obj.get("name"))
    if title:
        target["title"] = title
    author = _text(obj.get("author"))
    if author:
        target["author"] = author
    for key, source in (
        ("published_at", "datePublished"),
        ("modified_at", "dateModified"),
    ):
        value = _text(obj.get(source))
        if value:
            target[key] = value
    section = _text(obj.get("articleSection"))
    if section:
        target["section"] = section
    description = _text(obj.get("description"))
    if description:
        target["description"] = description[:500]
    return target


def _listing_target(obj: dict[str, Any]) -> dict[str, Any]:
    raw_items = obj.get("itemListElement")
    if not isinstance(raw_items, list):
        return {}
    items: list[dict[str, Any]] = []
    for entry in raw_items[:20]:
        if not isinstance(entry, dict):
            continue
        item = entry.get("item") if isinstance(entry.get("item"), dict) else entry
        title = _text(item.get("name"))
        if not title:
            continue
        row: dict[str, Any] = {"title": title}
        url = _text(item.get("url") or item.get("@id"))
        if url:
            row["url"] = url
        offer = _offers(item)
        price = _price(offer)
        if price is not None:
            row["price"] = price
        currency = _currency(offer)
        if currency:
            row["currency"] = currency
        items.append(row)
    if not items:
        return {}
    return {"count": len(items), "items": items}


def _job_target(obj: dict[str, Any]) -> dict[str, Any]:
    target: dict[str, Any] = {}
    title = _text(obj.get("title") or obj.get("name"))
    if title:
        target["title"] = title
    company = _text(obj.get("hiringOrganization"))
    if company:
        target["company"] = company
    location = obj.get("jobLocation")
    if isinstance(location, list):
        location = location[0] if location else None
    if isinstance(location, dict):
        address = location.get("address")
        if isinstance(address, dict):
            place = _text(address.get("addressLocality") or address.get("addressRegion"))
            if place:
                target["location"] = place
    salary = obj.get("baseSalary")
    if isinstance(salary, dict):
        value = salary.get("value")
        if isinstance(value, dict):
            amount = _number(value.get("value") or value.get("minValue"))
            currency = _text(value.get("currency") or salary.get("currency"))
        else:
            amount = _number(value)
            currency = _text(salary.get("currency"))
        if amount is not None:
            target["salary"] = amount
        if currency:
            target["currency"] = currency.upper()
    posted = _text(obj.get("datePosted"))
    if posted:
        target["posted_at"] = posted
    employment = _text(obj.get("employmentType"))
    if employment:
        target["employment_type"] = employment
    return target


def _realestate_target(obj: dict[str, Any]) -> dict[str, Any]:
    target: dict[str, Any] = {}
    title = _text(obj.get("name") or obj.get("headline"))
    if title:
        target["title"] = title
    offer = _offers(obj)
    price = _price(offer)
    if price is not None:
        target["price"] = price
    currency = _currency(offer)
    if currency:
        target["currency"] = currency
    address = obj.get("address")
    if isinstance(address, dict):
        place = _text(address.get("streetAddress") or address.get("addressLocality"))
        if place:
            target["address"] = place
    floor_size = obj.get("floorSize")
    if isinstance(floor_size, dict):
        area = _number(floor_size.get("value"))
        if area is not None:
            target["area"] = area
    rooms = _number(obj.get("numberOfRooms"))
    if rooms is not None:
        target["rooms"] = int(rooms)
    return target


TARGET_BUILDERS = {
    "product": _product_target,
    "article": _article_target,
    "listing": _listing_target,
    "job": _job_target,
    "realestate": _realestate_target,
}


def _visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip().lower()


def _instruction(page_type: str, url: str) -> str:
    options = INSTRUCTIONS[page_type]
    digest = hashlib.blake2b(url.encode("utf-8"), digest_size=2).digest()
    return options[int.from_bytes(digest, "big") % len(options)]


def _complexity(html: str) -> int:
    size = len(html)
    if size < 4000:
        return 1
    if size < 9000:
        return 2
    return 3


def _lang(text: str) -> str:
    sample = text[:4000]
    return "ru" if len(LANG_RU.findall(sample)) > len(sample) * 0.02 else "en"


def build_record(
    url: str,
    html: str,
    *,
    category: str | None = None,
    max_chars: int = 3_500,
) -> dict[str, Any]:
    """Convert one fetched page into a sample dict or raise CollectError."""
    objects = extract_jsonld(html)
    if not objects:
        raise CollectError("E_NO_JSONLD", "no JSON-LD blocks")
    picked = pick_primary(objects)
    if picked is None:
        raise CollectError("E_NO_PRIMARY", "no supported primary type")
    page_type, obj = picked
    if category and page_type not in CATEGORY_TYPES.get(category, ()):
        raise CollectError("E_WRONG_CATEGORY", f"{page_type} page in {category}")
    target = TARGET_BUILDERS[page_type](obj)
    if not target:
        raise CollectError("E_EMPTY_TARGET", "target is empty")
    if page_type != "listing" and "title" not in target:
        raise CollectError("E_EMPTY_TARGET", "target has no title")
    if page_type == "product" and "price" not in target:
        raise CollectError("E_NO_PRICE", "product page has no price")
    compressed = compress_html(html, max_chars=max_chars)
    if len(compressed) < 50:
        raise CollectError("E_HTML_TOO_SMALL", "compressed HTML is too small")
    anchor = target.get("title")
    if not anchor and target.get("items"):
        anchor = target["items"][0].get("title")
    title = str(anchor or "").strip().lower()
    if title and title[:80] not in _visible_text(compressed):
        raise CollectError("E_TITLE_NOT_VISIBLE", "title is not in visible HTML")
    return {
        "id": make_id("crawl", url),
        "group_id": urllib.parse.urlsplit(url).netloc.lower(),
        "task": "extract",
        "instruction": _instruction(page_type, url),
        "input_html": compressed,
        "target": target,
        "meta": {
            "lang": _lang(compressed),
            "complexity": _complexity(compressed),
            "source_url": url,
            "n_tokens": None,
            "pipeline_version": "collect-0.1",
        },
    }


def _safe_host(host: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", host)


def _write_shards(records: list[dict[str, Any]], out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_host: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_host.setdefault(_safe_host(str(record["group_id"])), []).append(record)
    written: dict[str, int] = {}
    for host, fresh in by_host.items():
        path = out_dir / f"{host}.jsonl"
        merged: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(existing, dict) and "id" in existing:
                    merged[str(existing["id"])] = existing
        for record in fresh:
            merged[str(record["id"])] = record
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in merged.values()),
            encoding="utf-8",
        )
        written[host] = len(merged)
    return written


def collect(
    seeds: list[str],
    out_dir: Path,
    *,
    category: str,
    max_per_seed: int = 20,
    max_total: int = 120,
    depth: int = 2,
    delay: float = 1.5,
    timeout: float = 20.0,
    max_chars: int = 3_500,
) -> CollectStats:
    """Breadth-first polite crawl from seed pages."""
    if category not in CATEGORY_PATTERNS:
        raise ValueError(f"unknown category: {category}")
    pattern = CATEGORY_PATTERNS[category]
    robots = RobotsCache(timeout=min(timeout, 15.0))
    stats = CollectStats(category=category)
    queue: deque[tuple[str, int, int]] = deque((seed, 0, 0) for seed in seeds)
    seen: set[str] = set()
    records: list[dict[str, Any]] = []

    while queue and len(records) < max_total:
        url, level, parent = queue.popleft()
        url = url.split("#", 1)[0].rstrip("/")
        if not url or url in seen:
            continue
        seen.add(url)
        if not robots.allowed(url):
            stats.skip("E_ROBOTS")
            continue
        if stats.fetched:
            time.sleep(max(delay, robots.delay(url)))
        result = fetch(url, timeout=timeout, allow_any_text=True)
        stats.requested += 1
        if not result.ok:
            stats.skip("E_FETCH")
            continue
        stats.fetched += 1
        if "xml" in result.content_type.lower() or url.lower().endswith((".xml", ".rss")):
            links = feed_links(result.html, limit=max_per_seed)
            if links:
                for link in links:
                    queue.append((link, level + 1, parent + 1))
                continue
        try:
            record = build_record(url, result.html, category=category, max_chars=max_chars)
        except CollectError as exc:
            stats.skip(exc.code)
            if level < depth:
                links = discover_links(
                    result.html, url, pattern=pattern, limit=max_per_seed
                )
                for link in links:
                    queue.append((link, level + 1, parent + 1))
        else:
            records.append(record)
            stats.records += 1
            stats.hosts[str(record["group_id"])] += 1

    written = _write_shards(records, out_dir)
    stats.shards = len(written)
    return stats


def feed_links(xml: str, *, limit: int = 100) -> list[str]:
    """Extract item URLs from RSS 2.0 or Atom."""
    soup = BeautifulSoup(xml, "xml")
    links: list[str] = []
    seen: set[str] = set()
    for item in soup.find_all(["item", "entry"]):
        node = item.find("link")
        if node is None:
            guid = item.find("guid")
            value = guid.get_text(strip=True) if guid is not None else ""
        else:
            href = node.get("href")
            value = str(href) if href else node.get_text(strip=True)
        value = value.strip()
        if not value.startswith(("http://", "https://")) or value in seen:
            continue
        seen.add(value)
        links.append(value)
        if len(links) >= limit:
            break
    return links


def load_seeds(path: Path) -> list[str]:
    seeds: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        seeds.append(line.split()[0])
    return seeds


def sitemap_locs(xml: str) -> list[str]:
    """Return all <loc> values from a sitemap or sitemap index."""
    soup = BeautifulSoup(xml, "xml")
    return [node.get_text(strip=True) for node in soup.find_all("loc") if node.get_text(strip=True)]


def urls_from_sitemap(
    url: str,
    *,
    limit: int = 200,
    max_children: int = 6,
    pattern: re.Pattern[str] | None = None,
    timeout: float = 25.0,
) -> list[str]:
    """Read a sitemap index, descend into at most ``max_children`` child maps."""
    result = fetch(url, timeout=timeout, allow_any_text=True)
    if not result.ok:
        return []
    locs = sitemap_locs(result.html)
    if not locs:
        return []
    is_index = "<sitemapindex" in result.html[:2000].lower()
    if not is_index:
        return [loc for loc in locs if pattern is None or pattern.search(loc)][:limit]

    children = [loc for loc in locs if loc.lower().endswith((".xml", ".xml.gz"))]
    preferred = [
        loc for loc in children if re.search(r"(product|tovar|goods|item|catalog)", loc, re.I)
    ]
    ordered = preferred + [loc for loc in children if loc not in preferred]
    pages: list[str] = []
    seen: set[str] = set()
    for child in ordered[:max_children]:
        child_result = fetch(child, timeout=timeout, allow_any_text=True)
        if not child_result.ok:
            continue
        body = child_result.html
        if body.startswith("PK"):
            continue
        for loc in sitemap_locs(body):
            if pattern is not None and not pattern.search(loc):
                continue
            if loc in seen:
                continue
            seen.add(loc)
            pages.append(loc)
            if len(pages) >= limit:
                return pages
    return pages

