"""Collect structured facts from a user-supplied task + URL list.

The big model supplies the search results (URLs); this CLI does the boring
part locally and never sends HTML to the big model:

    URL list -> polite fetch -> compress_html -> planner -> extractor
             -> repair -> strict validate -> JSON report

Only one merged model is resident at a time (4 GB GPU): all planner prompts are
answered first, the planner is unloaded, then all extractor prompts are
answered.  Plan errors and FactSet errors are recorded per page instead of
aborting the run.

Offline mode (``--html-dir``) maps a URL to a local file by
``sha1(url).html`` or by the URL's last path segment; nothing is fetched and
robots.txt is not consulted.  It exists for tests and for re-running saved
pages.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from web2spec.collect import RobotsCache, fetch  # noqa: E402
from web2spec.eval import extract_json  # noqa: E402
from web2spec.html_compress import compress_html  # noqa: E402
from web2spec.plan import (  # noqa: E402
    FactSet,
    PlanError,
    load_factset_lenient,
    load_plan,
    repair_factset,
    validate_factset,
)
from web2spec.schema import Meta, Sample, make_id  # noqa: E402
from web2spec.tokens import (  # noqa: E402
    build_plan_extract_prompt,
    build_plan_prompt,
    load_tokenizer,
)

PIPELINE_VERSION = "collect-urls-0.1.0"
MIN_HTML_CHARS = 50


def load_url_list(path: Path) -> list[str]:
    """Read one URL per line; blank lines and ``#`` comments are ignored."""

    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line.split()[0])
    return urls


def collect_urls(paths: list[Path], extra: list[str]) -> list[str]:
    """Merge ``--urls`` files and repeated ``--url`` values, deduplicated in order."""

    urls: list[str] = []
    for path in paths:
        urls.extend(load_url_list(path))
    urls.extend(extra)
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def local_html_path(html_dir: Path, url: str) -> Path | None:
    """Resolve an offline HTML file for ``url`` inside ``html_dir``.

    Candidates, in order: ``sha1(url).html``, the URL's last path segment,
    that segment plus ``.html``, and the percent-decoded segment variants.
    Every candidate must stay inside ``html_dir``.
    """

    root = html_dir.resolve()
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    names = [f"{digest}.html"]
    path = urllib.parse.urlsplit(url).path.rstrip("/")
    segment = path.rsplit("/", 1)[-1]
    if segment:
        decoded = urllib.parse.unquote(segment)
        for name in (segment, f"{segment}.html", decoded, f"{decoded}.html"):
            if name not in names:
                names.append(name)
    for name in names:
        candidate = (root / Path(name).name).resolve()
        if candidate.is_file() and candidate.parent == root:
            return candidate
    return None


def _empty_page(url: str) -> dict[str, Any]:
    return {
        "url": url,
        "status": "fetch_error",
        "entity": None,
        "facts": [],
        "missing": [],
        "repairs": {"evidence": [], "dropped": []},
        "dropped_foreign": [],
        "error": None,
    }


def _fact_row(fact: Any) -> dict[str, Any]:
    return {"name": fact.name, "value": fact.value, "evidence": fact.evidence}


def assemble_page(
    url: str,
    html: str,
    plan_raw: str,
    fact_raw: str,
    *,
    plan_stage_error: str | None = None,
    extract_stage_error: str | None = None,
    max_chars: int = 12_000,
) -> dict[str, Any]:
    """Deterministic part of the cascade: plan + fact JSON -> one page result.

    ``html`` is raw HTML; it is compressed here so the evidence offsets always
    refer to the text the models actually saw.
    """

    page = _empty_page(url)
    page["status"] = "ok"

    compressed = compress_html(html, max_chars=max_chars)
    if len(compressed) < MIN_HTML_CHARS:
        page["status"] = "fetch_error"
        page["error"] = "compressed HTML is empty or too small"
        return page

    if plan_stage_error is not None:
        page["status"] = "plan_error"
        page["error"] = plan_stage_error
        return page

    try:
        plan = load_plan(extract_json(plan_raw))
    except (PlanError, TypeError) as exc:
        page["status"] = "plan_error"
        page["error"] = str(exc)
        return page

    page["entity"] = plan.entity
    if plan.entity == "none" or not plan.fields:
        # Canonical empty plan: the page has nothing for this task.
        return page

    if extract_stage_error is not None:
        page["status"] = "extract_error"
        page["error"] = extract_stage_error
        return page

    try:
        pre_dropped: list[dict[str, Any]] = []
        factset = load_factset_lenient(extract_json(fact_raw), pre_dropped)
    except (PlanError, TypeError) as exc:
        page["status"] = "extract_error"
        page["error"] = str(exc)
        return page

    factset, repaired, dropped = repair_factset(factset, plan, compressed)
    page["repairs"] = {"evidence": repaired, "dropped": pre_dropped + dropped}
    for row in pre_dropped + dropped:
        if row.get("reason") in {"outside_plan", "invalid_name"}:
            page["dropped_foreign"].append(
                {
                    "name": row.get("name"),
                    "value": row.get("value"),
                    "evidence": row.get("evidence"),
                }
            )

    plan_names = {field.name for field in plan.fields}
    kept_facts = []
    for fact in factset.facts:
        if fact.name in plan_names:
            kept_facts.append(fact)
        else:
            page["dropped_foreign"].append(_fact_row(fact))
    kept_missing = [name for name in factset.missing if name in plan_names]
    factset = FactSet(facts=kept_facts, missing=kept_missing)

    try:
        validate_factset(factset, plan, compressed)
    except PlanError as exc:
        page["status"] = "extract_error"
        page["error"] = str(exc)
        page["facts"] = []
        page["missing"] = []
        return page

    page["facts"] = [_fact_row(fact) for fact in factset.facts]
    page["missing"] = list(factset.missing)
    return page


def aggregate_pages(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """One entry per requested URL, keyed by the URL as supplied."""

    aggregate: dict[str, Any] = {}
    for page in pages:
        aggregate[page["url"]] = {
            "status": page["status"],
            "entity": page["entity"],
            "facts": {fact["name"]: fact["value"] for fact in page["facts"]},
            "missing": page["missing"],
            "error": page["error"],
        }
    return aggregate


def build_report(task: str, pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap the per-page results in the stable external-agent contract."""

    ok = sum(1 for page in pages if page["status"] == "ok")
    by_status: dict[str, int] = {}
    for page in pages:
        by_status[page["status"]] = by_status.get(page["status"], 0) + 1
    return {
        "task": task,
        "generated_at": datetime.now(UTC).isoformat(),
        "pages": pages,
        "aggregate": aggregate_pages(pages),
        "stats": {
            "requested": len(pages),
            "ok": ok,
            "failed": len(pages) - ok,
            "by_status": by_status,
        },
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_causal(model_dir: str) -> tuple[Any, Any, Any]:
    """Load one merged model in float16, on GPU when available."""

    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model_dir)
    kwargs: dict[str, Any] = {"dtype": torch.float16}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
        kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.eval()
    return model, tokenizer, torch


def _generate(
    items: list[tuple[str, str]],
    samples: list[Sample],
    model_dir: str,
    build_prompt,
    max_new_tokens: int,
    label: str,
) -> dict[str, str]:
    """Run one model over a batch of ``(url, html)`` items, keyed by URL."""

    model, tokenizer, torch = _load_causal(model_dir)
    output: dict[str, str] = {}
    by_id = {sample.id: url for sample, (url, _) in zip(samples, items, strict=True)}
    try:
        for index, sample in enumerate(samples, 1):
            prompt = build_prompt(sample, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            text = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            output[by_id[sample.id]] = text
            print(
                f"[{label} {index}/{len(samples)}] {by_id[sample.id]}: {len(text)} chars",
                file=sys.stderr,
            )
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output


def generate_plans(
    items: list[tuple[str, str]], model_dir: str, max_new_tokens: int, task: str
) -> dict[str, str]:
    """Planner pass over prepared pages; returns plan JSON text per URL."""

    samples = [
        Sample(
            id=make_id("runtime-plan", url),
            group_id=urllib.parse.urlsplit(url).netloc.lower() or "offline",
            task="plan",
            instruction=task,
            input_html=html,
            target={"plan": {}},
            meta=Meta(source_url=url, pipeline_version=PIPELINE_VERSION),
        )
        for url, html in items
    ]
    return _generate(
        items, samples, model_dir, build_plan_prompt, max_new_tokens, "plan"
    )


def generate_facts(
    items: list[tuple[str, str, dict[str, Any]]],
    model_dir: str,
    max_new_tokens: int,
    task: str,
) -> dict[str, str]:
    """Extractor pass over pages that have a valid plan; FactSet JSON per URL."""

    pair_items = [(url, html) for url, html, _ in items]
    samples = [
        Sample(
            id=make_id("runtime-fact", url),
            group_id=urllib.parse.urlsplit(url).netloc.lower() or "offline",
            task="plan_extract",
            instruction=task,
            input_html=html,
            target={"facts": []},
            meta=Meta(
                source_url=url,
                pipeline_version=PIPELINE_VERSION,
                plan=plan,
            ),
        )
        for url, html, plan in items
    ]
    return _generate(
        pair_items, samples, model_dir, build_plan_extract_prompt, max_new_tokens, "fact"
    )


def _read_offline(html_dir: Path, url: str) -> tuple[str | None, str | None]:
    path = local_html_path(html_dir, url)
    if path is None:
        return None, "offline HTML file not found (sha1(url).html or last path segment)"
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except OSError as exc:
        return None, f"offline read error: {exc}"


def _fetch_online(
    url: str, robots: RobotsCache, delay: float, timeout: float, first: bool
) -> tuple[str | None, str | None]:
    if not robots.allowed(url):
        return None, "robots.txt disallows this URL or robots.txt is unavailable"
    if not first:
        time.sleep(max(delay, robots.delay(url)))
    result = fetch(url, timeout=timeout)
    if not result.ok:
        detail = result.error or f"http {result.status}"
        return None, f"fetch failed: {detail}"
    return result.html, None


def _plan_ok(plan_raw: str) -> bool:
    try:
        plan = load_plan(extract_json(plan_raw))
    except (PlanError, TypeError):
        return False
    return bool(plan.fields) and plan.entity != "none"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Задача + список URL → структурированные факты (JSON) без HTML"
    )
    parser.add_argument("--task", type=str, required=True, help="задача сбора данных")
    parser.add_argument(
        "--urls",
        type=Path,
        action="append",
        default=[],
        help="файл со списком URL (по одному в строке, # — комментарий)",
    )
    parser.add_argument(
        "--url", type=str, action="append", default=[], help="один URL (можно повторять)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data/reports/collected.json",
        help="куда записать JSON-отчёт",
    )
    parser.add_argument(
        "--planner", type=str, default=str(REPO_ROOT / "models/merged/qwen05b-plan")
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default=str(REPO_ROOT / "models/merged/qwen05b-plan-extract-v2"),
    )
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=12_000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=None,
        help="офлайн-режим: читать HTML из sha1(url).html или <последний сегмент>",
    )
    parser.add_argument("--plan-max-new-tokens", type=int, default=320)
    parser.add_argument("--fact-max-new-tokens", type=int, default=384)
    args = parser.parse_args(argv)

    task = " ".join(args.task.split())
    if len(task) < 5:
        parser.error("--task должен содержать минимум 5 символов")
    if len(task) > 500:
        parser.error("--task должен содержать не больше 500 символов")
    if args.max_pages < 1:
        parser.error("--max-pages должен быть >= 1")
    if args.max_chars < 200:
        parser.error("--max-chars должен быть >= 200")

    urls = collect_urls(args.urls, args.url)[: args.max_pages]
    if not urls:
        parser.error("нужен хотя бы один URL: --urls PATH или --url URL")

    robots = None
    if args.html_dir is None:
        robots = RobotsCache(timeout=min(args.timeout, 15.0))

    pages_by_url: dict[str, dict[str, Any]] = {}
    items: list[tuple[str, str]] = []
    fetched = 0

    for url in urls:
        page = _empty_page(url)
        pages_by_url[url] = page
        if args.html_dir is not None:
            html, error = _read_offline(args.html_dir, url)
        else:
            html, error = _fetch_online(
                url, robots, args.delay, args.timeout, first=fetched == 0
            )
            fetched += 1
        if error is not None or html is None:
            page["error"] = error or "empty response"
            print(f"[{page['status']}] {url}: {page['error']}", file=sys.stderr)
            continue
        compressed = compress_html(html, max_chars=args.max_chars)
        if len(compressed) < MIN_HTML_CHARS:
            page["error"] = "compressed HTML is empty or too small"
            print(f"[{page['status']}] {url}: {page['error']}", file=sys.stderr)
            continue
        items.append((url, compressed))
        print(f"[fetched] {url}: {len(compressed)} chars", file=sys.stderr)

    plan_raws: dict[str, str] = {}
    plan_batch_error: str | None = None
    if items:
        try:
            plan_raws = generate_plans(
                items, args.planner, args.plan_max_new_tokens, task
            )
        except Exception as exc:  # noqa: BLE001 - report, never abort the batch
            plan_batch_error = f"planner failed: {exc}"

    fact_items: list[tuple[str, str, dict[str, Any]]] = []
    for url, html in items:
        raw = plan_raws.get(url, "")
        if plan_batch_error is not None or not _plan_ok(raw):
            continue
        plan = load_plan(extract_json(raw))
        fact_items.append((url, html, plan.model_dump(mode="json")))

    fact_raws: dict[str, str] = {}
    fact_batch_error: str | None = None
    if fact_items:
        try:
            fact_raws = generate_facts(
                fact_items, args.extractor, args.fact_max_new_tokens, task
            )
        except Exception as exc:  # noqa: BLE001 - report, never abort the batch
            fact_batch_error = f"extractor failed: {exc}"

    for url, html in items:
        pages_by_url[url] = assemble_page(
            url,
            html,
            plan_raws.get(url, ""),
            fact_raws.get(url, ""),
            plan_stage_error=plan_batch_error,
            extract_stage_error=fact_batch_error,
            max_chars=args.max_chars,
        )

    pages = [pages_by_url[url] for url in urls]
    report = build_report(task, pages)
    write_report(args.out, report)

    for page in pages:
        print(
            f"[{page['status']}] {page['url']}: "
            f"{len(page['facts'])} facts, {len(page['missing'])} missing",
            file=sys.stderr,
        )
    stats = report["stats"]
    print(
        f"Готово: {stats['ok']}/{stats['requested']} страниц ok, "
        f"{stats['failed']} с ошибкой → {args.out}"
    )
    return 0 if stats["ok"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())