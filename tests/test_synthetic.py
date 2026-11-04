"""Tests for the offline synthetic dataset (scripts/09_make_synthetic.py).

The tests validate the generated shards against the Sample contract, check the
category/language composition, the HTML length bounds, target visibility and
the determinism of the generator.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from web2spec.schema import Sample

REPO_ROOT = Path(__file__).resolve().parents[1]
SYN_DIR = REPO_ROOT / "data" / "raw" / "synthetic"
STATS_PATH = REPO_ROOT / "data" / "reports" / "synthetic_stats.json"
GEN_PATH = REPO_ROOT / "scripts" / "09_make_synthetic.py"

CATEGORIES = ("product", "article", "listing", "job", "realestate")
TOTAL_MIN = 500
PER_CATEGORY_MIN = TOTAL_MIN // len(CATEGORIES)
HTML_MIN = 1500
HTML_MAX = 3500
SEED = 42
RU_SHARE = 0.6

_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_NON_DIGIT_RE = re.compile(r"\D+")


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("synthetic_gen", GEN_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {GEN_PATH}"
    module = importlib.util.module_from_spec(spec)
    # dataclass(slots=True) resolves annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = _load_generator()


def _read_shard(category: str) -> list[dict[str, Any]]:
    path = SYN_DIR / f"{category}.jsonl"
    assert path.is_file(), f"missing shard: {path}"
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


@pytest.fixture(scope="module")
def shards() -> dict[str, list[dict[str, Any]]]:
    return {cat: _read_shard(cat) for cat in CATEGORIES}


@pytest.fixture(scope="module")
def records(shards: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for cat in CATEGORIES for row in shards[cat]]


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    assert STATS_PATH.is_file(), f"missing report: {STATS_PATH}"
    return json.loads(STATS_PATH.read_text(encoding="utf-8"))


def _visible(html: str) -> str:
    """Strip tags and collapse whitespace, mirroring the generator's check."""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", html)).strip()


def _missing_leaves(target: Any, html: str, path: str = "") -> list[str]:
    """Return dotted paths of target leaves that are absent from the HTML."""

    digits = _NON_DIGIT_RE.sub("", html)
    lowered = html.lower()
    missing: list[str] = []

    def walk(node: Any, node_path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{node_path}.{key}" if node_path else str(key))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{node_path}[{i}]")
        elif isinstance(node, bool) or node is None:
            return
        elif isinstance(node, int):
            if str(node) not in digits:
                missing.append(node_path)
        elif isinstance(node, float):
            variants = (str(node), str(node).replace(".", ","))
            if not any(variant in html for variant in variants):
                missing.append(node_path)
        elif isinstance(node, str) and node.lower() not in lowered:
            missing.append(node_path)

    walk(target, path)
    return missing


def _anchor(target: Any) -> str | None:
    if not isinstance(target, dict):
        return None
    anchor = target.get("title")
    if not anchor and target.get("items"):
        anchor = target["items"][0].get("title")
    return str(anchor) if anchor else None


def _fingerprint(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            row["id"],
            row["group_id"],
            row["instruction"],
            row["input_html"],
            row["target"],
            row["meta"]["lang"],
            row["meta"]["complexity"],
        )
        for row in rows
    ]


def test_shard_counts(shards: dict[str, list[dict[str, Any]]]) -> None:
    assert set(shards) == set(CATEGORIES)
    for cat, rows in shards.items():
        assert len(rows) >= PER_CATEGORY_MIN, f"{cat}: only {len(rows)} rows"


def test_total_count_is_at_least_500(records: list[dict[str, Any]]) -> None:
    assert len(records) >= TOTAL_MIN


def test_every_record_validates(records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for row in records:
        sample = Sample.model_validate(row)
        assert sample.task == "extract"
        assert 5 <= len(sample.instruction) <= 500
        assert sample.id not in seen, f"duplicate id {sample.id}"
        seen.add(sample.id)
    assert len(seen) == len(records)


def test_language_split(shards: dict[str, list[dict[str, Any]]]) -> None:
    for cat, rows in shards.items():
        langs = Counter(row["meta"]["lang"] for row in rows)
        assert set(langs) == {"ru", "en"}, f"{cat}: languages {dict(langs)}"
        share = langs["ru"] / len(rows)
        assert 0.55 <= share <= 0.65, f"{cat}: ru share {share:.2f}"
    all_langs = Counter(row["meta"]["lang"] for rows in shards.values() for row in rows)
    overall = all_langs["ru"] / sum(all_langs.values())
    assert abs(overall - RU_SHARE) <= 0.02, f"overall ru share {overall:.3f}"


def test_html_length_bounds(records: list[dict[str, Any]]) -> None:
    for row in records:
        size = len(row["input_html"])
        assert HTML_MIN <= size <= HTML_MAX, f"{row['id']}: html {size} chars"


def test_meta_matches_shard(records: list[dict[str, Any]]) -> None:
    for row in records:
        category, lang = row["group_id"].split(".")[1:]
        assert category in CATEGORIES
        assert lang in ("ru", "en")
        assert row["meta"]["lang"] == lang
        assert 1 <= row["meta"]["complexity"] <= 3
        assert row["meta"]["source_url"].startswith("https://synthetic.local/")
        assert row["meta"]["pipeline_version"]


def test_no_scripts_or_jsonld(records: list[dict[str, Any]]) -> None:
    for row in records:
        html = row["input_html"]
        assert "<script" not in html.lower(), row["id"]
        assert "application/ld+json" not in html, row["id"]
        assert "<style" not in html.lower(), row["id"]


def test_anchor_title_visible(records: list[dict[str, Any]]) -> None:
    for row in records:
        anchor = _anchor(row["target"])
        assert anchor, f"{row['id']}: no anchor in target"
        visible = _visible(row["input_html"]).lower()
        assert anchor[:80].lower() in visible, f"{row['id']}: anchor not visible"


def test_target_values_present_in_html(records: list[dict[str, Any]]) -> None:
    for row in records:
        missing = _missing_leaves(row["target"], row["input_html"])
        assert not missing, f"{row['id']}: absent target values {missing}"


def test_report_consistent(
    report: dict[str, Any],
    shards: dict[str, list[dict[str, Any]]],
    records: list[dict[str, Any]],
) -> None:
    assert report["seed"] == SEED
    assert report["written"] == len(records) >= TOTAL_MIN
    assert report["requested"] == len(records)
    assert report["by_category"] == {cat: len(shards[cat]) for cat in CATEGORIES}
    assert abs(report["lang_share"]["ru"] - RU_SHARE) <= 0.02
    assert abs(report["lang_share"]["en"] - 0.4) <= 0.02
    assert report["html_chars"]["min"] >= HTML_MIN
    assert report["html_chars"]["max"] <= HTML_MAX
    tokens = report["token_stats"]
    assert tokens["p50"] is not None or tokens.get("note"), "no token stats and no note"
    for rel in report["files"]:
        assert (REPO_ROOT / rel).is_file(), f"report lists missing file {rel}"


def test_plan_buckets_exact_split() -> None:
    plan = gen.plan_buckets(TOTAL_MIN, SEED)
    assert len(plan) == TOTAL_MIN
    ru = sum(1 for _, lang, _ in plan if lang == "ru")
    assert ru == 300
    per_category: dict[str, int] = {}
    for cat, _, _ in plan:
        per_category[cat] = per_category.get(cat, 0) + 1
    assert per_category == {cat: 100 for cat in CATEGORIES}


def test_generation_deterministic(tmp_path: Path) -> None:
    first, _ = gen.generate(tmp_path / "a", 25, SEED)
    second, _ = gen.generate(tmp_path / "b", 25, SEED)
    assert _fingerprint(first) == _fingerprint(second)
    other, _ = gen.generate(tmp_path / "c", 25, SEED + 1)
    assert _fingerprint(first) != _fingerprint(other)


def test_determinism_matches_committed_shards(tmp_path: Path) -> None:
    """Regenerating with seed 42 must reproduce the ids/targets on disk."""
    regenerated, _ = gen.generate(tmp_path / "rerun", TOTAL_MIN, SEED)
    disk = [row for cat in CATEGORIES for row in _read_shard(cat)]
    assert len(regenerated) == len(disk)
    assert _fingerprint(regenerated) == _fingerprint(disk)
