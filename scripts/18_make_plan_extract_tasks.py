"""Build plan-conditioned extraction samples from planner tasks.

The planner dataset already pairs a page sample with a task-specific
``CollectionPlan``.  For extraction we reuse the same page and plan, and build
a gold ``FactSet`` from the source extraction label.  A field becomes a fact
only when its value can be located as an exact substring of the compressed
HTML; otherwise it is listed in ``missing``.  That keeps every training label
honest for the input the extractor actually sees.
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

from web2spec.evidence import find_evidence  # noqa: E402
from web2spec.plan import (  # noqa: E402
    Fact,
    FactSet,
    PlanField,
    fact_target,
    load_plan,
    validate_factset,
)
from web2spec.schema import Sample, load_task_schemas, make_id, validate_sample  # noqa: E402
from web2spec.tokens import build_prompt, load_tokenizer, target_text  # noqa: E402

PIPELINE_VERSION = "plan-extract-reliable-0.2.0"
SCALAR_TYPES = (str, int, float, bool)


def load_pages(path: Path) -> dict[str, Sample]:
    pages: dict[str, Sample] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = Sample.model_validate(json.loads(line))
        url = sample.meta.source_url
        if url:
            pages[url] = sample
    return pages


def build_sample(
    plan_sample: Sample, source: Sample
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """Return a plan_extract record and per-sample counters."""

    plan = load_plan(plan_sample.target)
    stats = {"facts": 0, "missing": 0, "fields": len(plan.fields)}
    if plan.entity == "none" or not plan.fields:
        return None, stats

    target = source.target
    if not isinstance(target, dict):
        return None, stats

    facts: list[Fact] = []
    missing: list[str] = []
    kept_fields: list[PlanField] = []
    for field in plan.fields:
        value = target.get(field.name)
        if not isinstance(value, SCALAR_TYPES):
            continue
        evidence = find_evidence(plan_sample.input_html, value, key=field.name)
        if evidence is None:
            continue
        facts.append(Fact(name=field.name, value=value, evidence=evidence))
        kept_fields.append(field)
        stats["facts"] += 1

    negative_fields: list[PlanField] = []
    negative_name = "salary" if plan.entity == "realestate" else "rooms"
    existing = {field.name for field in kept_fields}
    if (
        random.Random(make_id("neg", plan_sample.id)).random() < 0.35
        and negative_name not in existing
        and len(kept_fields) < 12
        and not isinstance(target.get(negative_name), SCALAR_TYPES)
    ):
        negative_fields.append(
            PlanField(
                name=negative_name,
                why="поле для выполнения задачи",
                required=False,
                kind="number",
            )
        )
        missing.append(negative_name)

    modified_plan = plan.model_copy(deep=True)
    modified_plan.fields = kept_fields + negative_fields
    stats["fields"] = len(modified_plan.fields)
    stats["missing"] = len(missing)
    if not facts and not missing:
        return None, stats

    factset = FactSet(facts=facts, missing=missing)
    validate_factset(factset, modified_plan, plan_sample.input_html)
    record = {
        "id": make_id("pex", plan_sample.id),
        "group_id": plan_sample.group_id,
        "task": "plan_extract",
        "instruction": plan_sample.instruction,
        "input_html": plan_sample.input_html,
        "target": fact_target(factset),
        "meta": {
            "lang": plan_sample.meta.lang,
            "complexity": plan_sample.meta.complexity,
            "source_url": plan_sample.meta.source_url,
            "n_tokens": None,
            "pipeline_version": PIPELINE_VERSION,
            "plan": modified_plan.model_dump(mode="json"),
        },
    }
    return record, stats


def _fits(sample: Sample, tokenizer: Any, max_seq_len: int) -> bool:
    prompt = build_prompt(sample, tokenizer)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text(sample), add_special_tokens=False)["input_ids"]
    return len(prompt_ids) + len(target_ids) + 1 <= max_seq_len


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan-conditioned extraction dataset")
    parser.add_argument("--plans", type=Path, default=REPO_ROOT / "data/plan")
    parser.add_argument("--pages", type=Path, default=REPO_ROOT / "data/processed/train.jsonl")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/plan_extract")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-seq-len", type=int, default=1536)
    args = parser.parse_args(argv)

    schemas = load_task_schemas(REPO_ROOT / "configs/tasks.yaml")
    tokenizer = load_tokenizer(args.tokenizer) if args.max_seq_len else None
    pages = load_pages(args.pages)
    args.out.mkdir(parents=True, exist_ok=True)

    totals: Counter[str] = Counter()
    by_entity: Counter[str] = Counter()
    split_sizes: dict[str, int] = {}
    for split in ("train", "val", "test"):
        plan_path = args.plans / f"{split}.jsonl"
        if not plan_path.exists():
            raise FileNotFoundError(f"planner split not found: {plan_path}")
        records: list[dict[str, Any]] = []
        for line in plan_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            plan_sample = Sample.model_validate(json.loads(line))
            totals["plans"] += 1
            source = pages.get(plan_sample.meta.source_url or "")
            if source is None:
                totals["skipped_no_source"] += 1
                continue
            plan = load_plan(plan_sample.target)
            if plan.entity == "none":
                totals["skipped_empty_plan"] += 1
                continue
            record, stats = build_sample(plan_sample, source)
            if record is None:
                totals["skipped_no_fields"] += 1
                continue
            validated = validate_sample(record, schemas)
            if tokenizer is not None and not _fits(validated, tokenizer, args.max_seq_len):
                totals["skipped_too_long"] += 1
                continue
            records.append(validated.model_dump(mode="json"))
            totals["samples"] += 1
            totals["fields"] += stats["fields"]
            totals["facts"] += stats["facts"]
            totals["missing"] += stats["missing"]
            by_entity[plan.entity] += 1

        path = args.out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        split_sizes[split] = len(records)

    totals["evidence_rate"] = round(
        totals["facts"] / totals["fields"] if totals["fields"] else 0.0, 4
    )
    report = {
        "splits": split_sizes,
        "by_entity": dict(by_entity),
        **dict(totals),
        "pipeline_version": PIPELINE_VERSION,
    }
    (args.out / "stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
