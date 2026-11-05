"""Evaluate the runtime cascade: task + page -> planner -> extractor -> FactSet.

Both student models run offline, one after another, so the 4 GB GPU only ever
holds one merged model.  The planner output is validated through the same
``CollectionPlan`` contract as in training; a failed plan is written to the
repair buffer and counts as a failed task instead of silently falling back to
the gold plan.  Facts are checked against the reliable-label gold FactSet:
values must match, evidence must be an exact HTML substring, and fields that
the page does not contain must be listed in ``missing``.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from web2spec.eval import extract_json  # noqa: E402
from web2spec.plan import (  # noqa: E402
    CollectionPlan,
    PlanError,
    load_factset,
    load_plan,
    repair_factset,
    validate_factset,
)
from web2spec.schema import Meta, Sample  # noqa: E402
from web2spec.tokens import (  # noqa: E402
    build_plan_extract_prompt,
    build_plan_prompt,
    load_tokenizer,
)
from web2spec.validate import load_samples  # noqa: E402

_NUM_RE = re.compile(r"^-?\d+(?:[.,]\d+)?$")


def _norm(value: Any) -> str:
    return " ".join(str(value).replace("\u00a0", " ").split()).strip().casefold()


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.replace("\u00a0", " ").replace(" ", "").replace(",", ".")
        if _NUM_RE.match(text):
            return float(text)
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _norm(value)
    if text in {"true", "1", "да", "yes"}:
        return True
    if text in {"false", "0", "нет", "no"}:
        return False
    return None


def _as_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def values_match(predicted: Any, gold: Any) -> bool:
    if isinstance(gold, bool) or isinstance(predicted, bool):
        left, right = _as_bool(predicted), _as_bool(gold)
        return left is not None and right is not None and left == right
    left_num, right_num = _as_number(predicted), _as_number(gold)
    if left_num is not None and right_num is not None:
        return abs(left_num - right_num) < 1e-6
    left_dt, right_dt = _as_datetime(predicted), _as_datetime(gold)
    if left_dt is not None and right_dt is not None:
        return left_dt == right_dt
    return _norm(predicted) == _norm(gold)


def generate(
    samples: list[Sample],
    model_dir: str,
    build_prompt,
    max_new_tokens: int,
    label: str,
) -> dict[str, str]:
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model_dir)
    kwargs: dict[str, Any] = {"dtype": torch.float16}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
        kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.eval()
    output: dict[str, str] = {}
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
            output[sample.id] = text
            print(f"[{label} {index}/{len(samples)}] {sample.id}: {len(text)} chars")
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output


def evaluate(
    samples: list[Sample],
    plans: dict[str, str],
    factsets: dict[str, str],
    gold_plans: dict[tuple[str | None, str], CollectionPlan],
    repair_buffer: Path | None,
    repair: bool = True,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    counters = {
        "plan_json_valid": 0,
        "plan_valid": 0,
        "entity_ok": 0,
        "factset_json_valid": 0,
        "factset_valid": 0,
        "strict_valid": 0,
        "success": 0,
        "required_total": 0,
        "required_found": 0,
        "expected": 0,
        "matched": 0,
        "wrong_values": 0,
        "predicted": 0,
        "extra_facts": 0,
        "missing_total": 0,
        "missing_ok": 0,
        "required_fact_total": 0,
        "required_fact_ok": 0,
        "evidence_repaired": 0,
        "facts_dropped": 0,
    }

    for sample in samples:
        gold = load_factset(sample.target)
        gold_plan = gold_plans.get(
            (sample.meta.source_url, sample.instruction)
        ) or load_plan(sample.meta.plan)
        gold_facts = {fact.name: fact for fact in gold.facts}
        gold_missing = set(gold.missing)

        plan_raw = plans.get(sample.id, "")
        plan_parsed = extract_json(plan_raw)
        if plan_parsed is not None:
            counters["plan_json_valid"] += 1
        plan = None
        plan_error = None
        try:
            plan = load_plan(plan_parsed) if plan_parsed is not None else None
            if plan is not None:
                counters["plan_valid"] += 1
        except PlanError as exc:
            plan_error = str(exc)

        factset = None
        factset_error = None
        strict = False
        repaired_facts: list[dict[str, Any]] = []
        dropped_facts: list[dict[str, Any]] = []
        if plan is None:
            factset_error = "no valid plan"
        else:
            raw = factsets.get(sample.id, "")
            parsed = extract_json(raw)
            if parsed is not None:
                counters["factset_json_valid"] += 1
            try:
                factset = load_factset(parsed) if parsed is not None else None
                if factset is not None:
                    counters["factset_valid"] += 1
                    if repair:
                        factset, repaired_facts, dropped_facts = repair_factset(
                            factset, plan, sample.input_html
                        )
                        counters["evidence_repaired"] += len(repaired_facts)
                        counters["facts_dropped"] += len(dropped_facts)
                    validate_factset(factset, plan, sample.input_html)
                    strict = True
                    counters["strict_valid"] += 1
            except PlanError as exc:
                factset_error = str(exc)

        plan_names = {field.name for field in plan.fields} if plan else set()
        pred_facts = {fact.name: fact for fact in factset.facts} if factset else {}
        pred_missing = set(factset.missing) if factset else set()

        entity_ok = plan is not None and plan.entity == gold_plan.entity
        counters["entity_ok"] += int(entity_ok)

        required_names = {field.name for field in gold_plan.fields if field.required}
        required_found = required_names & plan_names
        counters["required_total"] += len(required_names)
        counters["required_found"] += len(required_found)

        comparable = plan_names & (set(gold_facts) | gold_missing)
        for name in comparable:
            if name in gold_facts:
                counters["expected"] += 1
                if name in pred_facts:
                    counters["predicted"] += 1
                    if values_match(pred_facts[name].value, gold_facts[name].value):
                        counters["matched"] += 1
                    else:
                        counters["wrong_values"] += 1
                else:
                    counters["predicted"] += int(name in pred_missing)
            else:
                counters["missing_total"] += 1
                if name in pred_facts:
                    counters["wrong_values"] += 1
                elif name in pred_missing:
                    counters["missing_ok"] += 1
        counters["extra_facts"] += len(set(pred_facts) - (set(gold_facts) | gold_missing))

        required_gold_facts = [
            name for name in required_names if name in gold_facts and name in comparable
        ]
        required_fact_ok = sum(
            1
            for name in required_gold_facts
            if name in pred_facts and values_match(pred_facts[name].value, gold_facts[name].value)
        )
        counters["required_fact_total"] += len(required_gold_facts)
        counters["required_fact_ok"] += required_fact_ok

        no_wrong = all(
            values_match(pred_facts[name].value, gold_facts[name].value)
            for name in comparable
            if name in gold_facts and name in pred_facts
        ) and not any(name in pred_facts for name in (comparable & gold_missing))
        all_required_facts = required_fact_ok == len(required_gold_facts)
        success = bool(
            plan is not None
            and entity_ok
            and required_found == required_names
            and strict
            and no_wrong
            and all_required_facts
        )
        counters["success"] += int(success)

        if plan is None or not strict or repaired_facts or dropped_facts:
            repairs.append(
                {
                    "id": sample.id,
                    "source_url": sample.meta.source_url,
                    "instruction": sample.instruction,
                    "plan_error": plan_error,
                    "factset_error": factset_error,
                    "evidence_repaired": repaired_facts,
                    "facts_dropped": dropped_facts,
                    "plan_output": plan_raw,
                    "factset_output": factsets.get(sample.id, ""),
                }
            )

        details.append(
            {
                "id": sample.id,
                "source_url": sample.meta.source_url,
                "instruction": sample.instruction,
                "plan_json_valid": plan_parsed is not None,
                "plan_valid": plan is not None,
                "entity_ok": entity_ok,
                "gold_entity": gold_plan.entity,
                "predicted_entity": plan.entity if plan else None,
                "required_total": len(required_names),
                "required_found": len(required_found),
                "factset_json_valid": factset is not None,
                "factset_valid": factset is not None,
                "strict_valid": strict,
                "comparable_fields": sorted(comparable),
                "unverifiable_fields": sorted(
                    plan_names - (set(gold_facts) | gold_missing)
                ),
                "matched": sum(
                    1
                    for name in comparable
                    if name in gold_facts
                    and name in pred_facts
                    and values_match(pred_facts[name].value, gold_facts[name].value)
                ),
                "wrong_values": sum(
                    1
                    for name in comparable
                    if name in gold_facts
                    and name in pred_facts
                    and not values_match(pred_facts[name].value, gold_facts[name].value)
                ),
                "facts": len(pred_facts),
                "missing": len(pred_missing),
                "evidence_repaired": len(repaired_facts),
                "facts_dropped": len(dropped_facts),
                "success": success,
                "plan_error": plan_error,
                "factset_error": factset_error,
            }
        )

    if repair_buffer is not None and repairs:
        repair_buffer.parent.mkdir(parents=True, exist_ok=True)
        with repair_buffer.open("w", encoding="utf-8", newline="\n") as handle:
            for row in repairs:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    count = len(samples)
    precision = counters["matched"] / counters["predicted"] if counters["predicted"] else 0.0
    recall = counters["matched"] / counters["expected"] if counters["expected"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": count,
        "plan_json_valid_rate": counters["plan_json_valid"] / count if count else 0.0,
        "plan_valid_rate": counters["plan_valid"] / count if count else 0.0,
        "entity_accuracy": counters["entity_ok"] / count if count else 0.0,
        "required_recall": (
            counters["required_found"] / counters["required_total"]
            if counters["required_total"]
            else None
        ),
        "factset_json_valid_rate": (
            counters["factset_json_valid"] / count if count else 0.0
        ),
        "factset_valid_rate": counters["factset_valid"] / count if count else 0.0,
        "strict_valid_rate": counters["strict_valid"] / count if count else 0.0,
        "fact_precision": precision,
        "fact_recall": recall,
        "fact_f1": f1,
        "wrong_values": counters["wrong_values"],
        "extra_facts": counters["extra_facts"],
        "missing_accuracy": (
            counters["missing_ok"] / counters["missing_total"]
            if counters["missing_total"]
            else None
        ),
        "required_fact_accuracy": (
            counters["required_fact_ok"] / counters["required_fact_total"]
            if counters["required_fact_total"]
            else None
        ),
        "task_success_rate": counters["success"] / count if count else 0.0,
        "evidence_repaired": counters["evidence_repaired"],
        "facts_dropped": counters["facts_dropped"],
        "repairs": len(repairs),
        "details": details,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Оценка runtime-каскада planner → extractor")
    parser.add_argument(
        "--planner", type=str, default=str(REPO_ROOT / "models/merged/qwen05b-plan")
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default=str(REPO_ROOT / "models/merged/qwen05b-plan-extract-v2"),
    )
    parser.add_argument(
        "--data", type=Path, default=REPO_ROOT / "data/plan_extract/test.jsonl"
    )
    parser.add_argument(
        "--gold-plans", type=Path, default=REPO_ROOT / "data/plan/test.jsonl"
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="не исправлять evidence и не переносить недоказуемые факты в missing",
    )
    parser.add_argument("--plan-max-new-tokens", type=int, default=320)
    parser.add_argument("--fact-max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--report", type=Path, default=REPO_ROOT / "data/reports/eval_runtime.json"
    )
    parser.add_argument(
        "--save-predictions", type=Path, default=None, help="JSONL с планами и фактами"
    )
    parser.add_argument(
        "--repair-buffer",
        type=Path,
        default=REPO_ROOT / "data/reports/runtime_repair_buffer.jsonl",
    )
    args = parser.parse_args(argv)

    samples = load_samples(args.data)
    if args.limit:
        samples = samples[: args.limit]

    gold_plans: dict[tuple[str | None, str], CollectionPlan] = {}
    if args.gold_plans.exists():
        for sample in load_samples(args.gold_plans):
            try:
                gold_plans[(sample.meta.source_url, sample.instruction)] = load_plan(
                    sample.target
                )
            except PlanError:
                continue

    plans = generate(
        samples, args.planner, build_plan_prompt, args.plan_max_new_tokens, "plan"
    )

    extract_samples: list[Sample] = []
    plan_rows: dict[str, str] = {}
    for sample in samples:
        try:
            plan = load_plan(extract_json(plans.get(sample.id, "")))
        except (PlanError, TypeError):
            plan = None
        if plan is None:
            continue
        plan_rows[sample.id] = json.dumps(
            plan.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
        )
        extract_samples.append(
            Sample(
                id=sample.id,
                group_id=sample.group_id,
                task="plan_extract",
                instruction=sample.instruction,
                input_html=sample.input_html,
                target={"facts": []},
                meta=Meta(
                    lang=sample.meta.lang,
                    complexity=sample.meta.complexity,
                    source_url=sample.meta.source_url,
                    pipeline_version="runtime-0.1.0",
                    plan=plan.model_dump(mode="json"),
                ),
            )
        )

    factsets = generate(
        extract_samples,
        args.extractor,
        build_plan_extract_prompt,
        args.fact_max_new_tokens,
        "fact",
    )

    result = evaluate(
        samples, plans, factsets, gold_plans, args.repair_buffer, repair=not args.no_repair
    )

    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.save_predictions.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in samples:
                handle.write(
                    json.dumps(
                        {
                            "id": sample.id,
                            "source_url": sample.meta.source_url,
                            "instruction": sample.instruction,
                            "plan": plans.get(sample.id, ""),
                            "parsed_plan": plan_rows.get(sample.id),
                            "factset": factsets.get(sample.id, ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "details"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Отчёт: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())