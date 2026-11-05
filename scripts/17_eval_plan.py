"""Evaluate the task-conditioned planner.

Metrics are semantic: entity, field set, required fields, scope, empty plans.
Predictions are parsed through the same ``CollectionPlan`` contract used in
training, so a lucky JSON object with wrong keys does not count as valid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from web2spec.eval import extract_json  # noqa: E402
from web2spec.plan import CollectionPlan, PlanError, load_plan  # noqa: E402
from web2spec.schema import Sample  # noqa: E402
from web2spec.tokens import build_prompt, load_tokenizer  # noqa: E402
from web2spec.validate import load_samples  # noqa: E402


def gold_plan(sample: Sample) -> CollectionPlan:
    return load_plan(sample.target)


def _scope(plan: CollectionPlan) -> tuple[bool, bool]:
    return plan.pagination, plan.detail_pages


def score(predictions: dict[str, str], samples: list[Sample]) -> dict[str, Any]:
    json_valid = 0
    plan_valid = 0
    entity_ok = 0
    scope_ok = 0
    empty_total = 0
    empty_ok = 0
    exact = 0
    matched = predicted = expected_total = required_total = required_found = 0
    details: list[dict[str, Any]] = []

    for sample in samples:
        expected = gold_plan(sample)
        raw = predictions.get(sample.id, "")
        parsed = extract_json(raw)
        if parsed is not None:
            json_valid += 1
        plan = None
        error = None
        try:
            plan = load_plan(parsed) if parsed is not None else None
            if plan is not None:
                plan_valid += 1
        except PlanError as exc:
            error = str(exc)

        expected_names = {field.name for field in expected.fields}
        predicted_names = {field.name for field in plan.fields} if plan else set()
        matched += len(expected_names & predicted_names)
        predicted += len(predicted_names)
        expected_total += len(expected_names)
        for field in expected.fields:
            if field.required:
                required_total += 1
                if field.name in predicted_names:
                    required_found += 1

        if expected.entity == "none":
            empty_total += 1
            if plan is not None and plan.entity == "none":
                empty_ok += 1

        row_entity = plan is not None and plan.entity == expected.entity
        entity_ok += int(row_entity)
        row_scope = plan is not None and _scope(plan) == _scope(expected)
        scope_ok += int(row_scope)
        row_exact = plan is not None and plan == expected
        exact += int(row_exact)
        details.append(
            {
                "id": sample.id,
                "group_id": sample.group_id,
                "json_valid": parsed is not None,
                "plan_valid": plan is not None,
                "entity_ok": row_entity,
                "scope_ok": row_scope,
                "exact": row_exact,
                "expected_entity": expected.entity,
                "predicted_entity": plan.entity if plan else None,
                "missing_required": sorted(
                    field.name
                    for field in expected.fields
                    if field.required and field.name not in predicted_names
                ),
                "extra_fields": sorted(predicted_names - expected_names),
                "error": error,
            }
        )

    precision = matched / predicted if predicted else 0.0
    recall = matched / expected_total if expected_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = len(samples)
    return {
        "count": n,
        "json_valid_rate": json_valid / n if n else 0.0,
        "plan_valid_rate": plan_valid / n if n else 0.0,
        "entity_accuracy": entity_ok / n if n else 0.0,
        "scope_accuracy": scope_ok / n if n else 0.0,
        "empty_accuracy": empty_ok / empty_total if empty_total else None,
        "exact_plan": exact / n if n else 0.0,
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": f1,
        "required_recall": required_found / required_total if required_total else None,
        "details": details,
    }


def generate(samples: list[Sample], model_dir: str, max_new_tokens: int) -> dict[str, str]:
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
        print(f"[{index}/{len(samples)}] {sample.id}: {len(text)} chars")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Оценка planner-а")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/plan/test.jsonl")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "data/reports/eval_plan.json")
    parser.add_argument("--save-predictions", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    args = parser.parse_args(argv)

    samples = load_samples(args.data)
    if args.limit:
        samples = samples[: args.limit]
    if args.predictions:
        predictions = {}
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                predictions[row["id"]] = row.get("output", "")
    elif args.model:
        predictions = generate(samples, args.model, args.max_new_tokens)
    else:
        print("укажите --model или --predictions", file=sys.stderr)
        return 2

    result = score(predictions, samples)
    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.save_predictions.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in samples:
                row = {"id": sample.id, "output": predictions.get(sample.id, "")}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "details"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Отчёт: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
