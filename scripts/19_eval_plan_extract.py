"""Evaluate the plan-conditioned fact extractor.

A fact is correct only when the field is in the plan, the value matches the
gold value, and the evidence is an exact substring of the input HTML.  Fields
that the plan expects but the page does not contain must land in ``missing``.
"""

from __future__ import annotations

import argparse
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
    PlanError,
    load_factset,
    load_plan,
    validate_factset,
)
from web2spec.schema import Sample  # noqa: E402
from web2spec.tokens import build_prompt, load_tokenizer  # noqa: E402
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


def score(predictions: dict[str, str], samples: list[Sample]) -> dict[str, Any]:
    json_valid = factset_valid = strict_valid = exact = 0
    expected = matched = predicted_total = wrong = extra = 0
    evidence_ok = evidence_total = 0
    missing_total = missing_ok = 0
    coverage_sum = 0.0
    details: list[dict[str, Any]] = []

    for sample in samples:
        plan = load_plan(sample.meta.plan)
        gold = load_factset(sample.target)
        plan_names = {field.name for field in plan.fields}
        gold_facts = {fact.name: fact for fact in gold.facts}
        gold_missing = set(gold.missing)

        raw = predictions.get(sample.id, "")
        parsed = extract_json(raw)
        if parsed is not None:
            json_valid += 1
        factset = None
        error = None
        strict = False
        try:
            factset = load_factset(parsed) if parsed is not None else None
            if factset is not None:
                factset_valid += 1
                validate_factset(factset, plan, sample.input_html)
                strict = True
                strict_valid += 1
        except PlanError as exc:
            error = str(exc)

        pred_facts = {fact.name: fact for fact in factset.facts} if factset else {}
        pred_missing = set(factset.missing) if factset else set()
        predicted_total += len(pred_facts)
        extra += len(set(pred_facts) - plan_names)

        for fact in pred_facts.values():
            evidence_total += 1
            if fact.evidence in sample.input_html:
                evidence_ok += 1

        for name in plan_names:
            if name in gold_facts:
                expected += 1
                if name in pred_facts:
                    if values_match(pred_facts[name].value, gold_facts[name].value):
                        matched += 1
                    else:
                        wrong += 1
            else:
                missing_total += 1
                if name in pred_missing:
                    missing_ok += 1

        covered = len((set(pred_facts) | pred_missing) & plan_names)
        coverage_sum += covered / len(plan_names) if plan_names else 0.0
        row_exact = factset is not None and factset == gold
        exact += int(row_exact)
        details.append(
            {
                "id": sample.id,
                "group_id": sample.group_id,
                "json_valid": parsed is not None,
                "factset_valid": factset is not None,
                "strict_valid": strict,
                "exact": row_exact,
                "expected_facts": len(gold_facts),
                "predicted_facts": len(pred_facts),
                "matched": sum(
                    1
                    for name, fact in pred_facts.items()
                    if name in gold_facts and values_match(fact.value, gold_facts[name].value)
                ),
                "missing_expected": sorted(gold_missing),
                "missing_predicted": sorted(pred_missing),
                "extra_facts": sorted(set(pred_facts) - plan_names),
                "error": error,
            }
        )

    precision = matched / predicted_total if predicted_total else 0.0
    recall = matched / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = len(samples)
    return {
        "count": n,
        "json_valid_rate": json_valid / n if n else 0.0,
        "factset_valid_rate": factset_valid / n if n else 0.0,
        "strict_valid_rate": strict_valid / n if n else 0.0,
        "exact_factset": exact / n if n else 0.0,
        "fact_precision": precision,
        "fact_recall": recall,
        "fact_f1": f1,
        "evidence_rate": evidence_ok / evidence_total if evidence_total else None,
        "missing_accuracy": missing_ok / missing_total if missing_total else None,
        "coverage": coverage_sum / n if n else 0.0,
        "matched": matched,
        "expected": expected,
        "predicted": predicted_total,
        "wrong_values": wrong,
        "extra_facts": extra,
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
    parser = argparse.ArgumentParser(description="Оценка plan-conditioned extractor")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/plan_extract/test.jsonl")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--report", type=Path, default=REPO_ROOT / "data/reports/eval_plan_extract.json"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args(argv)

    samples = load_samples(args.data)
    if args.limit:
        samples = samples[: args.limit]
    if args.predictions:
        predictions: dict[str, str] = {}
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
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "details"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Отчёт: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
