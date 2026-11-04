"""Evaluate scraping-spec generation by executing the produced spec.

Unlike extract evaluation, this measures whether a spec is valid JSON, whether
it executes on the held-out HTML, and which fields it actually recovers.
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
from web2spec.schema import Sample  # noqa: E402
from web2spec.spec import SpecError, extract, load_spec  # noqa: E402
from web2spec.tokens import build_prompt, load_tokenizer  # noqa: E402
from web2spec.validate import load_samples  # noqa: E402


def load_gold(path: Path) -> dict[str, Any]:
    gold = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gold[row["id"]] = row["expected"]
    return gold


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def score(predictions: dict[str, str], samples: list[Sample], gold: dict[str, Any]) -> dict:
    spec_valid = 0
    exec_ok = 0
    page_exact = 0
    matched = gold_fields = pred_fields = 0
    details = []

    for sample in samples:
        raw = predictions.get(sample.id, "")
        parsed = extract_json(raw)
        rows: list[dict[str, Any]] = []
        valid = parsed is not None
        executed = False
        error = None
        try:
            spec = load_spec(parsed) if parsed is not None else None
            if spec is not None:
                spec_valid += 1
                rows = _rows(extract(spec, sample.input_html, base_url=sample.meta.source_url))
                executed = True
                exec_ok += 1
        except (SpecError, ValueError) as exc:
            error = str(exc)

        expected = _rows(gold.get(sample.id))
        got = rows
        exact = got == expected
        page_exact += int(exact)
        for index, gold_row in enumerate(expected):
            pred_row = got[index] if index < len(got) else {}
            for key, value in gold_row.items():
                gold_fields += 1
                if key in pred_row:
                    pred_fields += 1
                    if pred_row[key] == value:
                        matched += 1
        details.append(
            {
                "id": sample.id,
                "spec_valid": valid,
                "executed": executed,
                "exact": exact,
                "expected_rows": len(expected),
                "predicted_rows": len(got),
                "error": error,
            }
        )

    precision = matched / pred_fields if pred_fields else 0.0
    recall = matched / gold_fields if gold_fields else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": len(samples),
        "spec_valid_rate": spec_valid / len(samples) if samples else 0.0,
        "exec_success_rate": exec_ok / len(samples) if samples else 0.0,
        "page_exact": page_exact / len(samples) if samples else 0.0,
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": f1,
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
    parser = argparse.ArgumentParser(description="Оценка scraping-spec через исполнение")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/spec/test.jsonl")
    parser.add_argument("--gold", type=Path, default=REPO_ROOT / "data/spec/gold/test.jsonl")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "data/reports/spec_eval.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args(argv)

    samples = load_samples(args.data)
    if args.limit:
        samples = samples[: args.limit]
    gold = load_gold(args.gold)
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

    result = score(predictions, samples, gold)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "details"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Отчёт: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
