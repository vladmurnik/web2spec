"""Evaluation metrics for JSON extraction, plus an optional generation run."""

from __future__ import annotations

import json
import re
from typing import Any

from web2spec.config import REPO_ROOT, EvalConfig
from web2spec.schema import Sample
from web2spec.tokens import build_prompt
from web2spec.validate import load_samples


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts/lists into dotted keys with scalar leaves."""

    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(flatten(item, f"{prefix}[{index}]"))
    else:
        result[prefix] = value
    return result


def normalize(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _normalized_map(value: dict[str, Any]) -> dict[str, Any]:
    return {key: _typed(item) for key, item in flatten(value).items()}


def _typed(value: Any) -> tuple[str, Any]:
    """Type-aware comparison key: ``True`` must not equal ``1``."""

    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("num", float(value))
    if isinstance(value, str):
        return ("str", " ".join(value.split()))
    if value is None:
        return ("null", None)
    return ("other", str(value))


def field_scores(pred: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    pred_fields = _normalized_map(pred)
    gold_fields = _normalized_map(gold)
    shared = set(pred_fields) & set(gold_fields)
    matched = sum(1 for key in shared if pred_fields[key] == gold_fields[key])
    precision = matched / len(pred_fields) if pred_fields else 0.0
    recall = matched / len(gold_fields) if gold_fields else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": pred_fields == gold_fields,
        "matched": matched,
        "gold_fields": len(gold_fields),
        "pred_fields": len(pred_fields),
    }


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON object extraction from model output."""

    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        candidate = _balanced_object(text, start)
        if candidate is None:
            break
        try:
            parsed = json.loads(candidate.replace(",}", "}").replace(",]", "]"))
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


def _balanced_object(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-record scores; ``pred`` may be raw text or an object."""

    details: list[dict[str, Any]] = []
    totals = {"matched": 0, "pred_fields": 0, "gold_fields": 0}
    macro_p = macro_r = macro_f1 = 0.0
    exact = 0
    valid = 0
    by_domain: dict[str, dict[str, float]] = {}
    by_complexity: dict[str, dict[str, float]] = {}

    for record in records:
        gold = record.get("gold") or {}
        pred_raw = record.get("pred")
        pred = pred_raw if isinstance(pred_raw, dict) else extract_json(str(pred_raw or ""))
        if pred is not None:
            valid += 1
            scores = field_scores(pred, gold)
        else:
            scores = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "exact": False,
                "matched": 0,
                "gold_fields": len(flatten(gold)),
                "pred_fields": 0,
            }
        for key in totals:
            totals[key] += int(scores[key])
        macro_p += scores["precision"]
        macro_r += scores["recall"]
        macro_f1 += scores["f1"]
        exact += int(scores["exact"])
        detail = {
            "id": record.get("id"),
            "domain": str(record.get("domain") or "unknown"),
            "complexity": int(record.get("complexity") or 0),
            "json_valid": pred is not None,
            **scores,
        }
        details.append(detail)
        _accumulate(by_domain, detail["domain"], detail)
        _accumulate(by_complexity, str(detail["complexity"]), detail)

    count = len(records) or 1
    precision = totals["matched"] / totals["pred_fields"] if totals["pred_fields"] else 0.0
    recall = totals["matched"] / totals["gold_fields"] if totals["gold_fields"] else 0.0
    return {
        "count": len(records),
        "json_valid_rate": valid / count,
        "exact_match": exact / count,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "macro_precision": macro_p / count,
        "macro_recall": macro_r / count,
        "macro_f1": macro_f1 / count,
        "by_domain": by_domain,
        "by_complexity": by_complexity,
        "details": details,
    }


def _accumulate(bucket: dict[str, dict[str, float]], key: str, detail: dict[str, Any]) -> None:
    entry = bucket.setdefault(
        key, {"count": 0.0, "exact_match": 0.0, "f1": 0.0, "json_valid_rate": 0.0}
    )
    entry["count"] += 1
    entry["exact_match"] += float(detail["exact"])
    entry["f1"] += float(detail["f1"])
    entry["json_valid_rate"] += float(detail["json_valid"])


def _finalize_buckets(report: dict[str, Any]) -> None:
    for group in ("by_domain", "by_complexity"):
        for entry in report[group].values():
            count = entry["count"] or 1
            entry["exact_match"] /= count
            entry["f1"] /= count
            entry["json_valid_rate"] /= count


def run(cfg: EvalConfig) -> dict[str, Any]:
    """Generate predictions with a merged HF model and score them."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = (REPO_ROOT / cfg.model_dir).resolve()
    raw_model = str(cfg.model_dir).replace("\\", "/")
    model_ref = str(model_dir) if model_dir.exists() else raw_model
    data_path = (REPO_ROOT / cfg.data).resolve()
    report_path = (REPO_ROOT / cfg.report).resolve()
    samples = load_samples(data_path)
    if cfg.limit:
        samples = samples[: cfg.limit]

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    from transformers import __version__ as transformers_version

    dtype_kwarg = (
        "dtype" if int(transformers_version.split(".", 1)[0]) >= 5 else "torch_dtype"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        **{dtype_kwarg: dtype},
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=False,
    )
    model.eval()
    device = next(model.parameters()).device

    records = []
    for sample in samples:
        prompt = build_prompt(sample, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": cfg.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if cfg.temperature > 0:
            generate_kwargs.update(do_sample=True, temperature=cfg.temperature)
        else:
            generate_kwargs.update(do_sample=False)
        with torch.no_grad():
            output = model.generate(**inputs, **generate_kwargs)
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        records.append(_record(sample, text))

    report = evaluate_records(records)
    _finalize_buckets(report)
    report["config"] = cfg.model_dump(mode="json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _record(sample: Sample, text: str) -> dict[str, Any]:
    return {
        "id": sample.id,
        "pred": text,
        "gold": sample.target,
        "domain": sample.group_id,
        "complexity": sample.meta.complexity,
    }
