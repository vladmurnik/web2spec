"""Run the dated, real-page benchmark through the local collector.

The benchmark is intentionally small and explicit: every case names a live
page, a task, the expected field names and the expected values that were read
from that page on ``checked_at``.  It measures the actual runtime contract
(models + compression + evidence repair), not a saved prediction file.

Examples::

    web2spec benchmark
    web2spec benchmark --limit 3 --no-cache
    web2spec benchmark --out data/reports/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = REPO_ROOT / "scripts" / "21_collect_urls.py"
DEFAULT_CASES = REPO_ROOT / "benchmarks" / "cases.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "reports" / "benchmark.json"

sys.path.insert(0, str(REPO_ROOT / "src"))
from web2spec.llama_runtime import (  # noqa: E402
    DEFAULT_DEVICE,
    DEFAULT_EXTRACTOR,
    DEFAULT_PLANNER,
)


def load_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(case, dict):
            raise ValueError(f"{path}:{line_no}: case must be an object")
        for key in ("id", "url", "task", "fields"):
            if key not in case:
                raise ValueError(f"{path}:{line_no}: missing {key}")
        if not isinstance(case["fields"], dict) or not case["fields"]:
            raise ValueError(f"{path}:{line_no}: fields must be a non-empty object")
        cases.append(case)
    return cases


def _normalize(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    text = re.sub(r"\s+", " ", str(value)).strip().casefold().replace("ё", "е")
    return text.strip(" .,;:!?")


def _parse_number(text: str) -> float | None:
    """Parse a model number string without losing grouped thousands.

    Models often return ``$120,000`` or ``1.234,50`` instead of ``120000`` and
    ``1234.5``.  A naive ``replace(",", ".")`` turns the first value into 120,
    so detect the decimal separator before removing grouping characters.
    """

    compact = re.sub(r"[\s\u00a0\u202f]", "", text)
    match = re.search(r"-?\d+(?:[.,]\d+)*", compact)
    if not match:
        return None
    raw = match.group(0)
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            # European format: 1.234,50 -> 1234.50
            raw = raw.replace(".", "").replace(",", ".")
        else:
            # Anglo format: 1,234.50 -> 1234.50
            raw = raw.replace(",", "")
    elif "," in raw:
        integer, _, fraction = raw.rpartition(",")
        if len(fraction) == 3 and integer.replace(",", "").lstrip("-").isdigit():
            # Grouped integer: 120,000 -> 120000
            raw = raw.replace(",", "")
        else:
            # Decimal comma: 1,5 -> 1.5
            raw = raw.replace(",", ".")
    if raw.count(".") > 1:
        # Remaining dots are grouping separators: 1.234.567 -> 1234567
        raw = raw.replace(".", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _as_expected_type(actual: Any, expected: Any) -> Any:
    """Coerce common model output strings to the expected number/bool type."""

    if isinstance(expected, bool) and isinstance(actual, str):
        lowered = actual.strip().casefold()
        if lowered in {"true", "yes", "да", "1"}:
            return True
        if lowered in {"false", "no", "нет", "0"}:
            return False
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, str)
    ):
        number = _parse_number(actual)
        if number is not None:
            return number
    return actual


def compare_fields(
    expected: dict[str, Any], facts: dict[str, Any]
) -> dict[str, Any]:
    matches: dict[str, Any] = {}
    wrong: dict[str, Any] = {}
    missing: list[str] = []
    for name, wanted in expected.items():
        if name not in facts:
            missing.append(name)
            continue
        actual = facts[name]
        coerced = _as_expected_type(actual, wanted)
        if _normalize(coerced) == _normalize(wanted):
            matches[name] = coerced
        else:
            wrong[name] = {"expected": wanted, "actual": actual}
    total = len(expected)
    matched = len(matches)
    return {
        "total": total,
        "matched": matched,
        "accuracy": matched / total if total else 0.0,
        "matches": matches,
        "wrong": wrong,
        "missing": missing,
    }


def _run_case(
    case: dict[str, Any],
    args: argparse.Namespace,
    temp_dir: Path,
) -> dict[str, Any]:
    out = temp_dir / f"{case['id']}.json"
    command = [
        sys.executable,
        str(COLLECTOR),
        "--task",
        str(case["task"]),
        "--url",
        str(case["url"]),
        "--out",
        str(out),
        "--planner",
        str(args.planner),
        "--extractor",
        str(args.extractor),
        "--device",
        str(args.device),
        "--request-timeout",
        str(args.request_timeout),
    ]
    notes = case.get("field_notes") or {}
    for name, wanted in case["fields"].items():
        if isinstance(wanted, bool):
            kind = "bool"
        elif isinstance(wanted, (int, float)):
            kind = "number"
        else:
            kind = "text"
        note = str(notes.get(name, ""))
        spec = f"{name}:{kind}" + (f":{note}" if note else "")
        command += ["--field", spec]
    if args.llama_server:
        command += ["--llama-server", str(args.llama_server)]
    if args.no_cache:
        command.append("--no-cache")
    if not args.allow_partial:
        command.append("--no-allow-partial")

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "id": case["id"],
            "category": case.get("category", ""),
            "url": case["url"],
            "status": "timeout",
            "error": f"collector exceeded {args.timeout}s",
            "comparison": compare_fields(case["fields"], {}),
        }
    if not out.is_file():
        detail = (process.stderr or process.stdout or "").strip().splitlines()
        return {
            "id": case["id"],
            "category": case.get("category", ""),
            "url": case["url"],
            "status": "collector_error",
            "error": "\n".join(detail[-8:]),
            "comparison": compare_fields(case["fields"], {}),
        }

    report = json.loads(out.read_text(encoding="utf-8"))
    pages = report.get("pages") or []
    page = pages[0] if pages else {}
    facts = {
        str(fact.get("name")): fact.get("value")
        for fact in page.get("facts", [])
        if isinstance(fact, dict)
    }
    comparison = compare_fields(case["fields"], facts)
    status = "exact" if comparison["matched"] == comparison["total"] else "partial"
    if comparison["matched"] == 0:
        status = "miss"
    return {
        "id": case["id"],
        "category": case.get("category", ""),
        "url": case["url"],
        "status": status,
        "page_status": page.get("status"),
        "comparison": comparison,
        "facts": facts,
        "missing_from_page": page.get("missing", []),
        "error": page.get("error"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run real-page benchmark cases through the local collector"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="0 = all cases")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--planner", default=str(DEFAULT_PLANNER))
    parser.add_argument("--extractor", default=str(DEFAULT_EXTRACTOR))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--llama-server", default=None)
    args = parser.parse_args(argv)

    try:
        cases = load_cases(args.cases)
    except (OSError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    if args.limit > 0:
        cases = cases[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="web2spec-bench-") as tmp:
        temp_dir = Path(tmp)
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {case['id']}", file=sys.stderr, flush=True)
            result = _run_case(case, args, temp_dir)
            results.append(result)
            comparison = result["comparison"]
            print(
                f"  {result['status']}: {comparison['matched']}/{comparison['total']} fields",
                file=sys.stderr,
                flush=True,
            )

    fields_total = sum(row["comparison"]["total"] for row in results)
    fields_matched = sum(row["comparison"]["matched"] for row in results)
    exact_cases = sum(1 for row in results if row["status"] == "exact")
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": str(args.cases),
        "engine": {
            "planner": str(args.planner),
            "extractor": str(args.extractor),
            "device": str(args.device),
        },
        "summary": {
            "cases": len(results),
            "exact_cases": exact_cases,
            "case_accuracy": exact_cases / len(results) if results else 0.0,
            "fields": fields_total,
            "matched_fields": fields_matched,
            "field_accuracy": fields_matched / fields_total if fields_total else 0.0,
            "wrong_fields": sum(len(row["comparison"]["wrong"]) for row in results),
            "missing_fields": sum(len(row["comparison"]["missing"]) for row in results),
        },
        "cases": results,
    }
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary = payload["summary"]
    print(
        f"Готово: {summary['exact_cases']}/{summary['cases']} кейсов точно, "
        f"{summary['matched_fields']}/{summary['fields']} полей -> {args.out}"
    )
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
