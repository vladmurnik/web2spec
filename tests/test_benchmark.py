from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "27_benchmark.py"
_spec = importlib.util.spec_from_file_location("web2spec_benchmark", SCRIPT)
assert _spec is not None and _spec.loader is not None
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)


def test_real_cases_load() -> None:
    cases = bench.load_cases()
    assert len(cases) >= 7
    assert {case["id"] for case in cases} >= {
        "github-actions-limits",
        "stripe-api-errors",
        "anthropic-pricing",
    }
    assert all(case["fields"] for case in cases)


def test_compare_fields_accepts_model_number_strings() -> None:
    result = bench.compare_fields(
        {"price": 20, "hours": "6 hours", "enabled": False},
        {"price": "$20", "hours": "6 hours", "enabled": "false"},
    )
    assert result["matched"] == 3
    assert result["accuracy"] == 1.0
    assert result["wrong"] == {}


def test_compare_fields_reports_wrong_and_missing() -> None:
    result = bench.compare_fields(
        {"price": 20, "missing": "x"},
        {"price": 25},
    )
    assert result["matched"] == 0
    assert "missing" in result["missing"]
    assert result["wrong"]["price"] == {"expected": 20, "actual": 25}
