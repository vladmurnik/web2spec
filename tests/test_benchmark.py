from __future__ import annotations

import importlib.util
import re
import sys
from datetime import date
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


def test_cases_have_unique_ids_and_typed_fields() -> None:
    cases = bench.load_cases()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert case["url"].startswith("https://")
        date.fromisoformat(case["checked_at"])
        for name, value in case["fields"].items():
            assert re.fullmatch(r"[a-z][a-z0-9_]+", name)
            assert isinstance(value, (str, int, float, bool))


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
