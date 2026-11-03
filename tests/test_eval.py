from __future__ import annotations

import pytest

from web2spec.eval import (
    _finalize_buckets,
    evaluate_records,
    extract_json,
    field_scores,
    flatten,
    normalize,
)


def test_flatten_nested() -> None:
    flat = flatten({"a": {"b": 1}, "c": [1, {"d": 2}]})
    assert flat == {"a.b": 1, "c[0]": 1, "c[1].d": 2}


def test_normalize_whitespace_and_numbers() -> None:
    assert normalize("  a   b ") == "a b"
    assert normalize(3) == 3.0
    assert normalize(True) is True


def test_exact_match() -> None:
    scores = field_scores({"a": 1, "b": "x"}, {"a": 1, "b": " x "})
    assert scores["exact"] is True
    assert scores["f1"] == 1.0


def test_partial_match_hand_computed() -> None:
    scores = field_scores({"a": 1, "b": 2}, {"a": 1, "c": 3})
    assert scores["matched"] == 1
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["f1"] == 0.5
    assert scores["exact"] is False


def test_boolean_is_not_number() -> None:
    scores = field_scores({"a": True}, {"a": 1})
    assert scores["exact"] is False


def test_extract_fenced_json() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_prose() -> None:
    assert extract_json('Вот ответ: {"a": {"b": 2}} конец') == {"a": {"b": 2}}


def test_extract_json_trailing_comma() -> None:
    assert extract_json('{"a": 1,}') == {"a": 1}


def test_extract_json_invalid() -> None:
    assert extract_json("не json") is None
    assert extract_json('{"a": ') is None


def test_evaluate_invalid_prediction() -> None:
    report = evaluate_records([{"pred": "мусор", "gold": {"a": 1}, "domain": "d"}])
    assert report["json_valid_rate"] == 0.0
    assert report["count"] == 1
    assert report["details"][0]["pred_fields"] == 0


def test_evaluate_domain_breakdown() -> None:
    records = [
        {"pred": '{"a": 1}', "gold": {"a": 1}, "domain": "shop", "complexity": 1},
        {"pred": '{"a": 2}', "gold": {"a": 1}, "domain": "shop", "complexity": 2},
        {"pred": '{"a": 1}', "gold": {"a": 1}, "domain": "news", "complexity": 1},
    ]
    report = evaluate_records(records)
    _finalize_buckets(report)
    assert report["count"] == 3
    assert report["by_domain"]["shop"]["count"] == 2
    assert report["by_domain"]["shop"]["exact_match"] == 0.5
    assert report["by_domain"]["news"]["exact_match"] == 1.0
    assert report["by_complexity"]["1"]["count"] == 2


def test_evaluate_empty() -> None:
    report = evaluate_records([])
    assert report["count"] == 0
    assert report["json_valid_rate"] == 0.0
    assert report["micro_f1"] == 0.0


def test_micro_vs_macro() -> None:
    records = [
        {"pred": {"a": 1}, "gold": {"a": 1}, "domain": "x"},
        {"pred": {"a": 1, "b": 2}, "gold": {"a": 1}, "domain": "x"},
    ]
    report = evaluate_records(records)
    assert report["micro_precision"] == pytest.approx(2 / 3)
    assert report["micro_recall"] == 1.0
    assert report["macro_precision"] == pytest.approx(0.75)
