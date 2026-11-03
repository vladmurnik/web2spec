from __future__ import annotations

import pytest

from web2spec.config import REPO_ROOT
from web2spec.schema import (
    SampleError,
    length_stats,
    load_task_schemas,
    make_id,
    validate_sample,
)

TASK_SCHEMAS = load_task_schemas(REPO_ROOT / "configs/tasks.yaml")


def raw_sample(**overrides):
    payload = {
        "id": "extract-0123456789ab",
        "group_id": "shop.example.com",
        "task": "extract",
        "instruction": "Extract the product name, price and currency.",
        "input_html": (
            "<html><body><h1>Widget</h1>"
            "<span class='price'>19.99 EUR</span></body></html>"
        ),
        "target": {"name": "Widget", "price": 19.99, "currency": "EUR"},
        "meta": {"lang": "en", "complexity": 1, "source_url": "https://shop.example.com/widget"},
    }
    payload.update(overrides)
    return payload


def test_valid_sample_passes():
    sample = validate_sample(raw_sample(), TASK_SCHEMAS)
    assert sample.task == "extract"
    assert sample.target["price"] == 19.99
    assert sample.meta.complexity == 1


def test_missing_field_is_schema_error():
    payload = raw_sample()
    del payload["instruction"]
    with pytest.raises(SampleError) as exc:
        validate_sample(payload, TASK_SCHEMAS)
    assert exc.value.code == "E_SCHEMA"
    assert "instruction" in exc.value.message


def test_empty_target_is_rejected():
    with pytest.raises(SampleError) as exc:
        validate_sample(raw_sample(target={}), TASK_SCHEMAS)
    assert exc.value.code == "E_SCHEMA"


def test_unknown_task_is_rejected():
    with pytest.raises(SampleError) as exc:
        validate_sample(raw_sample(task="summarize"), TASK_SCHEMAS)
    assert exc.value.code == "E_SCHEMA"


def test_target_schema_mismatch_is_reported():
    payload = raw_sample(task="spec", target={"version": 1})
    with pytest.raises(SampleError) as exc:
        validate_sample(payload, TASK_SCHEMAS)
    assert exc.value.code == "E_TARGET"
    assert "fields" in exc.value.message


def test_control_characters_in_html_are_rejected():
    with pytest.raises(SampleError) as exc:
        validate_sample(raw_sample(input_html="<html>\x00</html>"), TASK_SCHEMAS)
    assert exc.value.code == "E_SCHEMA"


def test_non_object_payload_is_rejected():
    with pytest.raises(SampleError) as exc:
        validate_sample(["not", "an", "object"], TASK_SCHEMAS)
    assert exc.value.code == "E_NOT_OBJECT"


def test_make_id_is_deterministic_and_stable():
    first = make_id("extract", "https://a.example", 3)
    second = make_id("extract", "https://a.example", 3)
    assert first == second
    assert first.startswith("extract-")
    assert len(first) <= 64


def test_length_stats_nearest_rank():
    stats = length_stats(list(range(1, 101)))
    assert stats == {"p50": 50, "p95": 95, "p99": 99, "max": 100}


def test_length_stats_empty():
    assert length_stats([]) == {"p50": None, "p95": None, "p99": None, "max": None}


def test_task_schemas_are_loaded():
    assert set(TASK_SCHEMAS) == {"extract", "spec", "qa"}
    assert TASK_SCHEMAS["spec"]["required"] == ["version", "fields"]
