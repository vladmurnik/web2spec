from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "24_eval_q4km.py"
_spec = importlib.util.spec_from_file_location("q4km_eval", SCRIPT)
assert _spec is not None and _spec.loader is not None
q4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q4)


def test_row_from_timings() -> None:
    row = q4.row_from_response(
        {
            "timings": {
                "prompt_n": 1000,
                "predicted_n": 100,
                "prompt_ms": 250.0,
                "predicted_ms": 500.0,
            }
        },
        request_seconds=0.81,
    )
    assert row["prompt_tokens"] == 1000
    assert row["generated_tokens"] == 100
    assert row["prompt_seconds"] == 0.25
    assert row["generate_seconds"] == 0.5
    assert row["tokens_per_second"] == 200.0
    assert row["request_seconds"] == 0.81


def test_row_from_legacy_response_fields() -> None:
    row = q4.row_from_response(
        {"tokens_evaluated": 50, "tokens_predicted": 10}, request_seconds=1.0
    )
    assert row["prompt_tokens"] == 50
    assert row["generated_tokens"] == 10
    assert row["generate_seconds"] == 0.0
    assert row["tokens_per_second"] == 0.0


def test_compare_reports_marks_deltas() -> None:
    baseline = {
        "fact_f1": 0.9,
        "task_success_rate": 0.8,
        "wrong_values": 2,
        "speed": {
            "planner": {"tokens_per_second": 20.0},
            "extractor": {"tokens_per_second": 22.0},
            "wall_seconds": 500.0,
        },
    }
    current = {
        "fact_f1": 1.0,
        "task_success_rate": 1.0,
        "wrong_values": 0,
        "speed": {
            "planner": {"tokens_per_second": 170.0},
            "extractor": {"tokens_per_second": 160.0},
            "wall_seconds": 100.0,
        },
    }
    rows = q4.compare_reports(baseline, current)
    assert rows["fact_f1"]["delta"] == 0.1
    assert rows["planner_tps"]["delta"] == 150.0
    assert rows["wall_seconds"]["delta"] == -400.0
    assert rows["extractor_tps"]["current"] == 160.0


def test_compare_reports_missing_values_are_none() -> None:
    rows = q4.compare_reports({}, {})
    assert rows["fact_f1"] == {"baseline": None, "current": None, "delta": None}


def test_server_command_contains_required_flags() -> None:
    command = q4.server_command(
        "llama-server.exe", Path("model.gguf"), 8123, context=4096, device="Vulkan0"
    )
    joined = " ".join(command)
    assert "llama-server.exe" in joined
    assert "model.gguf" in joined
    assert "--port 8123" in joined
    assert "-c 4096" in joined
    assert "-ngl 99" in joined
    assert "--device Vulkan0" in joined


def test_server_command_can_omit_device() -> None:
    command = q4.server_command("llama-server.exe", Path("m.gguf"), 1, device="")
    assert "--device" not in command
