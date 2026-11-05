from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "20_eval_runtime.py"
_spec = importlib.util.spec_from_file_location("eval_runtime", SCRIPT)
assert _spec is not None and _spec.loader is not None
eval_runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_runtime)


def test_summarize_timings_aggregates_tokens_and_percentiles() -> None:
    stats = {
        "_load_seconds": 1.5,
        "_max_vram_mb": 998.0,
        "a": {
            "prompt_tokens": 100,
            "generated_tokens": 20,
            "prompt_seconds": 0.01,
            "generate_seconds": 1.0,
            "tokens_per_second": 20.0,
        },
        "b": {
            "prompt_tokens": 200,
            "generated_tokens": 40,
            "prompt_seconds": 0.02,
            "generate_seconds": 2.0,
            "tokens_per_second": 20.0,
        },
    }

    summary = eval_runtime.summarize_timings(stats)

    assert summary["samples"] == 2
    assert summary["load_seconds"] == 1.5
    assert summary["max_vram_mb"] == 998.0
    assert summary["prompt_tokens_total"] == 300
    assert summary["generated_tokens_total"] == 60
    assert summary["tokens_per_second"] == 20.0
    assert summary["seconds_per_sample_mean"] == 1.5
    assert summary["seconds_per_sample_median"] == 1.5
    assert summary["seconds_per_sample_p95"] == 2.0
    assert summary["generated_tokens_mean"] == 30.0


def test_summarize_timings_handles_empty_stats() -> None:
    assert eval_runtime.summarize_timings({"_load_seconds": 2.0}) == {"samples": 0}
