"""Evaluate the runtime cascade with llama.cpp Q4_K_M models.

This is the same 50-sample test as ``20_eval_runtime.py``, but both production
models are loaded from GGUF Q4_K_M through ``llama-server`` instead of PyTorch
fp16.  The planner and extractor run one after another, so the 4 GB GPU only
holds one server at a time.

The report keeps the same quality metrics as the fp16 runtime evaluation and
adds llama.cpp prompt/generation timings.  The held-out inputs are already
compressed, so the metadata header added by the new compressor only appears on
freshly downloaded pages, not in this fixture.  That limitation is reported
explicitly instead of being hidden in the numbers.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from web2spec.eval import extract_json  # noqa: E402
from web2spec.plan import (  # noqa: E402
    CollectionPlan,
    PlanError,
    load_plan,
)
from web2spec.schema import Meta, Sample  # noqa: E402
from web2spec.tokens import (  # noqa: E402
    build_plan_extract_prompt,
    build_plan_prompt,
    load_tokenizer,
)
from web2spec.validate import load_samples  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "runtime_eval", REPO_ROOT / "scripts" / "20_eval_runtime.py"
)
assert _spec is not None and _spec.loader is not None
runtime_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runtime_eval)

DEFAULT_PLANNER = REPO_ROOT / "models/gguf/qwen05b-plan-Q4_K_M.gguf"
DEFAULT_EXTRACTOR = REPO_ROOT / "models/gguf/qwen05b-plan-extract-v2-Q4_K_M.gguf"


def _find_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _gpu_memory_mb() -> float:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    values: list[float] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            with contextlib.suppress(ValueError):
                values.append(float(line))
    return max(values) if values else 0.0


def server_command(
    server_bin: str,
    model_path: Path,
    port: int,
    context: int = 8192,
    device: str = "Vulkan0",
) -> list[str]:
    command = [
        server_bin,
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(context),
        "-b",
        "2048",
        "-ub",
        "512",
        "-ngl",
        "99",
        "-np",
        "1",
        "--no-webui",
    ]
    if device:
        command.extend(["--device", device])
    return command


def _wait_health(port: int, proc: subprocess.Popen, timeout: float) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"llama-server did not become healthy in {timeout:.0f}s")


def _completion(
    port: int,
    prompt: str,
    max_new_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "n_predict": max_new_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "repeat_penalty": 1.0,
        "seed": 42,
        "cache_prompt": False,
        "stream": False,
        "stop": ["<|im_end|>"],
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict) or "content" not in body:
        raise RuntimeError(f"unexpected /completion response: {str(body)[:500]}")
    return body


def row_from_response(body: dict[str, Any], request_seconds: float) -> dict[str, Any]:
    timings = body.get("timings") or {}
    prompt_tokens = int(timings.get("prompt_n") or body.get("tokens_evaluated") or 0)
    generated_tokens = int(
        timings.get("predicted_n") or body.get("tokens_predicted") or 0
    )
    prompt_seconds = float(timings.get("prompt_ms") or 0.0) / 1000.0
    generate_seconds = float(timings.get("predicted_ms") or 0.0) / 1000.0
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prompt_seconds": prompt_seconds,
        "generate_seconds": generate_seconds,
        "tokens_per_second": (
            generated_tokens / generate_seconds if generate_seconds else 0.0
        ),
        "request_seconds": round(request_seconds, 6),
    }


def run_server(
    *,
    label: str,
    server_bin: str,
    model_path: Path,
    items: list[tuple[Sample, str]],
    max_new_tokens: int,
    context: int,
    device: str,
    startup_timeout: float,
    request_timeout: float,
    log_path: Path,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    port = _find_port()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8", errors="replace")
    command = server_command(server_bin, model_path, port, context, device)
    started = time.perf_counter()
    proc = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
    stats: dict[str, Any] = {}
    output: dict[str, str] = {}
    max_vram = 0.0
    try:
        _wait_health(port, proc, startup_timeout)
        stats["_load_seconds"] = round(time.perf_counter() - started, 3)
        max_vram = max(max_vram, _gpu_memory_mb())
        for index, (sample, prompt) in enumerate(items, 1):
            request_started = time.perf_counter()
            body = _completion(port, prompt, max_new_tokens, request_timeout)
            request_seconds = time.perf_counter() - request_started
            text = str(body.get("content") or "")
            output[sample.id] = text
            row = row_from_response(body, request_seconds)
            stats[sample.id] = row
            if index % 5 == 0 or index == len(items):
                max_vram = max(max_vram, _gpu_memory_mb())
            print(
                f"[{label} {index}/{len(items)}] {sample.id}: "
                f"{len(text)} chars, {row['generated_tokens']} tok, "
                f"{row['generate_seconds']:.2f}s, "
                f"{row['tokens_per_second']:.1f} tok/s"
            )
    except Exception:
        log_handle.flush()
        tail = ""
        with contextlib.suppress(OSError):
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-25:])
        raise RuntimeError(f"{label} server failed; log tail:\n{tail}") from None
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_handle.close()
        stats["_max_vram_mb"] = round(max_vram, 1)
    meta = {
        "model": str(model_path),
        "model_mb": round(model_path.stat().st_size / (1024**2), 1),
        "port": port,
        "command": command,
        "log": str(log_path),
    }
    return output, stats, meta


def compare_reports(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    def metric(report: dict[str, Any], path: str) -> float | None:
        value: Any = report
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return float(value) if isinstance(value, (int, float)) else None

    paths = [
        ("fact_f1", "fact_f1"),
        ("task_success_rate", "task_success_rate"),
        ("fact_precision", "fact_precision"),
        ("fact_recall", "fact_recall"),
        ("wrong_values", "wrong_values"),
        ("planner_tps", "speed.planner.tokens_per_second"),
        ("extractor_tps", "speed.extractor.tokens_per_second"),
        ("wall_seconds", "speed.wall_seconds"),
    ]
    rows: dict[str, Any] = {}
    for name, path in paths:
        base = metric(baseline, path)
        cur = metric(current, path)
        rows[name] = {
            "baseline": base,
            "current": cur,
            "delta": round(cur - base, 4) if base is not None and cur is not None else None,
        }
    return rows


def _build_gold_plans(path: Path) -> dict[tuple[str | None, str], CollectionPlan]:
    gold_plans: dict[tuple[str | None, str], CollectionPlan] = {}
    if not path.exists():
        return gold_plans
    for sample in load_samples(path):
        try:
            gold_plans[(sample.meta.source_url, sample.instruction)] = load_plan(
                sample.target
            )
        except PlanError:
            continue
    return gold_plans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Оценка runtime-каскада на llama.cpp Q4_K_M"
    )
    parser.add_argument("--planner", type=Path, default=DEFAULT_PLANNER)
    parser.add_argument("--extractor", type=Path, default=DEFAULT_EXTRACTOR)
    parser.add_argument(
        "--planner-tokenizer",
        type=Path,
        default=REPO_ROOT / "models/merged/qwen05b-plan",
    )
    parser.add_argument(
        "--extractor-tokenizer",
        type=Path,
        default=REPO_ROOT / "models/merged/qwen05b-plan-extract-v2",
    )
    parser.add_argument("--server-bin", type=str, default="llama-server")
    parser.add_argument(
        "--device",
        type=str,
        default="Vulkan0",
        help="llama.cpp device (Vulkan0 = RTX 3050 Ti; empty = auto)",
    )
    parser.add_argument(
        "--data", type=Path, default=REPO_ROOT / "data/plan_extract/test.jsonl"
    )
    parser.add_argument(
        "--gold-plans", type=Path, default=REPO_ROOT / "data/plan/test.jsonl"
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--plan-max-new-tokens", type=int, default=320)
    parser.add_argument("--fact-max-new-tokens", type=int, default=384)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument(
        "--report", type=Path, default=REPO_ROOT / "data/reports/eval_q4km.json"
    )
    parser.add_argument(
        "--baseline", type=Path, default=REPO_ROOT / "data/reports/eval_speed_quality.json"
    )
    parser.add_argument("--save-predictions", type=Path, default=None)
    parser.add_argument(
        "--repair-buffer",
        type=Path,
        default=REPO_ROOT / "data/reports/q4km_repair_buffer.jsonl",
    )
    args = parser.parse_args(argv)

    samples = load_samples(args.data)
    if args.limit:
        samples = samples[: args.limit]
    print(f"Загружено {len(samples)} примеров; planner={args.planner.name}")

    planner_tokenizer = load_tokenizer(str(args.planner_tokenizer))
    # GGUF has no tokenizer config next to it; use the merged HF tokenizer only
    # to render the exact same chat prompt as in training.
    if not hasattr(planner_tokenizer, "apply_chat_template"):
        raise RuntimeError("planner tokenizer has no chat template")

    plan_items = [
        (sample, build_plan_prompt(sample, planner_tokenizer)) for sample in samples
    ]
    plans, plan_stats, plan_meta = run_server(
        label="q4-plan",
        server_bin=args.server_bin,
        model_path=args.planner,
        items=plan_items,
        max_new_tokens=args.plan_max_new_tokens,
        context=args.context,
        device=args.device,
        startup_timeout=args.startup_timeout,
        request_timeout=args.request_timeout,
        log_path=REPO_ROOT / "data/reports/q4km_planner_server.log",
    )

    extract_samples: list[Sample] = []
    plan_rows: dict[str, str] = {}
    for sample in samples:
        try:
            plan = load_plan(extract_json(plans.get(sample.id, "")))
        except (PlanError, TypeError):
            plan = None
        if plan is None:
            continue
        plan_rows[sample.id] = json.dumps(
            plan.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
        )
        extract_samples.append(
            Sample(
                id=sample.id,
                group_id=sample.group_id,
                task="plan_extract",
                instruction=sample.instruction,
                input_html=sample.input_html,
                target={"facts": []},
                meta=Meta(
                    lang=sample.meta.lang,
                    complexity=sample.meta.complexity,
                    source_url=sample.meta.source_url,
                    pipeline_version="runtime-q4km-0.1.0",
                    plan=plan.model_dump(mode="json"),
                ),
            )
        )

    extractor_tokenizer = load_tokenizer(str(args.extractor_tokenizer))
    fact_items = [
        (sample, build_plan_extract_prompt(sample, extractor_tokenizer))
        for sample in extract_samples
    ]
    factsets, fact_stats, fact_meta = run_server(
        label="q4-fact",
        server_bin=args.server_bin,
        model_path=args.extractor,
        items=fact_items,
        max_new_tokens=args.fact_max_new_tokens,
        context=args.context,
        device=args.device,
        startup_timeout=args.startup_timeout,
        request_timeout=args.request_timeout,
        log_path=REPO_ROOT / "data/reports/q4km_extractor_server.log",
    )

    wall_seconds = round(
        float(plan_stats.get("_load_seconds", 0.0))
        + float(fact_stats.get("_load_seconds", 0.0))
        + sum(
            row["request_seconds"]
            for stats in (plan_stats, fact_stats)
            for key, row in stats.items()
            if not key.startswith("_")
        ),
        3,
    )
    result = runtime_eval.evaluate(
        samples,
        plans,
        factsets,
        _build_gold_plans(args.gold_plans),
        args.repair_buffer,
        repair=True,
    )
    result["speed"] = {
        "planner": runtime_eval.summarize_timings(plan_stats),
        "extractor": runtime_eval.summarize_timings(fact_stats),
        "wall_seconds": wall_seconds,
    }
    result["backend"] = {
        "name": "llama.cpp",
        "quantization": "Q4_K_M",
        "gpu": "RTX 3050 Ti Laptop 4 GB (Vulkan)",
        "planner": plan_meta,
        "extractor": fact_meta,
        "notes": [
            "Q4_K_M quantization fell back to q5_0/q6_K/q8_0 for 144/290 tensors "
            "because hidden size 896 is not divisible by 256; effective ~6.35 BPW.",
            "The held-out input_html is already compressed and contains no "
            "<meta>/JSON-LD/datetime attributes, so the new metadata header does "
            "not appear in this fixture. It only affects freshly downloaded pages.",
            "Quality is comparable with the fp16 run only for the same fixture; "
            "the fp16 baseline used the previous system prompts.",
        ],
    }
    if args.baseline.exists():
        try:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            result["comparison_to_fp16"] = compare_reports(baseline, result)
        except (json.JSONDecodeError, OSError):
            pass

    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.save_predictions.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in samples:
                handle.write(
                    json.dumps(
                        {
                            "id": sample.id,
                            "source_url": sample.meta.source_url,
                            "instruction": sample.instruction,
                            "plan": plans.get(sample.id, ""),
                            "parsed_plan": plan_rows.get(sample.id),
                            "factset": factsets.get(sample.id, ""),
                            "plan_timing": {
                                key: value
                                for key, value in plan_stats.get(
                                    sample.id, {}
                                ).items()
                            },
                            "fact_timing": {
                                key: value
                                for key, value in fact_stats.get(
                                    sample.id, {}
                                ).items()
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {key: value for key, value in result.items() if key != "details"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Отчёт: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
