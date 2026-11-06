"""Unified command line for the web2spec pipeline.

The numbered scripts remain runnable on their own; this module is a stable
entry point for day-to-day use::

    web2spec collect --task "..." --url https://...
    web2spec search  --task "..." --provider ddg
    web2spec mcp
    web2spec doctor

It also exposes the training/evaluation scripts so the repository has one
predictable interface instead of a pile of unrelated files.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from web2spec.llama_runtime import (
    DEFAULT_EXTRACTOR,
    DEFAULT_PLANNER,
    resolve_server,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts"
VERSION = "0.3.0"

COMMANDS = {
    "collect": "21_collect_urls.py",
    "search": "22_search_and_collect.py",
    "research": "22_search_and_collect.py",
    "mcp": "23_mcp_server.py",
    "eval-q4": "24_eval_q4km.py",
    "eval-runtime": "20_eval_runtime.py",
    "eval-plan-extract": "19_eval_plan_extract.py",
    "eval-plan": "17_eval_plan.py",
    "train": "02_train.py",
    "merge": "03_merge.py",
}


def _load(script: str) -> ModuleType:
    path = SCRIPT_DIR / script
    if not path.is_file():
        raise FileNotFoundError(f"скрипт не найден: {path}")
    name = f"web2spec_script_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"не удалось загрузить {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _doctor() -> int:
    checks: list[tuple[str, bool, str]] = []
    for label, path in (
        ("planner Q4_K_M", DEFAULT_PLANNER),
        ("extractor Q4_K_M", DEFAULT_EXTRACTOR),
    ):
        size = path.stat().st_size / (1024**2) if path.is_file() else 0.0
        checks.append((label, path.is_file(), f"{path} ({size:.1f} MB)"))
    try:
        server = resolve_server()
        checks.append(("llama-server", True, server))
    except FileNotFoundError as exc:
        checks.append(("llama-server", False, str(exc)))
    checks.append(("Python", True, sys.version.split()[0]))
    checks.append(("Repository", True, str(REPO_ROOT)))

    failed = 0
    for label, ok, detail in checks:
        status = "OK  " if ok else "MISS"
        print(f"[{status}] {label}: {detail}")
        failed += int(not ok)
    return 0 if failed == 0 else 1


def _help() -> str:
    lines = [
        f"web2spec {VERSION}",
        "",
        "Usage: web2spec <command> [options]",
        "",
        "Commands:",
        "  collect           task + URLs -> structured facts",
        "  search            task -> local search -> facts",
        "  research          alias for search (one-call research)",
        "  mcp               MCP stdio server for any MCP client",
        "  doctor            check models, llama-server and paths",
        "  eval-q4           Q4_K_M quality/speed evaluation",
        "  eval-runtime      fp16 runtime cascade evaluation",
        "  eval-plan         planner evaluation",
        "  eval-plan-extract extractor evaluation",
        "  train             QLoRA training",
        "  merge             merge adapter into base model",
        "",
        "Run `web2spec <command> --help` for command options.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(_help())
        return 0
    if args[0] in {"-V", "--version"}:
        print(VERSION)
        return 0

    command = args[0]
    if command == "doctor":
        return _doctor()
    script = COMMANDS.get(command)
    if script is None:
        print(f"unknown command: {command}\n\n{_help()}", file=sys.stderr)
        return 2
    return int(_load(script).main(args[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
