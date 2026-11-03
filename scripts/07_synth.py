"""Step 0: generate synthetic samples with a teacher model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web2spec.config import REPO_ROOT, load_data_config  # noqa: E402
from web2spec.schema import load_task_schemas  # noqa: E402
from web2spec.synth import OpenAICompatTeacher, build_messages, synthesize  # noqa: E402


def _read_inputs(path: Path, limit: int | None) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Синтетические данные через teacher-модель")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/raw/synthetic.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="qwen2.5-coder-32b-instruct")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--task", default="extract", choices=["extract", "spec", "qa"])
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_data_config()
    schemas = load_task_schemas((REPO_ROOT / cfg.task_schemas).resolve())
    schema = schemas.get(args.task, {"type": "object"})
    inputs = _read_inputs(args.input, args.limit)

    if args.dry_run:
        if not inputs:
            print("Вход пуст.")
            return 1
        system, user = build_messages(inputs[0], schema)
        print("=== SYSTEM ===")
        print(system)
        print("=== USER (first 1500 chars) ===")
        print(user[:1500])
        print(f"\nЗаписей: {len(inputs)}. Сеть не использовалась.")
        return 0

    teacher = OpenAICompatTeacher(base_url=args.base_url, model=args.model)
    samples, stats = synthesize(
        inputs, teacher, schema=schema, task=args.task, min_chars=args.min_chars
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Записано: {len(samples)} -> {args.output}")
    return 0 if samples else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
