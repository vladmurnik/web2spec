"""Step 1: normalize raw JSONL, validate, deduplicate, split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web2spec.config import REPO_ROOT, load_data_config  # noqa: E402
from web2spec.html_compress import compress_html  # noqa: E402
from web2spec.schema import SampleError, load_task_schemas  # noqa: E402
from web2spec.validate import (  # noqa: E402
    split_samples,
    validate_file,
    write_jsonl,
)


def _iter_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    files = [
        p
        for p in path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".jsonl", ".json"}
        and not p.name.startswith("quarantine")
    ]
    return sorted(files)


def _compress_input(source: Path, target: Path, max_chars: int, limit: int | None) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source.open("r", encoding="utf-8-sig") as src, target.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            if limit is not None and written >= limit:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dst.write(line)
                written += 1
                continue
            for key in ("input_html", "html", "page"):
                value = record.get(key)
                if isinstance(value, str) and value:
                    record[key] = compress_html(value, max_chars=max_chars)
                    break
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Валидация и разбиение датасета web2spec")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/processed")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--no-tokenize", action="store_true")
    parser.add_argument("--quarantine", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_data_config(args.config)
    paths = cfg.paths.resolve()
    schemas = load_task_schemas((REPO_ROOT / cfg.task_schemas).resolve())
    report_path = args.report or (paths.reports / "validation.json")
    quarantine = args.quarantine or (paths.raw / "quarantine.jsonl")

    tokenizer = None
    if not args.no_tokenize:
        try:
            from web2spec.tokens import load_tokenizer

            tokenizer = load_tokenizer(cfg.tokenizer)
            print(f"Токенизатор: {cfg.tokenizer}")
        except Exception as exc:  # transformers may still be installing
            print(f"! Токенизатор недоступен, проверка длины пропущена: {exc}")

    sources = _iter_inputs(args.input)
    if not sources:
        print(f"Нет входных файлов в {args.input}")
        return 1

    interim = paths.interim
    all_samples = []
    total_failed = 0
    file_reports: dict[str, dict] = {}
    for source in sources:
        working = source
        if not args.no_compress:
            working = interim / f"{source.stem}.compressed.jsonl"
            _compress_input(
                source,
                working,
                cfg.compress_max_chars,
                args.limit,
            )
        print(f"Обработка: {source.name} -> {working.name}")
        samples, report = validate_file(
            working,
            cfg=cfg,
            task_schemas=schemas,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            dedup=not args.no_dedup,
            quarantine=quarantine,
        )
        all_samples.extend(samples)
        total_failed += report.failed
        file_reports[source.name] = report.model_dump(mode="json")
        print(
            f"  всего={report.total} принято={report.passed} отклонено={report.failed} "
            f"причины={report.by_reason or '{}'}"
        )

    if args.limit is not None:
        all_samples = all_samples[: args.limit]

    splits = split_samples(all_samples, val_ratio=cfg.val_ratio, seed=cfg.seed)
    for name in ("train", "val", "test"):
        if splits[name]:
            path = write_jsonl(splits[name], args.output / f"{name}.jsonl")
            print(f"  {name}: {len(splits[name])} -> {path}")

    combined = {
        "files": file_reports,
        "total": sum(item["total"] for item in file_reports.values()),
        "passed": sum(item["passed"] for item in file_reports.values()),
        "failed": total_failed,
        "by_reason": {
            code: sum(item["by_reason"].get(code, 0) for item in file_reports.values())
            for code in {code for item in file_reports.values() for code in item["by_reason"]}
        },
        "splits": {name: len(items) for name, items in splits.items()},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    if not all_samples:
        print("Ни одного валидного примера. Смотри отчёт.")
    lengths = [s.meta.n_tokens for s in all_samples if s.meta.n_tokens is not None]
    if lengths:
        ordered = sorted(lengths)
        print(
            f"Токены: p50={ordered[len(ordered) // 2]} "
            f"p95={ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]} "
            f"max={ordered[-1]}"
        )
    print(f"Отклонено всего: {total_failed}. Карантин: {quarantine}")
    print(f"Отчёт: {report_path}")
    return 0 if all_samples else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, SampleError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
