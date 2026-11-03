"""Step 5: evaluate merged model or saved predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web2spec.config import REPO_ROOT, load_eval_config  # noqa: E402
from web2spec.eval import _finalize_buckets, evaluate_records, run  # noqa: E402


def _load_predictions(path: Path, limit: int | None) -> list[dict]:
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
    parser = argparse.ArgumentParser(description="Оценка качества извлечения JSON")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_eval_config(args.config)
    if args.limit is not None:
        cfg.limit = args.limit

    if args.predictions:
        report = evaluate_records(_load_predictions(args.predictions, args.limit))
        _finalize_buckets(report)
        print(json.dumps(_summary(report), ensure_ascii=False, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Отчёт: {args.report}")
        return 0

    report = run(cfg)
    print(json.dumps(_summary(report), ensure_ascii=False, indent=2))
    print(f"Отчёт: {(REPO_ROOT / cfg.report).resolve()}")
    return 0


def _summary(report: dict) -> dict:
    keys = (
        "count",
        "json_valid_rate",
        "exact_match",
        "micro_f1",
        "macro_f1",
        "micro_precision",
        "micro_recall",
    )
    return {key: report.get(key) for key in keys}


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
