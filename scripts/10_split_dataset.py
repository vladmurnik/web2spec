"""Stratified train/val/test split of accepted samples.

The collector uses a host as ``group_id``, so a fully group-aware split would
put an entire site into one split. For a small multi-source dataset we instead
stratify *within* each group: every host/category contributes to train, val and
test. Exact duplicates are already removed by validation.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web2spec.schema import Sample  # noqa: E402
from web2spec.validate import load_samples, write_jsonl  # noqa: E402


def stratified_split(
    samples: list[Sample],
    *,
    val_ratio: float = 0.07,
    test_ratio: float = 0.07,
    seed: int = 42,
) -> dict[str, list[Sample]]:
    by_group: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_group[sample.group_id].append(sample)

    splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for group_id in sorted(by_group):
        items = list(by_group[group_id])
        random.Random(f"{seed}:{group_id}").shuffle(items)
        n = len(items)
        n_val = max(1, round(n * val_ratio)) if n >= 10 else 0
        n_test = max(1, round(n * test_ratio)) if n >= 10 else 0
        if n - n_val - n_test < 1:
            n_val = n_test = 0
        splits["val"].extend(items[:n_val])
        splits["test"].extend(items[n_val : n_val + n_test])
        splits["train"].extend(items[n_val + n_test :])
    return splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Стратифицированное разбиение web2spec")
    parser.add_argument("--input", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-ratio", type=float, default=0.07)
    parser.add_argument("--test-ratio", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    sources = [args.input / "train.jsonl", args.input / "val.jsonl"]
    samples: list[Sample] = []
    for path in sources:
        if path.exists():
            samples.extend(load_samples(path))
    if not samples:
        print("нет данных для разбиения", file=sys.stderr)
        return 1

    splits = stratified_split(
        samples, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )
    ids = [sample.id for items in splits.values() for sample in items]
    if len(ids) != len(set(ids)):
        print("ошибка: пересечение id между splits", file=sys.stderr)
        return 2

    for name, items in splits.items():
        path = write_jsonl(items, args.input / f"{name}.jsonl")
        tasks = Counter(sample.task for sample in items)
        groups = Counter(sample.group_id for sample in items)
        print(f"{name}: {len(items)} -> {path} tasks={dict(tasks)} groups={len(groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
