"""Collect HTML -> JSON samples from public pages that publish schema.org JSON-LD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from web2spec.collect import (  # noqa: E402
    CATEGORY_PATTERNS,
    collect,
    load_seeds,
    urls_from_sitemap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, choices=["products", "articles", "listings"])
    parser.add_argument("--seeds", type=Path, default=None, help="one URL per line")
    parser.add_argument(
        "--sitemap",
        action="append",
        default=[],
        help="sitemap index URL; can be repeated",
    )
    parser.add_argument("--out", type=Path, required=True, help="directory for JSONL shards")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-per-seed", type=int, default=20)
    parser.add_argument("--max-total", type=int, default=120)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = load_seeds(args.seeds) if args.seeds and args.seeds.exists() else []
    for sitemap in args.sitemap:
        found = urls_from_sitemap(
            sitemap,
            limit=max(50, args.max_total * 10),
            pattern=CATEGORY_PATTERNS[args.category],
        )
        print(f"sitemap {sitemap}: {len(found)} URL")
        seeds.extend(found)
    seeds = list(dict.fromkeys(seeds))
    if not seeds:
        print("нет seeds: укажите --seeds и/или --sitemap", file=sys.stderr)
        return 2
    stats = collect(
        seeds,
        args.out,
        category=args.category,
        max_per_seed=args.max_per_seed,
        max_total=args.max_total,
        depth=args.depth,
        delay=args.delay,
        timeout=args.timeout,
    )
    report = stats.as_dict()
    report["seeds"] = seeds
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"{args.category}: запрошено {stats.requested}, получено {stats.fetched}, "
        f"записей {stats.records}, хостов {len(stats.hosts)}"
    )
    if stats.skipped:
        print("пропуски:", dict(stats.skipped))
    print("готово:", args.out)
    return 0 if stats.records else 1


if __name__ == "__main__":
    raise SystemExit(main())
