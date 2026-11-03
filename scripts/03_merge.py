"""Step 3: merge the LoRA adapter into the fp16/bf16 base model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web2spec.config import REPO_ROOT, load_train_config  # noqa: E402
from web2spec.merge import merge_adapter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Слияние LoRA-адаптера с базовой моделью")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--base", type=str, default=None)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    args = parser.parse_args(argv)

    cfg = load_train_config(args.config)
    base = args.base or cfg.base_model
    adapter = args.adapter or (REPO_ROOT / cfg.output_dir / "adapter")
    out = args.out or REPO_ROOT / "models/merged/qwen2.5-0.5b-ft"

    print(f"База:    {base}")
    print(f"Адаптер: {adapter}")
    print(f"Выход:   {out} ({args.dtype})")
    result = merge_adapter(base, adapter, out, dtype=args.dtype)
    total = sum(path.stat().st_size for path in result.glob("*.safetensors"))
    print(f"Готово: {result} ({total / 1024**3:.2f} GiB весов)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
