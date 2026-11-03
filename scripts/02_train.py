"""Step 2: QLoRA training or a dry run over the dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web2spec.config import load_data_config, load_train_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QLoRA-обучение web2spec (4 GB VRAM)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="только проверить датасет")
    parser.add_argument("--smoke", action="store_true", help="20 шагов на 32 примерах")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    train_cfg = load_train_config(args.config)
    data_cfg = load_data_config(args.data_config)

    if args.dry_run:
        from web2spec.dataset import ExtractionDataset
        from web2spec.tokens import load_tokenizer

        train_path = data_cfg.paths.resolve().processed / "train.jsonl"
        if not train_path.exists():
            print(f"Нет файла: {train_path}. Сначала запустите scripts/01_validate.py")
            return 1
        tokenizer = load_tokenizer(train_cfg.base_model)
        dataset = ExtractionDataset(
            train_path,
            tokenizer,
            train_cfg.max_seq_len,
        )
        count = min(len(dataset), args.limit or 100)
        lengths = [len(dataset[i]["input_ids"]) for i in range(count)]
        labels = sum(sum(1 for x in dataset[i]["labels"] if x != -100) for i in range(len(lengths)))
        print(f"Примеров: {len(dataset)}")
        print(f"Токенов на пример: min={min(lengths)} max={max(lengths)}")
        print(f"Обучаемых токенов в выборке: {labels}")
        print("Dry-run завершён, модель не загружалась.")
        return 0

    from web2spec.train import train

    adapter = train(train_cfg, data_cfg, smoke=args.smoke)
    print(f"Готово. Адаптер: {adapter}")
    print("Дальше: python scripts/03_merge.py")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
