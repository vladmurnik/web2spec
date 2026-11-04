# web2spec

Дообучение **Qwen2.5-0.5B-Instruct** превращать сжатый HTML в структурированный JSON на ноутбуке с **4 GB VRAM** (RTX 3050 Ti Laptop).

Проект намеренно не учит модель писать скраперы. Модель заполняет целевую схему, а код/селекторы при необходимости рендерит детерминированный слой. Причины и границы — в [`docs/PLAN.md`](docs/PLAN.md), порядок сборки — в [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

## Статус

| Слой | Состояние |
|---|---|
| Контракты данных и конфиги (`schema.py`, `config.py`, `configs/`) | готово |
| Сжатие HTML и токенизация | готово |
| Валидация, дедуп, group-aware splits, CLI | готово |
| QLoRA-обучение (dataset, trainer, smoke-режим) | готово, проверено dry-run |
| Слияние адаптера и конвертация в GGUF | готово |
| Оценка качества (JSON validity, P/R/F1) | готово |
| Синтетика через teacher-модель | готово |
| Сбор реальных страниц по JSON-LD (`08_collect.py`) | готово |
| Синтетический генератор (`09_make_synthetic.py`) | готово |
| Датасет и его ограничения | [`docs/DATASET.md`](docs/DATASET.md) |
| Источники данных и лицензии | [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) |

## Структура

```
configs/            data.yaml, train_qlora_4gb.yaml, eval.yaml, tasks.yaml
src/web2spec/       библиотека пайплайна
scripts/            01_validate.py … 09_make_synthetic.py, ps1-обёртки
tests/              pytest, включая проверки сборщика и синтетики
examples/demo.jsonl 3 примера для проверки пайплайна без скрапинга
data/{raw,interim,processed,reports}/
models/{base,merged,gguf}/
outputs/runs/       адаптеры, чекпоинты, логи
docs/               PLAN.md, IMPLEMENTATION_PLAN.md, DATA_SOURCES.md, DATASET.md
```

## Быстрый старт

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python scripts/01_validate.py --input examples/demo.jsonl --output data/processed
python scripts/02_train.py --dry-run
python scripts/02_train.py --config configs/train_qlora_4gb.yaml --smoke
python scripts/03_merge.py --config configs/train_qlora_4gb.yaml
pwsh scripts/04_convert_gguf.ps1
python scripts/05_eval.py --config configs/eval.yaml
pwsh scripts/06_serve.ps1
```

## Железо

- GPU: RTX 3050 Ti Laptop, 4 GB, compute 8.6 (bf16 поддерживается).
- RAM: 16 GB, свободно ~105 GB диска.
- Ожидаемый пик обучения: 2.0–2.5 GB VRAM, 2–6 часов на 1–2k примеров.

## Правила разработки

- Тяжёлые импорты (`torch`, `transformers`, `peft`, `bitsandbytes`) — **только внутри функций**, чтобы валидация и тесты работали без ML-стека.
- Все пути в YAML — относительно корня репозитория; резолвинг через `web2spec.config.REPO_ROOT`.
- Конфиги строгие (`extra="forbid"`): опечатка в ключе — ошибка, а не молчаливый дефолт.
