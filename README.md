# web2spec

Дообучение **Qwen2.5-0.5B-Instruct** превращать сжатый HTML в структурированный JSON на ноутбуке с **4 GB VRAM** (RTX 3050 Ti Laptop).

Проект намеренно не учит модель писать скраперы. Модель заполняет целевую схему, а код/селекторы при необходимости рендерит детерминированный слой. Причины и границы — в [`docs/PLAN.md`](docs/PLAN.md), порядок сборки — в [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

## Статус

| Слой | Состояние |
|---|---|
| Контракты данных и конфиги (`schema.py`, `config.py`, `configs/`) | готово, 18 тестов |
| Валидация и сжатие HTML | в работе |
| QLoRA-обучение | в работе |
| Слияние адаптера, GGUF | в работе |
| Оценка качества | в работе |
| Синтетика через teacher-модель | в работе |

## Структура

```
configs/            data.yaml, train_qlora_4gb.yaml, eval.yaml, tasks.yaml
src/web2spec/       библиотека пайплайна
scripts/            01_validate.py … 07_synth.py, ps1-обёртки
tests/              pytest, без тяжёлых зависимостей
data/{raw,interim,processed,reports}/
models/{base,merged,gguf}/
outputs/runs/       адаптеры, чекпоинты, логи
docs/               PLAN.md (стратегия), IMPLEMENTATION_PLAN.md (сборка)
```

## Быстрый старт

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python scripts/01_validate.py --input data/raw/samples.jsonl --output data/processed
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
