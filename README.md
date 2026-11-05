# web2spec

Дообучение **Qwen2.5-0.5B-Instruct** превращать сжатый HTML в структурированный JSON на ноутбуке с **4 GB VRAM** (RTX 3050 Ti Laptop).

Проект намеренно не учит модель писать скраперы. Текущий production-компонент — extraction: модель заполняет целевую схему, а код/селекторы при необходимости рендерит детерминированный слой. Следующий шаг — task-conditioned planner: модель по задаче решает, что собирать. Общий roadmap — [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md), детальный следующий шаг — [`docs/PLANNER_PLAN.md`](docs/PLANNER_PLAN.md). Причины и границы — в [`docs/PLAN.md`](docs/PLAN.md), порядок сборки — в [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

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
| Общий план развития | [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md) |
| План следующего шага (planner) | [`docs/PLANNER_PLAN.md`](docs/PLANNER_PLAN.md) |

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
docs/               PLAN.md, IMPLEMENTATION_PLAN.md, MASTER_PLAN.md, PLANNER_PLAN.md, DATA_SOURCES.md, DATASET.md, STATS.md
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

## Локальный сборщик: задача + список URL

`scripts/21_collect_urls.py` — CLI для внешнего агента. Большая модель (или поиск)
сама находит URL; локальный пайплайн скачивает страницы, сжимает HTML, строит
`CollectionPlan` планировщиком, извлекает `FactSet` plan-conditioned экстрактором,
детерминированно правит evidence и строго валидирует результат. Наружу отдаётся
только JSON с фактами — HTML в контекст большой модели не попадает.

### Что делает и чего не делает

Делает:

- качает по одному URL за раз (без параллельности), уважает `robots.txt` и `Crawl-delay`;
- сжимает HTML (`compress_html`, по умолчанию до 12 000 символов) и выкидывает скрипты/стили;
- planner → extractor одним merged-модельным проходом каждый, на 4 GB GPU (float16);
- `repair_factset`: evidence, которого нет в HTML дословно, либо перепривязывается к
  найденному фрагменту, либо поле уходит в `missing`; факты вне плана отбрасываются;
- `validate_factset`: покрытие полей плана и точное вхождение evidence;
- пишет UTF-8 JSON с `facts`, `missing`, `repairs`, `error` и `aggregate`.

Не делает:

- не исполняет JavaScript и не ждёт SPA-рендеринг;
- не обходит анти-бот, капчи, логины и платные стены;
- не ходит по пагинации и не обходит сайт — обрабатывает ровно переданные URL;
- не гарантирует «полноту» сайта: один URL = одна страница и один результат;
- не тянет большие модели: нужны локальные merged-каталоги (см. ниже).

### Команда

```powershell
.\.venv\Scripts\python.exe scripts\21_collect_urls.py `
  --task "Собери названия, авторов и даты публикации статей" `
  --urls data/urls.txt `
  --out data/reports/collected.json `
  --planner models/merged/qwen05b-plan `
  --extractor models/merged/qwen05b-plan-extract-v2 `
  --max-pages 20 --max-chars 12000 --timeout 20 --delay 1.0
```

`data/urls.txt` — по одному URL в строке; пустые строки и строки с `#` игнорируются.
Отдельный URL можно передать повторяемым `--url`. Дубликаты схлопываются с
сохранением порядка, затем применяется `--max-pages`.

Офлайн-режим (тесты, повторный прогон сохранённых страниц):

```powershell
.\.venv\Scripts\python.exe scripts\21_collect_urls.py `
  --task "Собери названия, авторов и даты публикации статей" `
  --url https://example.com/article/1 `
  --html-dir data/offline_html `
  --out data/reports/collected.json
```

В нём сеть и robots.txt не используются. Файл ищется внутри `--html-dir` по
`sha1(url).html` (UTF-8-хэш полного URL), затем по последнему сегменту пути
(`mug`, `mug.html`, с percent-decoding). Если файла нет — страница получает
`status="fetch_error"` и `error`.

### Выходной JSON

UTF-8, отступ 2, без HTML:

```json
{
  "task": "Собери названия, авторов и даты публикации статей",
  "generated_at": "2026-11-05T12:00:00+00:00",
  "pages": [
    {
      "url": "https://example.com/article/1",
      "status": "ok",
      "entity": "article",
      "facts": [
        {"name": "title", "value": "Как ускорить сбор данных", "evidence": "Как ускорить сбор данных"},
        {"name": "author", "value": "Иван Петров", "evidence": "Иван Петров"}
      ],
      "missing": ["published_at"],
      "repairs": {
        "evidence": [],
        "dropped": [
          {"name": "published_at", "value": "2023-01-05T05:00:00+03:00", "evidence": "2023-01-05"}
        ]
      },
      "dropped_foreign": [],
      "error": null
    }
  ],
  "aggregate": {
    "https://example.com/article/1": {
      "status": "ok",
      "entity": "article",
      "facts": {"title": "Как ускорить сбор данных", "author": "Иван Петров"},
      "missing": ["published_at"],
      "error": null
    }
  },
  "stats": {"requested": 1, "ok": 1, "failed": 0, "by_status": {"ok": 1}}
}
```

- `status`: `ok` | `fetch_error` | `plan_error` | `extract_error`.
- `facts` — только проверенные факты: `name` из плана, `value` скаляр, `evidence` —
  дословная подстрока сжатого HTML.
- `missing` — поля плана, которых на странице нет (или evidence не удалось подтвердить).
- `repairs.evidence` — какие evidence перепривязаны; `repairs.dropped` — какие факты
  потеряли доказательство; `dropped_foreign` — факты вне плана, отброшенные как лишние.
- `error` — текст ошибки стадии (`null` для `ok`).
- `aggregate` — по записи на каждый запрошенный URL, `stats` — счётчики.
- Если `validate_factset` всё же падает, страница получает `extract_error`,
  `facts` и `missing` пустые: непроверенные факты наружу не отдаются.

### Инструкция внешнему агенту

1. Сформируй список URL поиском/браузером и короткую задачу на естественном языке.
2. Запиши URL в файл (по одному в строке) и запусти CLI (команда выше).
3. Читай только `--out` JSON: `aggregate` для сводки, `pages` для evidence и ошибок.
4. Не вставляй HTML и не проси модель «посмотреть страницу» — весь HTML остаётся локально.
5. Политика повторов: `fetch_error` (таймаут, 429/5xx, robots) — повтор через паузу,
   увеличив `--delay`/`--timeout`; `plan_error` — уточни формулировку `--task`;
   `extract_error` — страница не подходит под план или модель не покрыла поля плана,
   повтор обычно не помогает, лучше исключить URL.
6. Коды выхода: `0` — хотя бы одна страница `ok`; `1` — все страницы с ошибкой;
   `2` — неверные аргументы (argparse).
7. Прогресс печатается в stderr, итоговая строка — в stdout; JSON пишется один раз
   в конце прогона.

### Модели

`models/merged/qwen05b-plan` и `models/merged/qwen05b-plan-extract-v2` — локальные
артефакты (каталог `models/` в `.gitignore`), их нужно собрать заранее
(`scripts/03_merge.py` / `02_train.py`). Без них CLI упадёт на загрузке модели;
чистая логика разбора и валидации тестируется без GPU:
`.\.venv\Scripts\python.exe -m pytest tests\test_runtime_cli.py -q`.
