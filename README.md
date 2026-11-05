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

## Локальный поиск: задача → ссылки → данные

`scripts/22_search_and_collect.py` добавляет к конвейеру этап поиска, чтобы
внешней большой модели не нужно было ни искать в платном API, ни читать HTML.

```
задача → локальная LLM (опционально) → поисковые запросы
       → локальный метапоиск (SearXNG или DuckDuckGo HTML)
       → дедуп и ранжирование ссылок
       → scripts/21_collect_urls.py (fetch → compress → planner → extractor)
       → один JSON с фактами, evidence и ошибками
```

Важно: LLM не «гуглит» сама. Она формулирует запросы, а ссылки находит
метапоисковик. Платные поисковые API не используются.

### Быстрый запуск

```powershell
# 1. Только поиск ссылок (без GPU-моделей), DDG без ключа:
.\.venv\Scripts\python.exe scripts\22_search_and_collect.py `
  --task "Найди статьи про Obsidian и LLM на Habr" `
  --provider ddg --search-only --max-results 10 `
  --out data\reports\links.json

# 2. Полный конвейер: поиск + сбор фактов:
.\.venv\Scripts\python.exe scripts\22_search_and_collect.py `
  --task "Найди статьи про Obsidian и LLM на Habr" `
  --provider ddg --max-results 10 --max-pages 5 `
  --out data\reports\search_collected.json
```

### Локальная LLM для запросов

Поддерживаются Ollama (по умолчанию) и OpenAI-совместимые серверы (LM Studio,
vLLM). Модель задаётся именем, веса и сервер пользователь ставит отдельно:

```powershell
# Ollama, модель 3B в Q4 (~2 ГБ VRAM)
ollama pull qwen2.5:3b
.\.venv\Scripts\python.exe scripts\22_search_and_collect.py `
  --task "Сравни цены на RTX 5060 Ti" `
  --provider searxng --llm-model qwen2.5:3b `
  --llm-url http://localhost:11434 --search-only `
  --out data\reports\links.json

# OpenAI-совместимый сервер
.\.venv\Scripts\python.exe scripts\22_search_and_collect.py `
  --task "..." --llm-provider openai --llm-url http://localhost:1234 `
  --llm-model local-model --search-only --out data\reports\links.json
```

Если LLM недоступна или вернула мусор, скрипт честно пишет warning и использует
саму задачу как запрос. Без `--llm-model` локальная LLM не нужна.

### Поисковые провайдеры

- `--provider searxng --searxng-url http://localhost:8080` — рекомендуется.
  SearXNG агрегирует публичные движки и отдаёт JSON без ключей и оплаты.
  В `settings.yml` экземпляра должен быть включён `formats: [html, json]`.
  Проще всего поднять локально через Docker; публичные инстансы часто
  ограничивают JSON.
- `--provider ddg` — HTML-эндпоинт DuckDuckGo без ключа. Работает без
  установки, но нестабилен и может отдавать 429/ captcha при частых запросах.

### Что в JSON

К отчёту коллектора добавляется блок `search`:

```json
{
  "task": "...",
  "pages": [{"url": "...", "status": "ok", "facts": [...], "missing": [...]}],
  "aggregate": {"https://...": {"facts": {"title": "..."}}},
  "stats": {"requested": 5, "ok": 4, "failed": 1, "by_status": {"ok": 4, "fetch_error": 1}},
  "search": {
    "provider": "ddg",
    "query_source": "ollama:qwen2.5:3b",
    "queries": ["..."],
    "candidates": [{"url": "...", "title": "...", "snippet": "...", "score": 0.42}],
    "count": 10,
    "warnings": [],
    "collector_error": null
  }
}
```

`--search-only` возвращает только `task`, `generated_at`, пустые `pages`/`stats`
и блок `search` — удобно, чтобы большая модель выбрала ссылки сама.

### Ограничения

- Поиск — это метапоиск/скрапинг публичных страниц, не официальный Google API.
  Возможны блокировки, лимиты и разное качество выдачи.
- 4 ГБ VRAM: не держите Ollama с 3B/7B и конвейер 0.5B одновременно. Надёжный
  путь — сначала `--search-only`, затем выгрузить LLM (`ollama stop <model>`)
  и запустить `21_collect_urls.py` по сохранённому списку ссылок.
- Коллектор по-прежнему берёт ровно переданные URL: без JS/SPA, антибота,
  пагинации и обхода сайта. Одна ссылка = одна страница.
- robots.txt, паузы и таймауты соблюдаются на этапе сбора; поисковые запросы
  идут на публичные эндпоинты провайдера.

## MCP-сервер: любая модель → задача → факты

`scripts/23_mcp_server.py` — это готовый MCP-сервер (stdio), который подключается
к Claude Desktop, Cursor, Cline, Continue или любому другому MCP-клиенту. Большая
модель по API пишет только задачу, а сервер локально делает всё остальное:

```
большая модель (MCP-клиент)
  → tool collect(task="что найти")
  → локальная LLM формулирует поисковые запросы (опционально)
  → локальный метапоиск ищет ссылки (SearXNG или DDG, без платных API)
  → страницы скачиваются и сжимаются
  → маленькие модели planner + extractor достают факты
  → большая модель получает компактный JSON: факты, missing, ошибки
```

HTML, сырые страницы и полные тексты наружу не отдаются. Модель-клиент платит
токены только за нужные поля.

### Инструменты

| Инструмент | Что делает |
| --- | --- |
| `collect` | Главный инструмент: `task` → поиск → скачивание → факты. Если передан `urls`, поиск пропускается и обрабатываются только эти ссылки. |
| `search_links` | Только поиск ссылок, без скачивания и моделей. Удобно, если большая модель хочет сама выбрать URL. |

Параметры `collect`:

- `task` (обязательный) — что найти, обычным текстом;
- `urls` — необязательный список URL;
- `max_results` — сколько ссылок взять из поиска (по умолчанию 10);
- `max_pages` — сколько страниц обработать (по умолчанию 8);
- `include_evidence` — добавить точные цитаты `evidence` (по умолчанию `false`,
  чтобы не тратить токены).

Пример ответа `collect` (по одной странице):

```json
{
  "task": "Track new articles, their authors, and publication dates.",
  "pages": [
    {
      "url": "https://habr.com/ru/articles/...",
      "status": "ok",
      "entity": "article",
      "facts": {"title": "Как я оформляю тренировки и питание в Obsidian"},
      "missing": ["published_at"],
      "error": null
    }
  ],
  "stats": {"requested": 1, "ok": 1, "failed": 0, "by_status": {"ok": 1}},
  "search": {"provider": "searxng", "queries": ["..."], "warnings": []}
}
```

Если поле не найдено, оно попадает в `missing`, а не выдумывается. `status`
может быть `ok`, `fetch_error`, `plan_error` или `extract_error`.

### Подключение

1. Собери окружение и модели (см. разделы выше).
2. Скопируй `configs/mcp.example.json` в конфиг клиента и замени пути.
3. Перезапусти клиент — появятся инструменты `collect` и `search_links`.

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "web2spec": {
      "command": "C:\\path\\to\\repo\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\repo\\scripts\\23_mcp_server.py"],
      "env": {
        "WEB2SPEC_PROVIDER": "ddg",
        "WEB2SPEC_LLM_MODEL": "qwen2.5:3b",
        "WEB2SPEC_LLM_URL": "http://localhost:11434"
      }
    }
  }
}
```

Cursor: тот же блок в `~/.cursor/mcp.json` (или в настройках MCP). Для любого
другого клиента достаточно запустить `python scripts\23_mcp_server.py` как
stdio-процесс.

### Настройки через переменные окружения

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `WEB2SPEC_PROVIDER` | `searxng` | `searxng` или `ddg` |
| `WEB2SPEC_SEARXNG_URL` | `http://localhost:8080` | адрес SearXNG с JSON |
| `WEB2SPEC_LLM_MODEL` | пусто | модель для запросов; без неё используется текст задачи |
| `WEB2SPEC_LLM_URL` | `http://localhost:11434` | Ollama или OpenAI-совместимый сервер |
| `WEB2SPEC_LLM_PROVIDER` | `ollama` | `ollama` или `openai` |
| `WEB2SPEC_LANGUAGE` | `ru` | язык поисковой выдачи |
| `WEB2SPEC_PLANNER` | `models/merged/qwen05b-plan` | путь к planner |
| `WEB2SPEC_EXTRACTOR` | `models/merged/qwen05b-plan-extract-v2` | путь к extractor |
| `WEB2SPEC_MAX_CHARS` | `12000` | сколько символов страницы видит модель |
| `WEB2SPEC_TIMEOUT` | `20` | таймаут загрузки страницы, секунды |
| `WEB2SPEC_DELAY` | `1.0` | пауза между страницами, секунды |
| `WEB2SPEC_HTML_DIR` | пусто | если задан, страницы читаются из локальных HTML вместо сети (офлайн-тесты) |

Те же значения можно передать аргументами сервера (`--provider`, `--llm-model`,
`--planner` и т.д.) — они переопределяют окружение.

### Проверка

```powershell
# 13 тестов протокола и инструментов, без GPU и сети:
.\.venv\Scripts\python.exe -m pytest tests\test_mcp_server.py -q

# Живой smoke только поиска (без моделей):
.\.venv\Scripts\python.exe scripts\23_mcp_server.py
# затем в клиенте вызвать search_links(task="...")
```

Сервер пишет диагностику только в stderr, stdout остаётся чистым JSON-RPC.
Один вызов `collect` загружает 0.5B модели и занимает заметное время; при 4 ГБ
VRAM не держите одновременно Ollama с 3B/7B и конвейер. Схема: `search_links` →
`ollama stop <model>` → `collect` с готовыми `urls`.
