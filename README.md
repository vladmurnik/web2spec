# web2spec

Локальный конвейер **«задача → поиск → страницы → проверяемые факты»** для ноутбука с **4 GB VRAM** (RTX 3050 Ti Laptop).

Большая модель по API пишет только цель и ограничения. Дальше всё происходит локально. Для **известных URL** быстрый путь `collect` использует две модели по 0.5B: planner строит план, extractor достаёт факты с точным `evidence`. Для **расплывчатой задачи** путь `research` поднимает анализную модель **Qwen3-4B Q4_K_M**: она уточняет вопросы, генерирует запросы, просматривает 20–50 страниц, ведёт заметки с цитатами, ищет пробелы и только в конце синтезирует ответ. Большая модель получает компактный JSON и не тратит токены на HTML, десятки ссылок и платные поисковые API.

Проект вырос из QLoRA-дообучения Qwen2.5-0.5B для извлечения данных. Обучение и эксперименты сохранены в репозитории, но повседневный runtime теперь не требует PyTorch: все модели запускаются как **Q4_K_M GGUF** через `llama-server`, по одной за раз.

- Roadmap: [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md)
- План planner-трека: [`docs/PLANNER_PLAN.md`](docs/PLANNER_PLAN.md)
- Метрики и ограничения: [`docs/STATS.md`](docs/STATS.md)
- Датасеты: [`docs/DATASET.md`](docs/DATASET.md)
- Источники: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)
- Индекс скриптов: [`scripts/README.md`](scripts/README.md)

## Статус

| Слой | Состояние |
|---|---|
| Контракты данных, конфиги, валидация | готово |
| Сжатие HTML + сохранение метаданных | готово |
| QLoRA-обучение planner/extractor | готово, артефакты в `models/` и `outputs/` |
| Planner + plan-conditioned extractor | готово |
| Runtime-каскад с repair/validation | готово |
| Q4_K_M GGUF как runtime по умолчанию | готово |
| Локальный поиск: задача → ссылки | готово |
| Один вызов `collect`: поиск + факты | готово |
| `research`: 4B-аналитик, 20–50 страниц, evidence | готово: 20/20 страниц, 37 заметок, офлайн-smoke |
| MCP-сервер для любой MCP-совместимой модели | готово |
| Единый CLI `web2spec` | готово |

Проверенные цифры на 50 held-out парах article/Habr: **task success 94%** у PyTorch float16 и **100%** у Q4_K_M, **fact F1 0.9388 / 1.0** соответственно, **0 неверных значений** после repair. Q4_K_M обрабатывает 50 страниц за **101 с** (~2 с/страницу). Это узкий срез: 18 уникальных URL, только статьи; перенос на job/product/realestate не измерен. Подробности и честные ограничения — в [`docs/STATS.md`](docs/STATS.md).

Отдельный офлайн-замер `research` на 20 Habr-страницах с Qwen3-4B Q4_K_M: **20/20 страниц ok, 37 заметок, 20 сущностей, 10 честных пропусков** (в основном отсутствующие авторы), **206 с**, **24.7 tok/s**, загрузка модели **7.0 с**, размер модели **2381.6 MB**. Это срез только по статьям Habr, причём страницы в фикстурах в основном metadata-only: даты относительные («6 часов назад»), авторов нет. Многодоменный бенчмарк качества для `research` пока не проводился; цифры выше не доказывают качество на тарифах, товарах или вакансиях.

## Архитектура

```
большая модель / MCP-клиент
        │  цель + ограничения; URL/домены необязательны
        ▼
┌──────────────────────────── research (20–50 страниц) ────────────────────────────┐
│ Qwen3-4B Q4_K_M: вопросы → запросы → выбор источников → чтение чанков            │
│       → заметки с дословными цитатами → анализ пробелов → новые запросы          │
│       → финальный ответ + детерминированная агрегация фактов/конфликтов          │
└──────────────────────────────────────────────────────────────────────────────────┘
        │
        │  либо быстрый URL-first путь
        ▼
collect: fetch + robots.txt + сжатие HTML + metadata header
        │
        ▼
planner 0.5B Q4_K_M  ──►  CollectionPlan: сущность и поля
        │
        ▼
extractor 0.5B Q4_K_M  ──►  FactSet: name / value / evidence
        │
        ▼
repair + validate  ──►  только проверенные факты, остальное в missing
```

HTML наружу не отдаётся. В `research` модель видит семантические чанки (заголовки, абзацы, строки таблиц), а не весь HTML; каждая заметка обязана содержать дословную цитату и URL. В `collect` каждый факт подтверждается точной подстрокой сжатого HTML. Если доказательства нет, значение уходит в `missing` или `conflicts`, а не выдумывается.

## Структура

```
src/web2spec/       библиотека: схемы, сжатие HTML, plan/fact, llama runtime, CLI
scripts/            numbered pipeline: данные → обучение → eval → runtime → MCP
configs/            YAML для данных, обучения и eval; mcp.example.json
tests/              pytest без GPU и сети, плюс контрактные тесты
docs/               планы, метрики, датасеты, источники
examples/           demo.jsonl для быстрой проверки
data/               raw/interim/processed/reports (gitignored)
models/             base/merged/gguf (gitignored)
outputs/runs/       адаптеры, чекпоинты, логи обучения (gitignored)
```

Ключевые модули:

- `web2spec/html_compress.py` — сжатие страницы и извлечение метаданных;
- `web2spec/semantic.py` — семантические чанки: заголовки, абзацы, строки таблиц;
- `web2spec/research.py` — question-first цикл, заметки, пробелы, агрегация;
- `web2spec/plan.py` — `CollectionPlan`, `FactSet`, `repair_factset`, lenient loader;
- `web2spec/llama_runtime.py` — запуск `llama-server`, Q4-пути по умолчанию, batch-генерация;
- `web2spec/tokens.py` — ChatML-промпты без зависимости от tokenizer;
- `web2spec/cli.py` — единая точка входа `web2spec`.

## Быстрый старт

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

web2spec doctor
```

`doctor` проверяет Q4-модели, `llama-server` и Python. Если модели ещё не собраны, см. раздел «Модели».

### Единый CLI

```powershell
web2spec collect --task "Собери названия и авторов статей" --urls data/urls.txt
web2spec search  --task "Найди статьи про Obsidian и LLM" --provider ddg
web2spec research --task "Сравни тарифы" --domains developers.sber.ru,cloud.vk.com
web2spec mcp
web2spec eval-q4 --limit 50
```

`collect` — быстрый URL-first путь: 0.5B planner/extractor, до 6 страниц по умолчанию. `search` — старый поисковый путь на 0.5B. `research` — новый question-first путь с 4B-аналитиком: сам строит вопросы, ищет, читает 20–50 страниц, ведёт заметки и возвращает ответ с цитатами. `--domain` (можно повторять) ограничивает выдачу нужными сайтами и добавляет `site:`-запросы, чтобы не тратить страницы на блоги и агрегаторы. Для одного вызова из большой модели используйте MCP-инструменты `research` (сложная задача) или `collect` (уже есть URL).

`web2spec <command> --help` показывает параметры конкретного этапа. Numbered-скрипты остаются самостоятельными: их можно запускать напрямую, если нужен полный контроль.

## Модели

По умолчанию runtime использует:

| Роль | Путь | Размер |
|---|---|---:|
| planner | `models/gguf/qwen05b-plan-Q4_K_M.gguf` | 379.4 MB |
| extractor | `models/gguf/qwen05b-plan-extract-v2-Q4_K_M.gguf` | 379.4 MB |
| analyst (`research`) | `models/gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2381.6 MB |

Аналитик — **Qwen3-4B-Instruct-2507 Q4_K_M**. 4B выбран как компромисс: он заметно сильнее 0.5B в понимании задачи и русском языке, но при 8k контекста помещается в 4 GB. 7B Q4_K_M (~4.4–4.8 GB) уже не оставляет запаса под KV-cache и не подходит для этой машины. Модели запускаются **по одной**: 4B и 0.5B одновременно не держите.

Все файлы gitignored: чистое клонирование не содержит весов. 0.5B собираются из merged-каталогов, аналитик скачивается готовым GGUF:

```powershell
curl.exe -L --fail --retry 5 -C - `
  -o models/gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf.part `
  https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
Move-Item models/gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf.part `
  models/gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

`llama-server` берётся из `PATH`, переменной `WEB2SPEC_LLAMA_SERVER` или аргумента `--llama-server`. Для 4 GB VRAM используется Vulkan-устройство `Vulkan0`; переопределяется через `WEB2SPEC_LLAMA_DEVICE` или `--device`.

### Пересборка Q4_K_M

```powershell
# merged-каталоги получаются через обучение + merge
pwsh scripts/04_convert_gguf.ps1 -ModelDir models/merged/qwen05b-plan `
  -Name qwen05b-plan -Quant Q4_K_M
pwsh scripts/04_convert_gguf.ps1 -ModelDir models/merged/qwen05b-plan-extract-v2 `
  -Name qwen05b-plan-extract-v2 -Quant Q4_K_M
```

Важно: hidden size 896 не делится на 256, поэтому квантователь оставляет часть тензоров в q5_0/q6_K/q8_0. Эффективный размер — ~6.35 BPW, а не чистые 4 бита. Это ожидаемое поведение, а не ошибка конвертации.

### PyTorch fallback

Если GGUF нет или нужен старый путь, collector умеет работать с merged-каталогами:

```powershell
web2spec collect --task "..." --urls data/urls.txt `
  --backend hf `
  --planner models/merged/qwen05b-plan `
  --extractor models/merged/qwen05b-plan-extract-v2
```

`--backend auto` (по умолчанию) выбирает GGUF, если путь заканчивается на `.gguf`, иначе PyTorch. Для повседневной работы Q4_K_M быстрее примерно в 5–6 раз и занимает меньше VRAM.

## Локальный сборщик: задача + список URL

`web2spec collect` (или `scripts/21_collect_urls.py`) принимает задачу и URL, скачивает страницы, сжимает HTML, строит `CollectionPlan`, извлекает `FactSet`, детерминированно правит evidence и строго валидирует результат.

Делает:

- качает по одному URL, уважает `robots.txt` и `Crawl-delay`;
- сжимает HTML (по умолчанию до 12 000 символов), удаляет скрипты/стили;
- сохраняет `title`, `description`, `author`, `published_at`, `modified_at`, canonical из `meta`, JSON-LD и `<time datetime>` в header перед контентом;
- запускает planner и extractor по очереди, чтобы 4 GB GPU держал только одну модель;
- `repair_factset`: evidence перепривязывается к найденному фрагменту или поле уходит в `missing`; факты вне плана отбрасываются;
- `validate_factset`: покрытие полей плана и точное вхождение evidence;
- пишет UTF-8 JSON с `facts`, `missing`, `repairs`, `error`, `aggregate`.

Не делает:

- не исполняет JavaScript и не ждёт SPA;
- не обходит анти-бот, капчи, логины и платные стены;
- не ходит по пагинации и не обходит сайт — обрабатывает ровно переданные URL;
- не гарантирует полноту сайта: один URL = одна страница = один результат.

### Команда

```powershell
web2spec collect `
  --task "Собери названия, авторов и даты публикации статей" `
  --urls data/urls.txt `
  --out data/reports/collected.json `
  --max-pages 20 --max-chars 12000 --timeout 20 --delay 1.0
```

`data/urls.txt` — по одному URL в строке; пустые строки и `#` игнорируются. Отдельный URL можно передать повторяемым `--url`. Дубликаты схлопываются с сохранением порядка, затем применяется `--max-pages`.

Офлайн-режим для тестов и повторного прогона сохранённых страниц:

```powershell
web2spec collect `
  --task "Собери названия, авторов и даты публикации статей" `
  --url https://example.com/article/1 `
  --html-dir data/offline_html `
  --out data/reports/collected.json
```

Файл ищется внутри `--html-dir` по `sha1(url).html` (UTF-8-хэш полного URL), затем по последнему сегменту пути (`mug`, `mug.html`, с percent-decoding). Если файла нет — страница получает `status="fetch_error"`.

### Выходной JSON

```json
{
  "task": "Собери названия, авторов и даты публикации статей",
  "generated_at": "2026-11-06T12:00:00+00:00",
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
- `facts` — только проверенные факты: `name` из плана, скалярный `value`, дословный `evidence`.
- `missing` — поля плана, которых на странице нет или evidence не подтвердился.
- `repairs.evidence` — перепривязанные evidence; `repairs.dropped` — факты, потерявшие доказательство; `dropped_foreign` — факты вне плана.
- `error` — текст ошибки стадии (`null` для `ok`).
- Если `validate_factset` падает, страница получает `extract_error`, а `facts` и `missing` остаются пустыми.

Коды выхода: `0` — хотя бы одна страница `ok`; `1` — все страницы с ошибкой; `2` — неверные аргументы.

## Два поисковых пути: search и research

`web2spec search` (или `scripts/22_search_and_collect.py`) — старый быстрый путь: локальная LLM (опционально) формулирует запросы, метапоиск находит ссылки, collector на 0.5B извлекает факты.

```
задача → локальная LLM (опционально) → поисковые запросы
       → локальный метапоиск (SearXNG или DuckDuckGo HTML)
       → дедуп и ранжирование ссылок
       → collector (fetch → compress → planner → extractor)
       → один JSON с фактами
```

```powershell
# 1. Только поиск ссылок, без GPU-моделей:
web2spec search `
  --task "Найди статьи про Obsidian и LLM на Habr" `
  --provider ddg --search-only --max-results 10 `
  --out data/reports/links.json

# 2. Полный конвейер:
web2spec search `
  --task "Найди статьи про Obsidian и LLM на Habr" `
  --provider ddg --max-results 10 --max-pages 5 `
  --out data/reports/search_collected.json
```

### research: вопрос → 20–50 страниц → отчёт с цитатами

`web2spec research` (или `scripts/26_research.py`) — новый путь для расплывчатых задач, где заранее неизвестны ни поля, ни источники. Схема не задаётся заранее: 4B-аналитик сам решает, что искать.

```
цель
  → ResearchBrief: сущности, вопросы, подсказки источников
  → 5–8 запросов → метапоиск → дедуп/ранжирование/фильтр доменов
  → 20–50 страниц → семантические чанки (заголовки, абзацы, строки таблиц)
  → заметки 4B-модели: сущность / поле / значение / дословная цитата / URL
  → анализ пробелов → новые запросы (до 2 раундов)
  → детерминированная агрегация facts / conflicts / missing
  → финальный ответ 4B-модели только по заметкам
```

```powershell
web2spec research `
  --task "Сравни тарифы распознавания речи Yandex SpeechKit, Sber SaluteSpeech и VK Cloud: цена за час, бесплатный лимит, русский язык, диаризация" `
  --domains developers.sber.ru,cloud.vk.com `
  --min-pages 20 --max-pages 30 `
  --out data/reports/asr_research.json
```

- `--min-pages` по умолчанию 20, `--max-pages` — 30; жёсткий диапазон 20–50. Если страниц объективно меньше, отчёт честно пишет `pages_shortfall`, а не добивает счётчик.
- `--domain` / `--domains` добавляют `site:`-запросы и отбрасывают чужие хосты.
- `--source-hint` подсказывает официальные домены/разделы; `--url` и `--urls` позволяют начать с известных страниц.
- `--max-rounds` (1–4) ограничивает цикл поиска пробелов.
- Выход: `answer`, `entities` (`facts` / `conflicts` / `evidence`), `notes`, `sources`, `gaps`, `errors`, `stats`, `speed`. HTML и полные тексты не сохраняются.
- `--html-dir` — офлайн-фикстуры по `sha1(url).html`, без сети.

Аналитик — Qwen3-4B Q4_K_M. Он запускается один на весь research-прогон и выгружается до вызова 0.5B-моделей. Это принципиально для 4 GB VRAM.

`--domain` можно повторять; домены нормализуются, результаты других хостов отбрасываются. `research` не является алиасом `search`: у него другой промпт, другая модель и бюджет страниц.

### Локальная LLM для запросов `search`

Поддерживаются Ollama (по умолчанию) и OpenAI-совместимые серверы (LM Studio, vLLM). Модель задаётся именем, веса и сервер пользователь ставит отдельно. Для `research` это не нужно: запросы строит 4B-аналитик.

```powershell
ollama pull qwen2.5:3b
web2spec search `
  --task "Сравни цены на RTX 5060 Ti" `
  --provider searxng --llm-model qwen2.5:3b `
  --llm-url http://localhost:11434 --search-only `
  --out data/reports/links.json
```

Если LLM недоступна или вернула мусор, скрипт пишет warning и использует текст задачи как запрос. Без `--llm-model` локальная LLM не нужна.

### Поисковые провайдеры

- `--provider searxng --searxng-url http://localhost:8080` — рекомендуется. SearXNG агрегирует публичные движки и отдаёт JSON без ключей. В `settings.yml` нужно включить `formats: [html, json]`.
- `--provider ddg` — HTML-эндпоинт DuckDuckGo без ключа. Работает без установки, но нестабилен и может отдавать 429/captcha.

### Ограничения

- Поиск — это метапоиск/скрапинг публичных страниц, не официальный Google API.
- 4 ГБ VRAM: модели запускаются по одной. `research` держит только Qwen3-4B, `collect` — только 0.5B planner/extractor. Не оставляйте Ollama с 3B/7B работать одновременно с конвейером.
- Коллектор берёт ровно переданные URL: без JS/SPA, антибота, пагинации и обхода сайта. `research` сам ищет 20–50 страниц, но тоже не исполняет JavaScript и не обходит WAF.
- Поисковые запросы идут на публичные эндпоинты провайдера; robots.txt и паузы соблюдаются на этапе сбора.
- Официальные сайты могут отдавать WAF/капчу. Если цена или дата не подтверждена точной подстрокой, поле честно уходит в `missing`, а не заполняется значением из блога. Для WAF-страниц `research` помечает `blocked`, а не `ok`.

## MCP-сервер: любая модель → задача → факты

`web2spec mcp` (или `scripts/23_mcp_server.py`) — MCP-сервер stdio для Claude Desktop, Cursor, Cline, Continue и любого другого MCP-клиента. Большая модель пишет только задачу, а сервер локально делает всё остальное:

```
большая модель (MCP-клиент)
  → research(task="сложный вопрос")  или  collect(task="...", urls=[...])
  → research: 4B-аналитик строит вопросы, ищет, читает 20–50 страниц,
    ведёт заметки с цитатами и синтезирует ответ
  → collect: 0.5B planner/extractor обрабатывает известные URL
  → большая модель получает компактный JSON без HTML
```

HTML и полные тексты наружу не отдаются.

Сервер отдаёт **два инструмента**: `research` для расплывчатой задачи и `collect` для уже известных URL. Большая модель не должна сама комбинировать поиск, выбор ссылок и извлечение.

| Инструмент | Что делает |
|---|---|
| `research` | `task` → вопросы → поиск → 20–50 страниц → заметки с цитатами → `answer` + `entities` + `gaps`. Один вызов для сложного исследования. |
| `collect` | `task` → (поиск) → скачивание → факты. `urls` пропускает поиск, `domains` ограничивает его нужными хостами. Быстрый путь, по умолчанию 6 страниц. |

Параметры `research`:

- `task` (обязательный) — что исследовать;
- `urls` — стартовые URL; поиск всё равно может добавить страницы;
- `domains` — домены для `site:`-запросов и фильтрации;
- `source_hints` — подсказки официальных источников;
- `min_pages` / `max_pages` — по умолчанию 20 / 30, жёсткий диапазон 20–50;
- `max_rounds` — 1–4 раунда поиска пробелов;
- `include_notes` — вернуть сырые заметки (по умолчанию `false`).

`research` возвращает `answer`, `entities` (`facts` / `conflicts` / `evidence`), `sources`, `gaps`, `errors`, `stats`, `speed`. Поле `notes` добавляется только по флагу.

Параметры `collect`:

- `task` (обязательный) — что найти;
- `urls` — необязательный список URL; поиск тогда не выполняется;
- `domains` — необязательный список доменов для `site:`-запросов и фильтрации;
- `max_results` — сколько ссылок взять из поиска (по умолчанию 6);
- `max_pages` — сколько страниц обработать (по умолчанию 6);
- `include_evidence` — добавить точные цитаты (по умолчанию `false`);
- `include_pages` — вернуть также компактные строки по каждой странице
  (по умолчанию `false`).

Ответ `collect` уже агрегирован:

```json
{
  "task": "Сравни тарифы",
  "facts": {"russian": true},
  "conflicts": {
    "price_per_hour_rub": [
      {"value": 36, "sources": ["https://developers.sber.ru/..."]},
      {"value": 30, "sources": ["https://cloud.vk.com/..."]}
    ]
  },
  "missing": {"diarization": ["https://cloud.vk.com/..."]},
  "by_source": {
    "https://developers.sber.ru/...": {"price_per_hour_rub": 36, "russian": true},
    "https://cloud.vk.com/...": {"price_per_hour_rub": 30, "russian": true}
  },
  "sources": ["https://developers.sber.ru/...", "https://cloud.vk.com/..."],
  "errors": [],
  "stats": {"requested": 2, "ok": 2, "failed": 0},
  "search": {"queries": ["..."], "count": 2, "warnings": []}
}
```

- `facts` — поля, где все страницы дали одно значение;
- `conflicts` — разные значения; сервер не выбирает за вас;
- `by_source` — компактная привязка фактов к URL, когда есть конфликты
  (удобно для сравнения нескольких компаний);
- `missing` — каких полей нет и на каких страницах;
- `errors` — только упавшие страницы, без HTML и полных текстов.

### Подключение

1. Собери окружение и Q4-модели (см. «Модели»).
2. Скопируй `configs/mcp.example.json` в конфиг клиента и замени пути.
3. Перезапусти клиент — появятся инструменты `research` и `collect`.

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

Cursor: тот же блок в `~/.cursor/mcp.json`. Для любого другого клиента достаточно запустить `python scripts\23_mcp_server.py` как stdio-процесс.

### Настройки

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `WEB2SPEC_PROVIDER` | `searxng` | `searxng` или `ddg` |
| `WEB2SPEC_SEARXNG_URL` | `http://localhost:8080` | адрес SearXNG с JSON |
| `WEB2SPEC_LLM_MODEL` | пусто | модель для запросов; без неё используется текст задачи |
| `WEB2SPEC_LLM_URL` | `http://localhost:11434` | Ollama или OpenAI-совместимый сервер |
| `WEB2SPEC_LLM_PROVIDER` | `ollama` | `ollama` или `openai` |
| `WEB2SPEC_LANGUAGE` | `ru` | язык поисковой выдачи |
| `WEB2SPEC_PLANNER` | `models/gguf/qwen05b-plan-Q4_K_M.gguf` | planner |
| `WEB2SPEC_EXTRACTOR` | `models/gguf/qwen05b-plan-extract-v2-Q4_K_M.gguf` | extractor |
| `WEB2SPEC_ANALYST` | `models/gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 4B-аналитик для `research` |
| `WEB2SPEC_RESEARCH_MIN_PAGES` | `20` | минимум страниц research |
| `WEB2SPEC_RESEARCH_MAX_PAGES` | `30` | максимум страниц research |
| `WEB2SPEC_RESEARCH_TIMEOUT` | `3600` | таймаут одного research-вызова, секунды |
| `WEB2SPEC_BACKEND` | `auto` | `auto`, `gguf` или `hf` |
| `WEB2SPEC_LLAMA_SERVER` | пусто | путь к `llama-server` |
| `WEB2SPEC_LLAMA_DEVICE` | `Vulkan0` | устройство llama.cpp; пусто — авто |
| `WEB2SPEC_CONTEXT` | `8192` | контекст llama-server |
| `WEB2SPEC_STARTUP_TIMEOUT` | `240` | ожидание старта llama-server, секунды |
| `WEB2SPEC_REQUEST_TIMEOUT` | `300` | таймаут одного запроса к llama-server |
| `WEB2SPEC_MAX_CHARS` | `12000` | сколько символов страницы видит модель |
| `WEB2SPEC_TIMEOUT` | `20` | таймаут загрузки страницы, секунды |
| `WEB2SPEC_DELAY` | `1.0` | пауза между страницами, секунды |
| `WEB2SPEC_HTML_DIR` | пусто | офлайн-HTML вместо сети |

Те же значения можно передать аргументами сервера (`--provider`, `--planner`, `--backend`, `--device` и т.д.) — они переопределяют окружение.

### Проверка и скорость

```powershell
# Протокол и инструменты, без GPU и сети:
.\.venv\Scripts\python.exe -m pytest tests\test_mcp_server.py -q
```

Сервер пишет диагностику только в stderr, stdout остаётся чистым JSON-RPC. `research` держит один 4B-сервер на весь прогон; `collect` — один 0.5B-сервер. При 4 ГБ VRAM не запускайте оба пути одновременно и не держите Ollama с 3B/7B рядом с конвейером.

Замер Q4_K_M на RTX 3050 Ti 4 GB: planner ~172 ток/с, extractor ~161 ток/с, **~2 с на страницу**, 50 страниц за 101 с. Для сравнения PyTorch float16 даёт ~24 ток/с и ~11.7 с на страницу. Поиск 5 ссылок через DDG — около 2 с. Подробности — в [`docs/STATS.md`](docs/STATS.md).

## Правила разработки

- Тяжёлые импорты (`torch`, `transformers`, `peft`, `bitsandbytes`) — только внутри функций, чтобы валидация и тесты работали без ML-стека.
- Все пути в YAML — относительно корня репозитория; резолвинг через `web2spec.config.REPO_ROOT`.
- Конфиги строгие (`extra="forbid"`): опечатка в ключе — ошибка.
- Факты никогда не выдумываются: нет точного evidence — поле уходит в `missing`.
- Production-артефакты и `outputs/runs` не перезаписываются без явного разрешения.
- Тесты: `.\.venv\Scripts\python.exe -m pytest -q`; известное Windows-исключение — `tests/test_synth.py::test_cli_dry_run` (cp1251), не связано с runtime.
