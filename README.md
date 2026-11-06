# web2spec

Локальный конвейер **«задача → поиск → страницы → структурированные факты»** для ноутбука с **4 GB VRAM** (RTX 3050 Ti Laptop).

Большая модель по API пишет только то, что нужно найти. Дальше всё происходит локально: маленькая модель строит план сбора, локальный метапоиск находит ссылки, страницы скачиваются и сжимаются, второй маленький модель-экстрактор достаёт факты с точным `evidence`. Большая модель получает компактный JSON и не тратит токены на HTML и платные поисковые API. В MCP-режиме для этого достаточно **одного вызова `collect`**: поиск, загрузка страниц и извлечение выполняются внутри сервера.

Проект вырос из QLoRA-дообучения Qwen2.5-0.5B для извлечения данных. Обучение и эксперименты сохранены в репозитории, но повседневный runtime теперь не требует PyTorch: planner и extractor запускаются как **Q4_K_M GGUF** через `llama-server`.

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
| MCP-сервер для любой MCP-совместимой модели | готово |
| Единый CLI `web2spec` | готово |

Проверенные цифры на 50 held-out парах article/Habr: **task success 94%** у PyTorch float16 и **100%** у Q4_K_M, **fact F1 0.9388 / 1.0** соответственно, **0 неверных значений** после repair. Q4_K_M обрабатывает 50 страниц за **101 с** (~2 с/страницу). Это узкий срез: 18 уникальных URL, только статьи; перенос на job/product/realestate не измерен. Подробности и честные ограничения — в [`docs/STATS.md`](docs/STATS.md).

## Архитектура

```
большая модель / MCP-клиент
        │  задача + необязательные URL
        ▼
локальный поиск (SearXNG / DuckDuckGo HTML)
        │  ссылки, дедуп, ранжирование, фильтр domains
        ▼
fetch + robots.txt + сжатие HTML + metadata header
        │
        ▼
planner Q4_K_M  ──►  CollectionPlan: сущность и поля
        │
        ▼
extractor Q4_K_M  ──►  FactSet: name / value / evidence
        │
        ▼
repair + validate  ──►  только проверенные факты, остальное в missing
        │
        ▼
агрегация по полям  ──►  facts / conflicts / missing / by_source
```

HTML наружу не отдаётся. Каждый факт подтверждается дословной подстрокой сжатого HTML; если доказательства нет, поле уходит в `missing`, а не выдумывается.

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

`search` и `research` — одно и то же: задача → локальный поиск → скачивание → факты. `--domain` (в CLI можно повторять) ограничивает выдачу нужными сайтами и добавляет `site:`-запросы, чтобы не тратить страницы на блоги и агрегаторы. Для одного вызова из большой модели используйте MCP-инструмент `collect`.

`web2spec <command> --help` показывает параметры конкретного этапа. Numbered-скрипты остаются самостоятельными: их можно запускать напрямую, если нужен полный контроль.

## Модели

По умолчанию runtime использует:

| Роль | Путь | Размер |
|---|---|---:|
| planner | `models/gguf/qwen05b-plan-Q4_K_M.gguf` | 379.4 MB |
| extractor | `models/gguf/qwen05b-plan-extract-v2-Q4_K_M.gguf` | 379.4 MB |

Оба файла gitignored: чистое клонирование не содержит весов. Их нужно собрать из merged-каталогов или скопировать на машину.

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

## Локальный поиск: задача → ссылки → данные

`web2spec search` (или `scripts/22_search_and_collect.py`) добавляет этап поиска, чтобы внешней большой модели не нужно было искать в платном API или читать HTML.

```
задача → локальная LLM (опционально) → поисковые запросы
       → локальный метапоиск (SearXNG или DuckDuckGo HTML)
       → дедуп и ранжирование ссылок
       → collector (fetch → compress → planner → extractor)
       → один JSON с фактами
```

LLM не «гуглит» сама: она формулирует запросы, а ссылки находит метапоисковик. Платные поисковые API не используются.

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

# 3. Только официальные сайты: site:-запросы + фильтр результатов
web2spec research `
  --task "Сравни тарифы распознавания речи" `
  --domain developers.sber.ru --domain cloud.vk.com `
  --max-results 6 --max-pages 6 `
  --out data/reports/asr_tariffs.json
```

`research` — алиас `search`. В одном запуске он делает поиск, скачивание,
извлечение и агрегацию; большая модель не обязана вызывать поиск и сбор по
отдельности. `--domain` можно повторять; домены нормализуются, результаты
других хостов отбрасываются, а в запросы добавляются `site:<domain>`.

### Локальная LLM для запросов

Поддерживаются Ollama (по умолчанию) и OpenAI-совместимые серверы (LM Studio, vLLM). Модель задаётся именем, веса и сервер пользователь ставит отдельно:

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
- 4 ГБ VRAM: не держите Ollama с 3B/7B и конвейер 0.5B одновременно. Если нужна локальная LLM для запросов, сначала выполните `--search-only`, затем `ollama stop <model>`; либо не задавайте `--llm-model` и используйте детерминированные запросы.
- Коллектор берёт ровно переданные URL: без JS/SPA, антибота, пагинации и обхода сайта.
- Поисковые запросы идут на публичные эндпоинты провайдера; robots.txt и паузы соблюдаются на этапе сбора.
- Официальные сайты могут отдавать WAF/капчу. Если цена или дата не подтверждена точной подстрокой, поле честно уходит в `missing`, а не заполняется значением из блога.

## MCP-сервер: любая модель → задача → факты

`web2spec mcp` (или `scripts/23_mcp_server.py`) — MCP-сервер stdio для Claude Desktop, Cursor, Cline, Continue и любого другого MCP-клиента. Большая модель пишет только задачу, а сервер локально делает всё остальное:

```
большая модель (MCP-клиент)
  → tool collect(task="что найти")
  → локальная LLM формулирует запросы (опционально)
  → локальный метапоиск ищет ссылки
  → страницы скачиваются и сжимаются
  → planner + extractor достают факты
  → большая модель получает компактный JSON
```

HTML и полные тексты наружу не отдаются.

Сервер намеренно отдаёт **один инструмент** `collect`: большая модель не должна
сама комбинировать поиск, выбор ссылок и извлечение. Если ссылки уже известны,
они передаются в `urls`; если известны только сайты — в `domains`. Во всех
случаях это один вызов.

| Инструмент | Что делает |
|---|---|
| `collect` | `task` → (поиск) → скачивание → факты. `urls` пропускает поиск, `domains` ограничивает его нужными хостами. |

Параметры `collect`:

- `task` (обязательный) — что найти;
- `urls` — необязательный список URL; поиск тогда не выполняется;
- `domains` — необязательный список доменов для `site:`-запросов и фильтрации;
- `max_results` — сколько ссылок взять из поиска (по умолчанию 6);
- `max_pages` — сколько страниц обработать (по умолчанию 6);
- `include_evidence` — добавить точные цитаты (по умолчанию `false`);
- `include_pages` — вернуть также компактные строки по каждой странице
  (по умолчанию `false`).

Ответ уже агрегирован:

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
3. Перезапусти клиент — появится инструмент `collect`.

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

Сервер пишет диагностику только в stderr, stdout остаётся чистым JSON-RPC. Один вызов `collect` поднимает `llama-server` и обрабатывает страницы последовательно; при 4 ГБ VRAM не держите одновременно Ollama с 3B/7B и конвейер. Если нужна локальная LLM для запросов, сделайте поиск заранее или задайте модель только на время поиска.

Замер Q4_K_M на RTX 3050 Ti 4 GB: planner ~172 ток/с, extractor ~161 ток/с, **~2 с на страницу**, 50 страниц за 101 с. Для сравнения PyTorch float16 даёт ~24 ток/с и ~11.7 с на страницу. Поиск 5 ссылок через DDG — около 2 с. Подробности — в [`docs/STATS.md`](docs/STATS.md).

## Правила разработки

- Тяжёлые импорты (`torch`, `transformers`, `peft`, `bitsandbytes`) — только внутри функций, чтобы валидация и тесты работали без ML-стека.
- Все пути в YAML — относительно корня репозитория; резолвинг через `web2spec.config.REPO_ROOT`.
- Конфиги строгие (`extra="forbid"`): опечатка в ключе — ошибка.
- Факты никогда не выдумываются: нет точного evidence — поле уходит в `missing`.
- Production-артефакты и `outputs/runs` не перезаписываются без явного разрешения.
- Тесты: `.\.venv\Scripts\python.exe -m pytest -q`; известное Windows-исключение — `tests/test_synth.py::test_cli_dry_run` (cp1251), не связано с runtime.
