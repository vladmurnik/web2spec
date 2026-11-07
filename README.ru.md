# web2spec

[English](README.md) · **Русский**

[![CI](https://github.com/vladmurnik/web2spec/actions/workflows/ci.yml/badge.svg)](https://github.com/vladmurnik/web2spec/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Локальный конвейер **«задача → поиск → страницы → проверяемые факты»** для
моделей, работающих по API.

Большая модель пишет только **что нужно найти**. Дальше всё происходит на
вашей машине: локальный аналитик формулирует запросы, находит и читает
страницы, а возвращает компактные факты с дословными цитатами. Вызывающая
модель не получает HTML, не читает десятки ссылок и не платит за поисковый
API.

```text
задача -> локальный поиск -> страницы -> локальные модели -> факты с evidence
```

Эталонная машина — ноутбук с **4 GB VRAM** (RTX 3050 Ti). Runtime использует
GGUF-модели через `llama-server` и держит в памяти ровно одну модель.

## Зачем это нужно

- **Экономия токенов.** Если большая модель читает сырые страницы, одна
  исследовательская задача может стоить сотни тысяч токенов. web2spec
  возвращает небольшой JSON вместо HTML и полного текста.
- **Локальный поиск.** SearXNG или HTML-выдача DuckDuckGo: без платных ключей
  и без учёта запросов.
- **Либо evidence, либо missing.** Каждый факт должен быть точной подстрокой
  сжатой страницы. Непроверенное значение уходит в `missing` или `conflicts`;
  конвейер не заполняет пробелы по памяти модели.
- **Маленькая GPU.** Быстрый путь `collect` использует две модели по 0.5B
  (~380 MB каждая). Путь `research` поднимает 4B-аналитик (~2.4 GB) и
  выгружает его до запуска collector.

## Измеренные результаты

Это прокси-замеры, а не общий бенчмарк. Collector проверен на **50 held-out
парах задача/страница из 18 статей Habr**. Отдельный датированный срез реальных
страниц (документация, API, цены, характеристики товара) дал **1/7 точных
кейсов и 8/19 полей** на Q4_K_M. Низкий результат оставлен видимым: он показывает
реальные ошибки выбора строки таблицы и не является общим заявлением о качестве.
Вакансии, недвижимость, новости и тарифы по-прежнему не покрыты.

| Путь | Task success | Fact F1 | Evidence | Неверных значений после repair | Скорость |
| --- | ---: | ---: | --- | ---: | ---: |
| Collect, Q4_K_M | **100%** (50/50) | **1.0000** | точная подстрока | **0** | **101 с / 50 страниц** (~2 с/стр.) |
| Collect, PyTorch fp16 | 94% (47/50) | 0.9388 | точная подстрока | 0 | ~11.7 с/стр. |
| Planner (test) | — | field F1 **0.8220** | — | — | — |
| Extractor v2 (test) | — | fact F1 **0.9357** | evidence rate 0.9524 | — | — |

Высокий результат Q4 объясняется узким срезом: 18 уникальных страниц, один
хост, один тип сущности. Это регрессионная проверка, а не заявление, что
модель решает произвольные сайты.

`research` проверен офлайн на 20 фикстурах Habr с Qwen3-4B Q4_K_M: 20/20
страниц обработано, 37 заметок с цитатами, 20 сущностей, 35 пробелов,
10 честных пропусков заметок (в основном отсутствующие авторы), 206 с,
24.7 ток/с, загрузка модели 7.0 с. Фикстуры в основном metadata-only, поэтому
это проверка цикла и контракта вывода, а не качества ответов на разных
доменах.

Полная методика, железо и ограничения — в
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) и лабораторных заметках
[`docs/STATS.md`](docs/STATS.md).

## Два пути, один контракт

### `research` — расплывчатая задача, неизвестные источники

Для вопросов вида «сравни актуальные тарифы трёх провайдеров», где заранее
неизвестны ни поля, ни URL.

```text
цель
  -> ResearchBrief: сущности, вопросы, подсказки источников
  -> 5-8 запросов -> локальный метапоиск -> дедуп / ранжирование / фильтр доменов
  -> 20-50 страниц -> семантические чанки (заголовки, абзацы, строки таблиц)
  -> заметки 4B: сущность / поле / значение / дословная цитата / URL
  -> анализ пробелов -> новые запросы (до 2 раундов)
  -> детерминированная агрегация: facts / conflicts / missing
  -> финальный ответ 4B только по заметкам
```

Аналитик — **Qwen3-4B-Instruct-2507 Q4_K_M**. Он работает весь research-вызов
и завершается до загрузки любой 0.5B-модели.

### `collect` — известные URL, быстрый путь

Для задачи «достань заголовок, автора и дату публикации с этих пяти
официальных страниц».

```text
URL -> вежливый fetch + robots.txt -> сжатый HTML + metadata header
    -> planner 0.5B -> CollectionPlan (сущность + поля)
    -> extractor 0.5B -> FactSet (name / value / evidence)
    -> детерминированный repair -> строгая валидация -> facts / missing
```

`collect` не обходит сайт. Он обрабатывает ровно переданные URL.

Если схема уже известна, передайте `--field name[:kind[:description]]`
(можно повторять): planner не запускается, имена полей сохраняются точно.

```powershell
web2spec collect --task "Собери цену и наличие" --url https://shop.example/item `
  --field "price:number:цена в рублях без скидки" --field in_stock:bool
```

Типы: `text`, `number`, `date`, `url`, `bool`, `null`. Описание видит extractor;
оно помогает выбрать нужную строку таблицы. MCP-инструмент `collect` принимает
тот же список в `fields`.

### Инвариант evidence

Факт принимается только тогда, когда его `evidence` — точная подстрока
сжатой страницы. `repair_factset` пытается перепривязать evidence к nearby
точному фрагменту; если не получается, поле становится `missing`. Факты вне
плана отбрасываются. Поэтому результату можно доверять больше, чем памяти
модели.

## Быстрый старт

### Требования

- Python **3.12**;
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) с `llama-server` в
  `PATH` (или `WEB2SPEC_LLAMA_SERVER`);
- ~3.3 GB на диске для трёх GGUF-моделей;
- GPU с ~4 GB VRAM — проверенная цель; CPU работает, но намного медленнее;
- эталонная платформа — Windows; Linux и macOS используют тот же Python-код.

### Установка

Одна команда в Windows:

```powershell
git clone https://github.com/vladmurnik/web2spec.git
cd web2spec
powershell -ExecutionPolicy Bypass -File scripts/00_install.ps1
```

Добавьте `-InstallLlama`, чтобы поставить `llama.cpp` через `winget`, `-Dev`
для тестовых зависимостей или `-SkipModels`, чтобы пропустить скачивание.

Одна команда в Linux/macOS:

```bash
git clone https://github.com/vladmurnik/web2spec.git
cd web2spec
bash scripts/00_install.sh
```

Добавьте `--dev` для тестовых зависимостей или `--skip-models`, чтобы
пропустить скачивание. Скрипты создают `.venv`, ставят пакет, проверяют
SHA-256 моделей и запускают `web2spec doctor`.

Ручной вариант:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
web2spec download-models --only all
web2spec doctor
```

Ставьте именно `-e` (editable): CLI ищет `scripts/` и `models/` относительно
checkout, поэтому проект рассчитан на клонирование репозитория, а не на
библиотечный wheel.

`download-models` проверяет SHA-256 по
[`models/checksums.sha256`](models/checksums.sha256). Две 0.5B-модели
прикреплены к GitHub release; 4B-аналитик скачивается из официального
Qwen GGUF-репозитория. Веса не хранятся в git.

Ожидаемый вывод `doctor`:

```text
[OK  ] analyst Q4_K_M: .../models/gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf (2381.6 MB)
[OK  ] planner Q4_K_M: .../models/gguf/qwen05b-plan-Q4_K_M.gguf (379.4 MB)
[OK  ] extractor Q4_K_M: .../models/gguf/qwen05b-plan-extract-v2-Q4_K_M.gguf (379.4 MB)
[OK  ] llama-server: .../llama-server
[OK  ] Python: 3.12.x
```

### Кэш повторных запусков

`collect`, `search` и `research` держат скачанные страницы и результаты поиска
в `.cache/web2spec.sqlite` 24 часа. Это удешевляет повторные запуски, но не
превращает устаревшие страницы в вечные факты:

```powershell
web2spec collect --task "..." --url https://example.com/page
web2spec collect --task "..." --url https://example.com/page --cache-ttl 1
web2spec collect --task "..." --url https://example.com/page --no-cache
```

Флаг `--previous data/reports/old.json` в `collect` и `search` переиспользует
страницы, у которых не изменился content hash. В отчёте есть `content_hash`,
`cache_hit` и блок `cache` в `stats`, поэтому переиспользование прозрачно.

### Первый запуск: известные URL

```powershell
web2spec collect `
  --task "Собери названия, авторов и даты публикации статей" `
  --url https://habr.com/ru/articles/1103276/ `
  --out data/reports/demo.json
```

### Первый запуск: research

```powershell
web2spec research `
  --task "Сравни тарифы распознавания речи Yandex SpeechKit, Sber SaluteSpeech и VK Cloud в 2026: цена за час, бесплатный лимит, русский язык, диаризация. Используй официальные страницы и честно покажи missing." `
  --domains developers.sber.ru,cloud.vk.com `
  --min-pages 20 --max-pages 30 `
  --out data/reports/research.json
```

Один research-вызов обычно занимает 2–4 минуты на эталонном ноутбуке. JSON
содержит `answer`, `entities`, `sources`, `gaps`, `errors`, `stats` и `speed`;
`--include-notes` доступен для аудита.

### MCP-сервер

`web2spec mcp` запускает stdio MCP-сервер. Любой MCP-клиент может вызвать два
инструмента:

| Инструмент | Когда использовать | Основной вывод |
| --- | --- | --- |
| `research` | Задача расплывчатая, источники и поля неизвестны | `answer`, `entities`, `sources`, `gaps`, `errors` |
| `collect` | Уже есть официальные URL | агрегированные `facts`, `conflicts`, `missing`, `by_source` |

Скопируйте [`configs/mcp.example.json`](configs/mcp.example.json) в конфиг
клиента и замените пути. Claude Desktop хранит его в
`claude_desktop_config.json`; Cursor использует `~/.cursor/mcp.json`.

```json
{
  "mcpServers": {
    "web2spec": {
      "command": "C:\\path\\to\\repo\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\repo\\scripts\\23_mcp_server.py"],
      "env": {
        "WEB2SPEC_PROVIDER": "ddg",
        "WEB2SPEC_LLM_MODEL": "qwen2.5:3b"
      }
    }
  }
}
```

`WEB2SPEC_LLM_MODEL` необязателен и влияет только на старый путь `search`; MCP
`research` использует локальный 4B-аналитик. На GPU с 4 GB не запускайте
`research` и `collect` одновременно. Сервер пишет диагностику в stderr;
stdout остаётся чистым JSON-RPC.

Ответ — компактный JSON, никогда HTML. Типичный результат `collect`:

```json
{
  "task": "Сравни тарифы распознавания речи",
  "facts": { "russian": true },
  "conflicts": {
    "price_per_hour_rub": [
      { "value": 36, "sources": ["https://developers.sber.ru/..."] },
      { "value": 30, "sources": ["https://cloud.vk.com/..."] }
    ]
  },
  "missing": { "diarization": ["https://cloud.vk.com/..."] },
  "by_source": {
    "https://developers.sber.ru/...": { "price_per_hour_rub": 36 },
    "https://cloud.vk.com/...": { "price_per_hour_rub": 30 }
  },
  "sources": ["https://developers.sber.ru/...", "https://cloud.vk.com/..."],
  "errors": [],
  "stats": { "requested": 2, "ok": 2, "failed": 0 }
}
```

Если источники расходятся, возвращаются оба значения. Конвейер не выбирает
победителя.

## CLI

```powershell
web2spec doctor
web2spec download-models --only collector
web2spec collect --task "..." --urls data/urls.txt
web2spec search  --task "..." --provider ddg --search-only
web2spec research --task "..." --domains vendor-x.ru --min-pages 20
web2spec mcp
web2spec benchmark
web2spec eval-q4 --limit 50
```

| Команда | Назначение |
| --- | --- |
| `download-models` | скачать GGUF и проверить checksums |
| `upload-models` | зеркалировать дообученные артефакты на Hugging Face |
| `benchmark` | прогнать датированные реальные страницы через collector |
| `doctor` | проверить модели, `llama-server` и Python |
| `collect` | задача + URL -> структурированные факты |
| `search` | задача -> запросы -> локальный поиск -> collector |
| `research` | задача -> 20–50 страниц -> отчёт с evidence |
| `mcp` | MCP stdio-сервер |
| `eval-q4` | оценка качества/скорости Q4_K_M |
| `eval-runtime` | оценка PyTorch-каскада |
| `eval-plan`, `eval-plan-extract` | оценка planner/extractor |
| `train`, `merge` | QLoRA-обучение и merge адаптера |

Любой numbered-скрипт из [`scripts/`](scripts/) остаётся самостоятельным.
Полная карта — в [`scripts/README.md`](scripts/README.md).

## Поисковые провайдеры

- **SearXNG JSON** (рекомендуется): self-hosted, агрегирует публичные движки,
  без API-ключа. Включите `formats: [html, json]` в `settings.yml` и передайте
  `--provider searxng --searxng-url http://localhost:8080`.
- **DuckDuckGo HTML** (fallback): без установки, но публичный endpoint и
  best-effort; может отдавать 429 или captcha.

`search` может использовать локальную LLM для генерации запросов (Ollama или
любой OpenAI-совместимый сервер). Без неё текст задачи используется как
запрос. `research` не нуждается в Ollama: запросы пишет его 4B-аналитик.

## Модели

| Роль | Файл | Размер | Источник |
| --- | --- | ---: | --- |
| Аналитик (`research`) | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2381.6 MB | официальный Qwen GGUF |
| Planner (`collect`) | `qwen05b-plan-Q4_K_M.gguf` | 379.4 MB | дообученный Qwen2.5-0.5B |
| Extractor (`collect`) | `qwen05b-plan-extract-v2-Q4_K_M.gguf` | 379.4 MB | дообученный Qwen2.5-0.5B |

Базовые модели — Apache-2.0. Дообученные GGUF прикреплены к GitHub release;
те же артефакты можно зеркалировать на Hugging Face командой `upload-models`.
Checksums, шаги пересборки, экспериментальные Q3_K_M/Q2_K и нюанс
квантования — в [`docs/MODELS.md`](docs/MODELS.md). Для пересборки collector
нужны обучающие данные, которых нет в git.

Q3_K_M/Q2_K собраны локально; перед скачиванием их release-ассеты нужно
прикрепить отдельно:

```powershell
web2spec download-models --only collector-q3   # экспериментальные planner + extractor
web2spec download-models --only collector-q2
```

PyTorch fallback (`--backend hf`) требует extra `hf`:

```powershell
pip install -e ".[hf]"
web2spec collect --task "..." --urls data/urls.txt `
  --backend hf `
  --planner models/merged/qwen05b-plan `
  --extractor models/merged/qwen05b-plan-extract-v2
```

Обучение и полная оценка используют `.[train]`:

```powershell
pip install -e ".[train]"
web2spec train --config configs/train_plan_extract_reliable_4gb.yaml
```

## Контракт вывода

`collect` пишет один JSON:

```json
{
  "task": "Собери названия и авторов статей",
  "generated_at": "2026-11-06T12:00:00+00:00",
  "pages": [
    {
      "url": "https://example.com/article/1",
      "status": "ok",
      "entity": "article",
      "facts": [
        { "name": "title", "value": "Как ускорить сбор данных", "evidence": "Как ускорить сбор данных" }
      ],
      "missing": ["published_at"],
      "repairs": { "evidence": [], "dropped": [] },
      "dropped_foreign": [],
      "error": null
    }
  ],
  "aggregate": {
    "https://example.com/article/1": {
      "status": "ok",
      "entity": "article",
      "facts": { "title": "Как ускорить сбор данных" },
      "missing": ["published_at"],
      "error": null
    }
  },
  "stats": { "requested": 1, "ok": 1, "failed": 0, "by_status": { "ok": 1 } }
}
```

`status`: `ok`, `fetch_error`, `plan_error` или `extract_error`. `missing` —
не сбой запуска, а честный ответ, что проверяемого значения на странице нет.
Коды выхода: `0` — хотя бы одна страница `ok`, `1` — все страницы с ошибкой,
`2` — неверные аргументы.

`research` добавляет `goal`, `questions`, `entities` (`facts`, `conflicts`,
`evidence`), `notes`, `gaps`, `sources`, `errors`, `stats` и `speed`.

## Структура репозитория

```text
src/web2spec/       библиотека: схемы, сжатие HTML, research loop, CLI
scripts/            numbered pipeline: данные -> обучение -> eval -> runtime -> MCP
configs/            YAML-конфиги, MCP-пример, seeds
tests/              pytest без GPU и сети
docs/               архитектура, бенчмарки, модели, история экспериментов
examples/           небольшие примеры задач и URL
data/               датасеты и отчёты (gitignored)
models/             GGUF-веса (gitignored) и checksum manifest
outputs/runs/       адаптеры, чекпоинты, логи обучения (gitignored)
```

Ключевые модули:

- `web2spec/html_compress.py` — сжатие HTML и извлечение метаданных;
- `web2spec/semantic.py` — семантические чанки: заголовки, абзацы, строки
  таблиц;
- `web2spec/research.py` — question-first цикл, заметки, пробелы, агрегация;
- `web2spec/plan.py` — `CollectionPlan`, `FactSet`, `repair_factset`,
  валидация;
- `web2spec/llama_runtime.py` — жизненный цикл `llama-server` и Q4-пути;
- `web2spec/tokens.py` — ChatML-промпты без зависимости от tokenizer;
- `web2spec/cli.py` — единая точка входа `web2spec`.

## Разработка

```powershell
pip install -e ".[dev]"
python -m ruff check .
python -m pytest
```

Тесты не требуют GPU и сети: 240+ тестов покрывают схемы, сжатие, repair,
агрегацию research, парсинг поиска, MCP-протокол и CLI. CI запускает те же
команды на Ubuntu и Windows с Python 3.12.

Данные и отчёты не коммитятся. Исходные эксперименты, планы и метрики
сохранены в `docs/` и numbered-скриптах. Перед изменением контракта evidence
или добавлением нового пути модели прочитайте
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Ограничения

- **Узкая доказательная база.** Качество collector измерено только на
  article/Habr, research — только на Habr-фикстурах. Мульти-доменного
  бенчмарка нет. Не подавайте эти цифры как общую точность.
- **Нет браузера.** Fetcher не исполняет JavaScript, не ждёт SPA, не решает
  captcha, не обходит WAF, не логинится и не листает пагинацию. Закрытая
  страница получает `blocked`/`fetch_error`, а не тихо пропускается.
- **Поиск best-effort.** Публичные метапоисковики могут rate-limit. Для
  повторяемой работы рекомендуется SearXNG.
- **Недоверенный текст страниц.** В страницах может быть prompt injection.
  Конвейер возвращает данные, а не инструкции или tool calls, и требует точный
  evidence, но злонамеренная страница всё равно может повлиять на то, что
  модель заметит. Считайте вывод недоверенными данными и проверяйте важные
  числа в источнике.
- **Одна модель за раз.** На 4 GB VRAM 4B-аналитик и 0.5B-collector вместе не
  помещаются. CLI запускает и останавливает `llama-server` вокруг каждой
  стадии.
- **Это не краулер.** `collect` обрабатывает ровно переданные URL. Он не
  находит весь каталог сайта и не дедуплицирует сущности между страницами.
- **Веса отдельно.** В чистом клоне нет `.gguf`. Запустите
  `web2spec download-models` или соберите модели через training pipeline.

## Лицензия

MIT — см. [`LICENSE`](LICENSE). Базовые модели Qwen — Apache-2.0; лицензии и
условия распространения весов — в [`docs/MODELS.md`](docs/MODELS.md) и
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Датасеты и скачанные
страницы не распространяются.
