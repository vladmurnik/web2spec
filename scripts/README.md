# Индекс скриптов

Numbered-скрипты сохранены как воспроизводимый pipeline. Для повседневной
работы используйте единый CLI `web2spec`; он вызывает те же файлы.

```powershell
bash scripts/00_install.sh        # Linux/macOS
powershell -File scripts/00_install.ps1  # Windows
web2spec download-models --only all
web2spec doctor
web2spec collect --task "..." --urls data/urls.txt
web2spec search --task "..." --provider ddg
web2spec research --task "..." --min-pages 20 --max-pages 30
web2spec mcp
```

## Данные и обучение

| Скрипт | Назначение |
|---|---|
| `00_install.ps1`, `00_install.sh` | установка venv, зависимостей, моделей и doctor |
| `00_download_models.py` | скачивание GGUF и проверка SHA-256; Q4/Q3/Q2 и группы `collector-*` |
| `00_upload_models.py` | зеркалирование GGUF/merged на Hugging Face (нужен `HF_TOKEN`) |
| `01_validate.py` | нормализация JSONL, валидация, дедуп, splits |
| `02_train.py` | QLoRA-обучение или dry-run |
| `03_merge.py` | слияние LoRA-адаптера с базовой моделью |
| `04_convert_gguf.ps1` | merged-каталог → GGUF / Q4_K_M |
| `05_eval.py` | оценка merged-модели или сохранённых предсказаний |
| `06_serve.ps1` | локальный сервер для ручных проверок |
| `07_synth.py` | синтетика через teacher-модель |
| `08_collect.py` | сбор реальных страниц по JSON-LD |
| `09_make_synthetic.py` | офлайн-синтетика HTML → JSON |
| `10_split_dataset.py` | стратифицированный split |
| `11_make_spec_data.py` | синтетический датасет scraping-spec |
| `12_eval_spec.py` | оценка spec через исполнение |
| `14_make_real_spec_data.py` | real-page spec из существующих меток |
| `15_make_plan_tasks.py` | bootstrap-датасет task-conditioned planner |
| `17_eval_plan.py` | оценка planner |
| `18_make_plan_extract_tasks.py` | plan-conditioned extraction-датасет |
| `19_eval_plan_extract.py` | оценка plan-conditioned extractor |
| `20_eval_runtime.py` | runtime-каскад planner → extractor на PyTorch |

## Runtime и интерфейсы

| Скрипт | Назначение |
|---|---|
| `21_collect_urls.py` | задача + URL → сжатие → planner → extractor → факты; `--cache`, `--previous`, lenient-план и `--allow-partial` |
| `22_search_and_collect.py` | задача → запросы → локальный поиск → collector; `--domain` / `--domains`; кэш страниц и выдачи |
| `23_mcp_server.py` | MCP stdio-сервер: `research` (4B, 20–50 страниц) и `collect` (быстрый URL-first) |
| `24_eval_q4km.py` | тот же runtime-тест на Q4_K_M GGUF |
| `26_research.py` | question-first research: 4B-аналитик, 20–50 страниц, заметки с цитатами, отчёт |
| `27_benchmark.py` | реальные страницы + датированный gold → точность полей collector |

## Порядок воспроизведения

1. `01_validate.py` → `10_split_dataset.py`.
2. `02_train.py` → `03_merge.py` → `04_convert_gguf.ps1`.
3. `05_eval.py`, `17_eval_plan.py`, `19_eval_plan_extract.py`,
   `20_eval_runtime.py`, `24_eval_q4km.py`.
4. Runtime: `21_collect_urls.py`, `22_search_and_collect.py`,
   `26_research.py`, `23_mcp_server.py`.
5. Real-page regression: `27_benchmark.py` (или `web2spec benchmark`).

`13`, `16` и `25` в репозитории отсутствуют: нумерация отражает исторические
шаги, а не непрерывную последовательность.
