# Implementation plan — web2spec v1

**Цель:** локальная модель 0.5B, которая по сжатому HTML и инструкции выдаёт JSON, соответствующий целевой схеме. Обучение — QLoRA на 4 GB VRAM, артефакт — GGUF Q4_K_M, инференс — `llama-server`.

**Не-цели v1:** генерация Python-скраперов, 1.5B+, full fine-tune, multi-GPU, обучение на 32k контексте, открытый чат. Генерация «спеки» (селекторы) — v2, после того как извлечение стабильно.

---

## 1. Архитектура

```
data/raw/*.html ──> compress_html ──> interim JSONL
                                        │
teacher-модель ──> synth ──> execution/схема-фильтр ──┘
                                        │
                                   validate.py
                              (схема + токены + дедуп)
                                        │
                         train.jsonl / val.jsonl / test.jsonl
                                        │
                                  QLoRA (4-bit NF4)
                                        │
                                  adapter (r=16)
                                        │
                              merge_and_unload → fp16 HF
                                        │
                          convert_hf_to_gguf → f16
                                        │
                              llama-quantize → Q4_K_M
                                        │
                    llama-server (приёмка) / Ollama (раздача)
```

Два независимых контура: **данные** (слева) и **модель** (справа). Они сходятся только на `data/processed/*.jsonl`, поэтому могут разрабатываться параллельно.

---

## 2. Контракты (уже зафиксированы)

`src/web2spec/schema.py`:

| Сущность | Назначение |
|---|---|
| `Sample` | строгий контракт одного примера: `id`, `group_id`, `task`, `instruction`, `input_html`, `target`, `meta` |
| `validate_sample(raw, task_schemas)` | поднимает `SampleError(code, message)`; коды `E_SCHEMA`, `E_TARGET`, `E_NOT_OBJECT` |
| `ValidationReport` | агрегат: `total/passed/failed`, `by_reason`, перцентили токенов, первые 200 issues |
| `make_id(prefix, *parts)` | детерминированный id (blake2b, 12 байт) |
| `load_task_schemas(path)` | YAML `task → JSON Schema` |

`src/web2spec/config.py`:

| Сущность | Назначение |
|---|---|
| `REPO_ROOT` | `transformer/`, все относительные пути резолвятся от него |
| `Paths` | `raw/interim/processed/reports/base_model/merged/gguf/runs` |
| `DataConfig` | tokenizer, `max_seq_len=1536`, лимиты длины, `val_ratio`, `seed`, `task_schemas` |
| `TrainConfig` | quant + lora + batch/accum/epochs/lr/optim/attn + шаги логирования |
| `EvalConfig` | model_dir, data, report, max_new_tokens, temperature, limit |
| `load_*_config()` | строгая загрузка YAML (`extra="forbid"`) |

**Правило:** эти два файла — общий контракт. Субагенты их не редактируют. Если чего-то не хватает — добавляют приватный helper в своём модуле и сообщают о пробеле в отчёте.

---

## 3. Структура репозитория

```
transformer/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data.yaml
│   ├── train_qlora_4gb.yaml
│   ├── eval.yaml
│   └── tasks.yaml
├── src/web2spec/
│   ├── __init__.py
│   ├── schema.py            [готово]
│   ├── config.py            [готово]
│   ├── html_compress.py     [M1]
│   ├── tokens.py            [M1]
│   ├── validate.py          [M1]
│   ├── dataset.py           [M2]
│   ├── train.py             [M2]
│   ├── merge.py             [M3]
│   ├── gguf.py              [M3]
│   ├── eval.py              [M4]
│   └── synth.py             [M5]
├── scripts/
│   ├── 01_validate.py       [M1]
│   ├── 02_train.py          [M2]
│   ├── 03_merge.py          [M3]
│   ├── 04_convert_gguf.ps1  [M3]
│   ├── 05_eval.py           [M4]
│   ├── 06_serve.ps1         [M4]
│   └── 07_synth.py          [M5]
├── tests/
│   ├── test_schema.py       [готово]
│   ├── test_config.py       [готово]
│   ├── test_html_compress.py
│   ├── test_validate.py
│   ├── test_dataset.py
│   ├── test_merge.py
│   ├── test_eval.py
│   └── test_synth.py
├── data/{raw,interim,processed,reports}/
├── models/{base,merged,gguf}/
├── outputs/runs/
└── docs/{PLAN.md, IMPLEMENTATION_PLAN.md, DATA_SOURCES.md}
```

---

## 4. Milestones

| # | Результат | Критерий приёмки |
|---|---|---|
| M0 | Контракты, конфиги, тесты, зависимости | `pytest` зелёный, `ruff` чистый, `torch.cuda.is_available()` |
| M1 | Сжатие HTML, токенизация, валидатор, CLI 01 | 100 примеров проходят/отклоняются с кодами; отчёт с перцентилями; group-aware split |
| M2 | Dataset + QLoRA-обучение, CLI 02 | `--smoke` (20 шагов) без OOM, adapter сохраняется, loss падает |
| M3 | Слияние + GGUF, CLI 03/04 | merged-папка грузится `AutoModelForCausalLM`; Q4_K_M файл создаётся |
| M4 | Eval + serve, CLI 05/06 | отчёт: JSON-valid rate, field-F1, exact match; сервер отвечает на curl |
| M5 | Синтетика через teacher, CLI 07 | mock-teacher в тестах; реальный прогон на 50 страницах даёт ≥20 валидных примеров |
| M6 | Источники данных и seed-набор | `docs/DATA_SOURCES.md` + 100 страниц с эталонами |
| M7 | End-to-end на 1k примеров | полный цикл до GGUF + eval-отчёт, зафиксированные метрики |

M1–M5 независимы по файлам и могут идти параллельно. M6 — исследовательская, не блокирует код. M7 — только после интеграции.

---

## 5. Распределение задач между субагентами

Все задачи получают один и тот же контекст: Windows 11, venv `.venv` (Python 3.12), GPU 4 GB, репозиторий `transformer/`, контракты в `src/web2spec/{schema,config}.py`, тяжёлые импорты — ленивые.

| ID | Агент | Зона ответственности (только эти файлы) | Зависит от | Приёмка |
|---|---|---|---|---|
| **S1** | implementer | `html_compress.py`, `tokens.py`, `validate.py`, `scripts/01_validate.py`, `tests/test_html_compress.py`, `tests/test_validate.py` | M0 | `pytest tests/test_validate.py tests/test_html_compress.py`; CLI на 100 строках даёт отчёт; HTML сжимается до ≤12k символов без потери текста |
| **S2** | implementer | `dataset.py`, `train.py`, `scripts/02_train.py`, `tests/test_dataset.py` | M0 | датасет собирается и маскирует prompt; `--smoke` проходит на GPU; `--dry-run` без torch не падает |
| **S3** | implementer | `merge.py`, `gguf.py`, `scripts/03_merge.py`, `scripts/04_convert_gguf.ps1`, `tests/test_merge.py` | M0 | `find_binary` находит exe в `build/bin/Release`; merge валидирует входы; ps1 идемпотентен |
| **S4** | implementer | `eval.py`, `scripts/05_eval.py`, `scripts/06_serve.ps1`, `tests/test_eval.py` | M0 | метрики считаются на синтетических парах; отчёт детерминирован; сервер-скрипт проверяет наличие модели |
| **S5** | implementer | `synth.py`, `scripts/07_synth.py`, `tests/test_synth.py` | M0 | mock-teacher, парсинг ```json-блоков, отбраковка невалидного; без сети в тестах |
| **S6** | general | `docs/DATA_SOURCES.md` | M0 | 10+ конкретных источников с лицензиями, схема seed-набора, протокол проверки |

**Правила для субагентов:**

1. Не редактировать чужие файлы, `schema.py`, `config.py`, `pyproject.toml`, `README.md`.
2. Не запускать `pip install`, `git`, не создавать новые зависимости без отчёта.
3. Тяжёлые импорты — внутри функций; модуль должен импортироваться без torch.
4. Тесты — только на своих файлах, `--basetemp=.pytest_tmp/<id>`, без сети.
5. Финальный отчёт: что создано, как проверено, какие допущения, какие пробелы в контракте.

---

## 6. Интерфейсы, которые обязаны совпасть

```python
# html_compress.py
def compress_html(html: str, *, max_chars: int = 12_000) -> str

# tokens.py
def load_tokenizer(name_or_path: str)
def build_prompt(sample: Sample, tokenizer) -> str
def count_tokens(text: str, tokenizer) -> int

# validate.py
def validate_file(input_path: Path, *, cfg: DataConfig,
                  task_schemas: dict[str, dict], tokenizer=None,
                  max_seq_len: int | None = None,
                  dedup: bool = True) -> tuple[list[Sample], ValidationReport]
def split_samples(samples: list[Sample], *, val_ratio: float,
                  seed: int) -> dict[str, list[Sample]]
def write_jsonl(samples: list[Sample], path: Path) -> Path

# dataset.py
class ExtractionDataset:          # duck-typed, без наследования от torch
    def __len__(self) -> int
    def __getitem__(self, index: int) -> dict[str, list[int]]
class CausalLMCollator:
    def __call__(self, features: list[dict]) -> dict

# train.py
def train(cfg: TrainConfig, data_cfg: DataConfig, *, smoke: bool = False) -> Path

# merge.py
def merge_adapter(base_model: str | Path, adapter_dir: Path,
                  out_dir: Path, *, dtype: str = "float16") -> Path

# gguf.py
def find_binary(llama_cpp: Path, name: str) -> Path
def convert(model_dir: Path, out_file: Path, *, llama_cpp: Path,
            outtype: str = "f16") -> Path
def quantize(src: Path, dst: Path, *, llama_cpp: Path,
             quant_type: str = "Q4_K_M") -> Path

# eval.py
def field_scores(pred: dict, gold: dict) -> dict[str, float]
def evaluate_records(records: list[dict]) -> dict
def run(cfg: EvalConfig) -> dict

# synth.py
class Teacher(Protocol):
    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str
def synthesize(inputs: list[dict], teacher: Teacher, *,
               schema: dict) -> tuple[list[dict], dict]
```

---

## 7. Порядок интеграции

1. M0 закоммичен (текущее состояние).
2. S1–S6 работают параллельно в одном дереве, но в непересекающихся файлах.
3. После возврата: `ruff check src tests`, `pytest`, `python -m compileall src scripts`.
4. Ручная проверка стыков: `01_validate → 02_train --dry-run` на одном файле; `03_merge` на фиктивном adapter-каталоге; `05_eval` на 5 предсказаниях.
5. Только после этого — реальные данные, скачивание модели, smoke-обучение.
6. Каждый milestone — отдельный коммит; при провале `git reset --hard <tag>`.

---

## 8. Проверки

```powershell
# без ML-стека
pytest
ruff check src tests
python -m compileall -q src scripts

# после установки torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/02_train.py --dry-run
python scripts/02_train.py --smoke
```

Метрики приёмки модели (M7), а не «красивый loss»:

- `json_valid_rate` — доля ответов, парсящихся как JSON;
- `field_f1` micro/macro по плоским ключам;
- `exact_match` — полное совпадение объекта;
- `latency_p50/p95` на 256 новых токенах;
- срез по `meta.complexity` и по доменам (train vs unseen).

---

## 9. Риски и rollback

| Риск | Митигация |
|---|---|
| bitsandbytes не заводится на CUDA 12.x/Windows | проверить импорт до обучения; версия колеса фиксируется в `requirements.lock` |
| OOM на seq 1536 | fallback: seq 1024, r=8, accum 8; smoke ловит до полного запуска |
| Субагенты конфликтуют в общих файлах | жёсткое владение файлами + `git diff` перед коммитом |
| Синтетика шумная | фильтр по схеме + дедуп + ручная выборка 5% |
| Модель выучила шаблон, а не извлечение | eval на unseen-доменах и мутациях DOM; baseline-эвристика для сравнения |
| Долгая установка torch | уже идёт в фоне, лог `logs/pip_heavy.log` |

**Rollback:** `git tag m0` после текущего коммита; каждый следующий milestone — тег. Откат одного субагента — `git checkout m0 -- <его файлы>`.
