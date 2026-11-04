# Статистика датасетов

Срез на 04.11.2026. Все размеры — на диске, все токены — Qwen2.5 tokenizer,
`build_prompt` с system prompt и инструкцией.

## 1. Extract-датасет (HTML → JSON)

### Сырые данные: 877 примеров, 3.76 MB

| Источник | Файл | Примеров | HTML, символов (min/median/max) |
|---|---|---:|---|
| Lenta | `data/raw/articles/lenta.ru.jsonl` | 172 | 3151 / 3448 / 3500 |
| Habr | `data/raw/articles/habr.com.jsonl` | 40 | 3127 / 3465 / 3500 |
| hh.ru | `data/raw/listings/kaluga.hh.ru.jsonl` | 100 | 2342 / 2384 / 2400 |
| Regard | `data/raw/products/www.regard.ru.jsonl` | 60 | 3299 / 3473 / 3499 |
| Citilink | `data/raw/products/www.citilink.ru.jsonl` | 5 | 11842 / 11983 / 11993 |
| Synthetic product | `data/raw/synthetic/product.jsonl` | 100 | 2057 / 2341 / 3191 |
| Synthetic article | `data/raw/synthetic/article.jsonl` | 100 | 2572 / 2763 / 2837 |
| Synthetic listing | `data/raw/synthetic/listing.jsonl` | 100 | 1854 / 2351 / 2846 |
| Synthetic job | `data/raw/synthetic/job.jsonl` | 100 | 1602 / 1864 / 2169 |
| Synthetic realestate | `data/raw/synthetic/realestate.jsonl` | 100 | 1633 / 1895 / 2121 |
| **Итого** | | **877** | |

Из них:
- **реальные страницы — 377** (212 статей, 100 вакансий, 65 товаров);
- **синтетика — 500** (по 100 на категорию).

### После валидации: 866 примеров, 3.17 MB

| Split | Примеров |
|---|---:|
| train | 744 |
| val | 61 |
| test | 61 |

Отклонено 11: 6 Lenta и 5 Citilink по `E_TOO_LONG`.

Токены промпта: **p50 1015, p95 1334, max 1416** при `max_seq_len=1536`.

## 2. Spec-датасет (HTML → spec)

### Синтетический: 600 примеров, 1.36 MB

| Split | Примеров |
|---|---:|
| train | 480 |
| val | 60 |
| test | 60 |

- 4 категории × 150: product, article, job, realestate;
- 3 варианта вёрстки;
- 70% списков / 30% детальных страниц;
- 300 RU / 300 EN;
- каждый spec проверен исполнением на своей странице при генерации.

Токены промпта: **p50 450, p95 694, max 765**.

### Реальный: 157 примеров, 0.72 MB

Построен из реальных extract-страниц: значение ищется в видимом DOM, из него
строится CSS-селектор, spec проверяется исполнением.

| Split | Всего | Lenta | Regard | hh.ru |
|---|---:|---:|---:|---:|
| train | 125 | 67 | 50 | 8 |
| val | 16 | 10 | 6 | 0 |
| test | 16 | 13 | 3 | 0 |

Habr в real-spec не попал: не удалось надёжно сопоставить значения target с
видимыми элементами после сжатия.

Токены промпта: **p50 1306, p95 1367, max 1399**.

## 3. Модели и артефакты

| Артефакт | Размер | Статус |
|---|---:|---|
| Extract LoRA `outputs/runs/qwen05b-qlora-v1/adapter` | 33.6 MB | обучен |
| Extract merged `models/merged/qwen2.5-0.5b-ft` | 942 MB | готов |
| Extract GGUF Q4_K_M | 379.4 MB | готов, llama-server проверен |
| Spec LoRA `outputs/runs/qwen05b-spec-v1/adapter` | 33.6 MB | обучен на синтетике |
| Spec merged `models/merged/qwen05b-spec` | 942 MB | готов, но не обобщает |
| Spec-real `outputs/runs/qwen05b-spec-real/checkpoint-20` | — | обучение остановлено на 32/48 |

## 4. Качество

### Extract, test 61

| Метрика | База | QLoRA |
|---|---:|---:|
| Валидный JSON | 78.7% | **98.4%** |
| Exact match | 0% | **42.6%** |
| Micro F1 | 18.6% | **84.3%** |
| Macro F1 | 17.9% | **76.7%** |

По доменам: synthetic 0.95–1.0, Regard 0.80, Lenta 0.67, hh.ru 0.39, Habr 0.17.

### Spec, синтетический test 60

`spec_valid=1.0`, `exec=1.0`, `page_exact=1.0`, `F1=1.0`.

### Spec, реальные страницы (18)

`valid/executed ≈ 44%`, `page_exact=0`, `field_F1=0`.

Вывод: синтетический spec-adapter выучил классы шаблонов, а не реальную вёрстку.
Нужен spec-real adapter; его обучение остановлено по команде.

## 5. Что готово к запуску

- `configs/data_spec_real.yaml` — реальные spec-данные;
- `configs/train_spec_real_4gb.yaml` — QLoRA на 6 эпох, 48 шагов;
- `scripts/12_eval_spec.py` — eval через исполнение;
- `scripts/14_make_real_spec_data.py` — расширение real-spec датасета.

Обучение не запущено. Незавершённый `checkpoint-20` оставлен для возможного
resume, но финального adapter нет.
