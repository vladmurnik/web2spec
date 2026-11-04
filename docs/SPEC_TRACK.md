# Scraping-spec: второй трек

## Зачем

Текущий adapter решает `HTML → JSON` (extraction). Он не умеет ходить по сайту и не
пишет код. Цель этого трека — научить модель выдавать **исполнимую спеку скрапера**,
а не свободный Python.

```
HTML + инструкция → spec (JSON) → детерминированный рендерер → Python / результат
```

Так 0.5B не выдумывает API библиотек: она выбирает селекторы и поля, а код собирается
шаблонно.

## Формат spec

```json
{
  "version": 1,
  "item_selector": "div.card",
  "fields": {
    "title": {"selector": "a.title", "required": true},
    "url": {"selector": "a.title", "attr": "href", "required": true},
    "price": {"selector": "span.price"}
  },
  "pagination": {"selector": "a.next", "attr": "href", "max_pages": 3}
}
```

- `item_selector` — если задан, поля извлекаются из каждого элемента списка;
  иначе страница считается детальной и возвращается один объект.
- `attr` — атрибут (`href`, `src`, ...); если не задан, берётся текст элемента.
- `pagination` — ссылка на следующую страницу.

Схема продублирована в `configs/tasks.yaml` (задача `spec`).

## Компоненты

| Файл | Назначение |
|---|---|
| `src/web2spec/spec.py` | Pydantic-модель, `load_spec`, `extract`, `next_page_url`, `render_python` |
| `scripts/11_make_spec_data.py` | генератор 600 синтетических страниц + spec + gold |
| `scripts/12_eval_spec.py` | генерация spec и оценка **через исполнение** на HTML |
| `configs/data_spec.yaml` | данные `data/spec`, tokenizer, seq 1536 |
| `configs/train_spec_4gb.yaml` | QLoRA-конфиг для второго adapter |
| `tests/test_spec.py` | 10 тестов: извлечение, атрибуты, пагинация, рендерер |

## Данные

`data/spec/`:

- `train.jsonl` — 480;
- `val.jsonl` — 60;
- `test.jsonl` — 60;
- `gold/{split}.jsonl` — ожидаемые извлечённые данные (sidecar, не часть Sample);
- `stats.json` — состав.

4 категории (product, article, job, realestate), 3 варианта вёрстки, 70% списков и
30% детальных страниц, 50/50 RU/EN. Каждый spec проверяется исполнением на той же
сжатой странице ещё на этапе генерации.

## Обучение

```powershell
.venv\Scripts\python.exe scripts\02_train.py `
  --config configs\train_spec_4gb.yaml `
  --data-config configs\data_spec.yaml
```

Base: `Qwen/Qwen2.5-0.5B-Instruct` (Coder-0.5B пробовали, но загрузка с HF оказалась
непрактично медленной; для spec-задачи structured JSON важнее code-бенча).
QLoRA NF4, r=16, seq 1536, effective batch 16, 3 эпохи, ~90 шагов.

## Оценка

```powershell
.venv\Scripts\python.exe scripts\12_eval_spec.py `
  --model models\merged\qwen05b-spec `
  --data data\spec\test.jsonl `
  --gold data\spec\gold\test.jsonl
```

Метрики:

- `spec_valid_rate` — spec разобрался как JSON по схеме;
- `exec_success_rate` — spec выполнился на HTML;
- `page_exact` — извлечённые данные полностью совпали с gold;
- `field_f1` — точность по полям.

Это честнее текстового exact match: проверяется реальный результат на странице.

## Ограничения

1. Синтетика: 3 шаблона вёрстки, модель может выучить классы, а не общий навык.
2. Нет реальных страниц в spec-треке; следующий шаг — собрать 100–200 реальных
   списков и разметить селекторы.
3. Модель не выполняет HTTP-запросы и не обходит антибот; это задача runtime.
4. Для свободного кода на bs4/Playwright нужен Coder 1.5B и execution-eval.
