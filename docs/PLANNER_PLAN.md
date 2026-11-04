# План следующего шага: task-conditioned planner

**Статус:** проект плана, обучение не запущено. Документ описывает следующий шаг после extraction и spec-трека: научить модель по задаче пользователя строить план сбора данных, а не заполнять фиксированную схему.

**Связанные документы:** [`MASTER_PLAN.md`](MASTER_PLAN.md) — общий roadmap; [`STATS.md`](STATS.md) — текущие данные; [`SPEC_TRACK.md`](SPEC_TRACK.md) — результаты spec-трека.

---

## 1. Цель

Дано:

- задача на естественном языке: «сравни цены на RTX 5060 Ti», «собери вакансии с зарплатой», «найди статьи про рынок недвижимости»;
- сэмпл сайта: URL, главная/листинг/карточка, сжатый HTML.

Получить:

- `CollectionPlan` — сущность, поля, фильтры, пагинация, detail pages, ключ дедупликации;
- план не содержит CSS-селекторов: это семантика, а не реализация;
- по плану extractor извлекает факты, renderer при необходимости строит код.

### Non-goals

- не учить модель ходить по сайту и обходить антибот;
- не генерировать Python-код в этом этапе;
- не заменять текущий extraction-адаптер;
- не обещать «любую задачу на любом сайте» для 0.5B.

---

## 2. Гипотеза и критерий успеха

**Гипотеза:** если на одной странице показать 3–5 разных задач с разными наборами полей, 0.5B выучит не «все поля подряд», а отображение «задача → релевантные поля». Разделение planner/extractor не даст схеме доминировать.

**Минимальный успех:**

- plan field F1 ≥ **0.75** на val;
- entity accuracy ≥ **0.85**;
- empty-plan accuracy ≥ **0.80**;
- end-to-end fact F1 ≥ **0.60** на unseen hosts.

Если порог не достигнут — planner остаётся teacher-only, 0.5B не внедряется.

---

## 3. Схема данных

### 3.1 CollectionPlan

```json
{
  "version": 1,
  "goal": "сравнить цены на видеокарты RTX 5060 Ti",
  "entity": "product",
  "entry": "listing",
  "fields": [
    {"name": "model", "why": "сопоставление товаров", "required": true, "kind": "text"},
    {"name": "price", "why": "сравнение цен", "required": true, "kind": "number"},
    {"name": "availability", "why": "можно ли купить", "required": false, "kind": "text"}
  ],
  "filters": {"model": "RTX 5060 Ti"},
  "pagination": true,
  "detail_pages": true,
  "dedup_key": "url",
  "stop_conditions": ["нет следующей страницы", "достигнут лимит 20 страниц"]
}
```

Ограничения:

- `goal` 5–300 символов;
- `entity` — snake_case, одно слово;
- `entry` ∈ `listing | detail | search | sitemap`;
- `fields` 1–12, имена уникальны, snake_case;
- `kind` ∈ `text | number | date | url | bool | null`;
- `filters` ≤ 8 пар;
- `dedup_key` ∈ `url | id | sku | title`;
- `stop_conditions` ≤ 5 строк;
- никаких `selector`, `xpath`, `css` в плане.

### 3.2 FactSet (для следующего этапа)

```json
{
  "facts": [
    {"name": "model", "value": "ASUS RTX 5060 Ti", "evidence": "ASUS ... RTX 5060TI"},
    {"name": "price", "value": 89560, "evidence": "89 560 ₽"}
  ],
  "missing": ["availability"]
}
```

- `evidence` — точная подстрока сжатого HTML;
- `value` — скаляр (string / number / bool / null); списки в v1 не поддерживаются;
- `missing` — поля плана, которых на странице нет;
- факт без evidence отбрасывается валидатором.

### 3.3 Sample для обучения

Используем существующий `Sample`, добавляем `task="plan"`:

```json
{
  "id": "plan-...",
  "group_id": "lenta.ru",
  "task": "plan",
  "instruction": "Сравни цены на видеокарты RTX 5060 Ti",
  "input_html": "<main>...</main>",
  "target": {"plan": {"version": 1, "...": "..."}},
  "meta": {"lang": "ru", "complexity": 2, "source_url": "...", "n_tokens": null}
}
```

Требуется:

- расширить `TaskName` значением `plan`;
- добавить `plan` в `configs/tasks.yaml`;
- отдельный `build_plan_prompt` в `tokens.py`, чтобы не смешивать с extraction-system-prompt;
- `01_validate.py` должен принимать `--task plan`.

---

## 4. Данные

### 4.1 Источники страниц

| Источник | Страниц | Роль |
|---|---:|---|
| Lenta | 172 | статьи, реальная вёрстка |
| Habr | 40 | статьи, другой шаблон |
| hh.ru | 100 | вакансии |
| Regard | 60 | товары |
| Citilink | 5 | товары, контроль |
| Synthetic | 500 | покрытие категорий и сложности |

Итого 877 страниц. Для planner используем **все реальные** и **200–300 синтетических** (не всю синтетику, чтобы не переобучиться на шаблоны).

### 4.2 Банк задач

5 категорий × 6 интентов:

| Интент | Пример |
|---|---|
| compare | «сравни цены и характеристики» |
| monitor | «отслеживай новые публикации» |
| enrich | «собери авторов и даты» |
| filter | «только товары в наличии» |
| research | «найди статьи про рынок» |
| negative | «собери данные о недвижимости» (на странице товара) |

Для каждой страницы генерируем 3–5 задач: 2–3 релевантные, 1 пограничную, 1 негативную/пустую. Формулировки — RU и EN, парафразы обязательны.

### 4.3 Teacher-разметка

1. `scripts/15_make_plan_tasks.py` создаёт `data/plan/tasks.jsonl` — задача + страница + category + intent.
2. `scripts/16_distill_plans.py` вызывает teacher-модель через `opencode run -m <model> --format json --file page.html`:
   - основной: `opencode/deepseek-v4.1-flash-free`;
   - массовая разметка: `opencode/flint-40b-free`;
   - резерв: `mimo-v2.6-flash-free`.
3. Ответ парсится, проверяется схемой, кэшируется по `(page_hash, task_hash, model)`.
4. Верификатор:
   - JSON соответствует `CollectionPlan`;
   - `entity` согласована с типом страницы;
   - поля не дублируются, не пустые, без селекторов;
   - `goal` не копирует задачу дословно;
   - для negative-задачи допускается `fields: []`? Нет: empty-plan кодируем отдельным флагом `"empty": true` или `fields: []` с `entity: "none"`. Решение: `entity: "none"`, `fields: []`, `entry: "detail"`, `pagination: false`.
5. 10% планов проверяет второй teacher; расхождения — в ручной аудит.
6. 100 планов размечает человек (или проверяет вручную) до обучения.

### 4.4 Объём и splits

- сырых планов: 877 × 3–5 ≈ **3 000–4 000**;
- после фильтра и дедупа: **2 000–2 500**;
- train / val / test: 80 / 10 / 10;
- split по `group_id` (хостам), а не по строкам;
- отдельный test: 2 хоста, не встречавшихся в train;
- отдельный test: 3 интента, не встречавшихся в train.

---

## 5. Обучение planner

### 5.1 База и адаптер

- база: `Qwen/Qwen2.5-0.5B-Instruct` (та же, что extraction);
- отдельный adapter: `outputs/runs/qwen05b-plan-v1`;
- extraction-adapter не трогаем.

### 5.2 Гиперпараметры

| Параметр | Значение |
|---|---|
| quant | 4-bit NF4, double quant, bf16 compute |
| max_seq_len | 1536 |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| target modules | q,k,v,o,gate,up,down |
| batch / accum | 1 / 16 |
| lr / schedule | 2e-4 / cosine |
| warmup | 3% |
| epochs | 3 |
| eval / save | каждые 50 шагов / 100 |
| early stopping | patience 2 |
| optim | paged_adamw_8bit |
| gradient checkpointing | on |
| seed | 42 |

Ожидаемо: 2–3 GB VRAM, 15–40 минут на 2–2.5 тыс. примеров. Перед полным запуском — smoke на 32 примерах.

### 5.3 Формат промпта

System:

```
Ты планировщик сбора данных. По задаче и образцу страницы верни только JSON CollectionPlan.
Не добавляй CSS-селекторы. Если задача не соответствует странице, верни entity="none" и пустые fields.
```

User:

```
Задача: {instruction}
Сайт: {source_url}
Тип страницы (предположительно): {category}
HTML-фрагмент:
{input_html}
```

Целевой ответ — компактный JSON без markdown.

---

## 6. Plan-conditioned extractor

После planner'а нужен второй adapter, иначе план не влияет на извлечение.

- adapter: `outputs/runs/qwen05b-plan-extract-v1`;
- вход: `CollectionPlan` + HTML;
- выход: `FactSet`;
- target строится из gold extraction: значения ищутся в HTML, evidence = точная подстрока;
- если поле плана отсутствует на странице — в `missing`;
- 2–3 тыс. примеров, seq 1536, те же гиперпараметры;
- eval: fact F1, evidence rate, missing accuracy.

Без этого этапа planner остаётся «советником», а extraction — фиксированным.

---

## 7. Оценка

### 7.1 Planner

| Метрика | Как считаем |
|---|---|
| entity accuracy | точное совпадение `entity` |
| field precision | доля полей плана, совпавших с gold |
| field recall | доля gold-полей, попавших в план |
| field F1 | harmonic mean |
| required recall | не потеряны обязательные поля |
| scope accuracy | совпали `pagination` и `detail_pages` |
| empty accuracy | правильно ли `entity="none"` |
| schema validity | доля валидных JSON |

### 7.2 End-to-end

- task success (ручная оценка 50 задач);
- fact F1;
- evidence rate;
- pages crawled;
- repair rate.

### 7.3 Baselines

1. Текущий extraction с фиксированной схемой.
2. Planner-teacher без student.
3. Student planner + текущий extractor (без plan-conditioned).
4. Полный каскад.

---

## 8. Runtime-каскад

```
task + page sample
   → planner (student)
   → schema validator
   → если invalid / entity=none / confidence low:
         teacher planner
   → fetcher/navigator
   → plan-conditioned extractor
   → fact validator (evidence)
   → если провал:
         teacher extraction + repair buffer
```

Confidence без logits оцениваем косвенно: schema validity, доля required-полей, согласованность с page type. Порог и правило эскалации фиксируем после eval.

---

## 9. Этапы

| Этап | Работа | Выход | Критерий |
|---|---|---|---|
| P0 | Схема `CollectionPlan` / `Fact`, тесты | `src/web2spec/plan.py`, `tests/test_plan.py` | 15+ тестов, negative-кейсы |
| P1 | Банк задач + 500 планов | `data/plan/tasks.jsonl`, `plans_500.jsonl` | 0 ошибок схемы, аудит 50 |
| P2 | Расширение до 2–2.5 тыс. | `data/plan/{train,val,test}.jsonl` | покрыты 5 категорий, 3 хоста в test |
| P3 | Smoke + обучение planner | adapter `qwen05b-plan-v1` | val F1 ≥ 0.75 |
| P4 | Eval + error analysis | `data/reports/eval_plan.json` | отчёт с примерами ошибок |
| P5 | Plan-conditioned extractor | adapter `qwen05b-plan-extract-v1` | fact F1 ≥ 0.70 |
| P6 | Runtime + repair buffer | `scripts/18_serve_plan.py` | 50 задач, success ≥ 70% |

Обучение на P3/P5 запускается только после отдельного разрешения.

---

## 10. Планируемые файлы и команды

```
src/web2spec/plan.py             CollectionPlan, FactSet, validate_plan
src/web2spec/tokens.py           build_plan_prompt, build_plan_extract_prompt
scripts/15_make_plan_tasks.py    задача + страница → tasks.jsonl
scripts/16_distill_plans.py      teacher → plans.jsonl, cache, retries
scripts/17_eval_plan.py          plan metrics + end-to-end
configs/data_plan.yaml           пути, split, seq 1536
configs/train_plan_4gb.yaml      QLoRA-пресет planner
configs/eval_plan.yaml           model_dir, data, report
tests/test_plan.py               схема, prompt, метрики, negative
docs/PLANNER_PLAN.md             этот документ
```

Команды (после реализации, не сейчас):

```powershell
python scripts\15_make_plan_tasks.py --pages data\processed\train.jsonl --out data\plan\tasks.jsonl
python scripts\16_distill_plans.py --tasks data\plan\tasks.jsonl --out data\plan\plans.jsonl
python scripts\02_train.py --config configs\train_plan_4gb.yaml --smoke
python scripts\02_train.py --config configs\train_plan_4gb.yaml
python scripts\17_eval_plan.py --config configs\eval_plan.yaml
```

---

## 11. Риски и rollback

| Риск | Признак | Действие |
|---|---|---|
| 0.5B не обобщает | val F1 < 0.6 | не внедрять, оставить teacher planner |
| Шум teacher | противоречия между моделями > 15% | второй teacher, ручной аудит |
| План слишком абстрактный | extractor не находит поля | добавить `hints`/`kind`, пересмотреть схему |
| Утечка | высокий test, низкий unseen | split по хостам и интентам |
| Переобучение на synthetic | real F1 падает | ограничить synthetic 20–30% |
| OOM | падение на seq 1536 | seq 1024, r=8, smoke |

Rollback: planner и extractor — отдельные adapter'ы и отдельные папки. Текущие `models/merged/qwen2.5-0.5b-ft` и GGUF не изменяются. При провале удаляем только `outputs/runs/qwen05b-plan-v1` и `qwen05b-plan-extract-v1`, production не затрагивается.

---

## 12. Что не делаем в этом шаге

- не генерируем код скрапера;
- не обучаем на библиотеках/фреймворках;
- не переходим на 1.5B;
- не запускаем обучение без разрешения;
- не обещаем автономность: fetcher, пагинация и антибот остаются детерминированным кодом.
