# web2spec — общий план развития

**Статус:** 2026-11-06. Документ фиксирует текущее состояние и оба трека: уже обученный extraction и task-conditioned planner + plan-conditioned extractor. Детальный план следующего шага — в [`PLANNER_PLAN.md`](PLANNER_PLAN.md). Исторические документы [`PLAN.md`](PLAN.md) и [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) не переписаны, но их решения по 0.5B, 4 GB и QLoRA остаются в силе.

---

## 1. Что уже сделано

| Компонент | Состояние | Артефакт |
|---|---|---|
| Сбор реальных страниц по JSON-LD | готово | `scripts/08_collect.py`, 377 реальных примеров |
| Синтетика для extraction | готово | `scripts/09_make_synthetic.py`, 500 примеров |
| Валидация, дедуп, splits | готово | 866 принято: train 744 / val 61 / test 61 |
| Extraction QLoRA | обучено | adapter 33.6 MB, merged 942 MB |
| Extraction GGUF | готово | Q4_K_M 379.4 MB, llama-server проверен |
| Extraction eval | готово | JSON 98.4%, exact 42.6%, micro F1 84.3% |
| Spec-трек, синтетика | обучено, но не переносится | adapter/merged есть, real F1 0 |
| Spec-трек, реальные данные | данные готовы, обучение остановлено | 157 примеров, checkpoint-20 |
| Planner / task-conditioned extraction | готово | adapter v1 + extractor v2, test fact F1 0.9357 |
| Plan-conditioned extractor v2 | готово | reliable-label датасет 1942, merged 942 MB |
| Runtime-каскад / repair buffer | готово, Q4_K_M проверен | P6, `scripts/24_eval_q4km.py` |

### Честная оценка extraction

Модель — **schema-bound extractor**, а не универсальный скрапер. На знакомых шаблонах работает хорошо, на незнакомых сайтах теряет поля:

- Regard: F1 0.80;
- Lenta: 0.67;
- hh.ru: 0.39;
- Habr: 0.17 (3 примера).

Синтетика завышает средние метрики. Модель не загружает страницы, не обходит антибот и не строит план обхода.

---

## 2. Цель проекта

Научить модель **по задаче пользователя понимать, какие данные нужны**, а не заполнять один заранее заданный набор полей.

Целевая архитектура:

```
задача пользователя
        │
        ▼
   planner (0.5B)  ──►  CollectionPlan: сущность, поля, фильтры, пагинация
        │
        ▼
   fetcher / navigator (детерминированный код)
        │
        ▼
   plan-conditioned extractor (0.5B)  ──►  facts + evidence
        │
        ▼
   validator ──► при провале: teacher / repair
```

### Non-goals v1

- автономный обход любых сайтов и антибот-защит;
- генерация произвольного Python-кода моделью 0.5B;
- обучение на 1.5B+ и full fine-tune;
- замена deterministic fetcher'а моделью;
- запуск обучения без отдельного разрешения.

---

## 3. Два трека

### Track A — extraction (готов)

HTML → JSON по фиксированной схеме. Остаётся production-компонентом для известных типов страниц. Не трогаем без причины: adapter, merged и GGUF уже проверены.

### Track B — task-conditioned planner (готов)

Задача + сэмпл сайта → `CollectionPlan`. План определяет, **что** собирать. Extraction выполняется отдельно, уже по плану. Это снимает главное ограничение текущей модели: она не может решить, что важно, потому что её учили только фиксированным полям. Принят checkpoint-50: val field F1 0.8220, test 136 (unseen Habr) 0.7714.

### Track C — plan-conditioned extractor (готов)

Тот же HTML, но target — facts с именами из плана, а не глобальная схема. Нужен, чтобы план действительно управлял извлечением, а не оставался текстом. v2 на reliable-label данных: test 90 fact F1 0.9357, evidence rate 0.9524, missing accuracy 0.9118. Ограничение — test только article/Habr.

### Track D — runtime (готов, базовый прокси)

Каскад: planner → extractor → validator → teacher fallback. Ошибки сохраняются в буфер для дообучения. Базовый прогон на 50 задачах: task success 94% (47/50), fact F1 0.9388, 0 неверных значений; 48 фактов ушли в repair buffer. Teacher fallback ещё не подключён, ручного аудита не было — success измерен детерминированным прокси.

### Track E — code generation (заморожен)

Генерация скраперов на библиотеках. Для 0.5B признана малоперспективной: высокий риск выдуманных API. Возврат к теме — только после B/C и только с execution-eval и, вероятно, базой 1.5B Coder.

---

## 4. Дорожная карта

| Этап | Содержание | Критерий готовности | Статус |
|---|---|---|---|
| M0 | Схема `CollectionPlan` / `Fact`, тесты | схема валидирует примеры, есть negative-кейсы | выполнено |
| M1 | Банк задач и teacher-разметка 500 планов | 500 планов, 0 ошибок схемы, ручной аудит 50 | выполнено |
| M2 | Расширение до 2–3 тыс. планов | покрыты 5 категорий, 3–5 задач на страницу | выполнено |
| M3 | Обучение planner-адаптера | val plan F1 ≥ 0.75, entity ≥ 0.85 | выполнено: F1 0.8220 |
| M4 | Eval на unseen hosts/tasks | end-to-end fact F1 ≥ 0.60 | частично: test F1 0.7714/0.9357 на Habr, другие хосты не покрыты |
| M5 | Plan-conditioned extractor | F1 ≥ 0.70 на своих хостах | выполнено: fact F1 0.9357 |
| M6 | Runtime-каскад + repair buffer | 50 задач, success ≥ 70% | выполнено: 94% (прокси), F1 0.9388 |
| M7 | Опционально: 1.5B Coder / code track | pass@1 на execution-фикстурах | заморожено |
| M8 | MCP-сервер: внешняя модель → задача → факты | stdio, `collect`/`search_links`, живой smoke, изоляция stdout | выполнено: 13 тестов, реальный `collect` на Habr |
| M9 | Q4_K_M GGUF для planner/extractor | те же 50 задач, quality/speed отчёт | выполнено: 100% success, F1 1.0, ~161–172 ток/с; ограничения в `STATS.md` |
| M10 | Q4_K_M как runtime по умолчанию + единый CLI | `web2spec collect/search/mcp/doctor`, тесты, чистая структура | выполнено: GGUF по умолчанию, 53 теста, README/scripts index |

M8 не меняет качество извлечения: это транспорт и упаковка. Большая модель
вызывает `collect(task=...)`, сервер локально делает поиск, загрузку и извлечение,
а наружу отдаёт только компактный JSON без HTML. Если ссылки уже есть, они
передаются в `urls`, и поиск пропускается.

M9 — это тот же runtime-каскад, но planner и extractor запускаются как Q4_K_M
GGUF через `llama-server`. На 50 held-out страницах: task success 100%,
fact F1 1.0, 0 неверных значений, 101 с на 50 страниц (~2 с на страницу).
Это узкий article/Habr-набор: 60 дат ушли в `missing`, 51 поле плана оказалось
вне gold-плана; перенос на другие хосты не измерен.

M10 закрепляет Q4_K_M как путь по умолчанию для `21_collect_urls.py`,
`22_search_and_collect.py` и MCP-сервера. Появился общий runtime
`src/web2spec/llama_runtime.py` (поиск `llama-server`, запуск сервера,
batch-генерация, Q4-пути) и единый CLI `web2spec` с командами
`collect`, `search`, `mcp`, `doctor`, `eval-*`, `train`, `merge`. PyTorch
float16 остаётся fallback через `--backend hf`; merged-каталоги нужны для
пересборки GGUF и обучения.

Каждый этап заканчивается отчётом и commit'ом. Обучение запускается только после явного разрешения.

---

## 5. Метрики

### Planner

- **entity accuracy** — правильно ли определена сущность;
- **field precision / recall / F1** — совпали ли поля с эталонным планом;
- **required-field recall** — не потеряны ли обязательные поля;
- **scope accuracy** — нужны ли пагинация и detail pages;
- **empty-plan accuracy** — умеет ли модель сказать «здесь нечего собирать».

### End-to-end

- **task success** — решила ли система задачу пользователя;
- **fact F1** — совпали ли значения;
- **evidence rate** — доля фактов, подтверждённых точной подстрокой;
- **pages crawled** — сколько карточек реально собрано;
- **repair rate** — сколько раз понадобился teacher/повтор.

### Anti-cheat

- split по хостам, а не случайный;
- отдельный срез unseen tasks;
- запрет синтетики в test;
- ручная проверка 50 примеров перед публикацией цифр.

---

## 6. Ограничения 4 GB

- база: Qwen2.5-0.5B-Instruct, 4-bit NF4, seq 1536, r=16;
- batch 1, grad accumulation 16, gradient checkpointing;
- один adapter обучается за раз; planner и extractor — отдельные adapter'ы;
- inference: Q4_K_M ~0.38 GB, сервер с `-np 2..4` ~1.2–1.5 GB VRAM;
- 1.5B Coder на 4 GB — только seq 768–1024, r=8, с риском OOM; не в v1.

---

## 7. Риски

| Риск | Вероятность | Смягчение |
|---|---|---|
| 0.5B не обобщает произвольные задачи | высокая | ограничить таксономию, teacher fallback |
| Шум teacher-разметки | средняя | verifier + ручной аудит + второй teacher |
| План и extraction расходятся | средняя | общий schema-контракт, end-to-end eval |
| Утечка задач в test | средняя | split по хостам и шаблонам задач |
| Переобучение на синтетику | высокая | реальные страницы в train, synthetic только как дополнение |
| 4 GB OOM | средняя | seq 1536, r16, checkpointing, smoke до полного запуска |
| Юридические ограничения источников | средняя | только публичные страницы, robots.txt, без авторизации |

---

## 8. Принятые решения

1. Не учить 0.5B генерировать свободный код скрапера в v1.
2. Разделить «понять задачу» (planner) и «извлечь данные» (extractor).
3. План — семантический: имена полей и причины, без CSS-селекторов. Селекторы — дело extractor'а.
4. Факты подтверждаются `evidence` — точной подстрокой HTML.
5. Текущий extraction-адаптер не перезаписывать; новые треки — отдельные adapter'ы.
6. Любое обучение — только после явного разрешения.

---

## 9. Ссылки

- [`PLANNER_PLAN.md`](PLANNER_PLAN.md) — детальный план следующего шага.
- [`SPEC_TRACK.md`](SPEC_TRACK.md) — что уже пробовали со spec и почему остановились.
- [`STATS.md`](STATS.md) — статистика данных и артефактов.
- [`DATASET.md`](DATASET.md) — состав датасета и ограничения.
- [`PLAN.md`](PLAN.md) — исходная стратегия.
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — исходный план реализации.
