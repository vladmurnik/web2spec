# Датасет web2spec

Документ фиксирует, что именно лежит в `data/raw`, как это собрано и какие ограничения
нужно учитывать до запуска обучения. Все цифры — на 4 ноября 2026 года.

## Состав

| Источник | Хост / генератор | Записей | Тип | Статус |
|---|---|---:|---|---|
| Статьи | lenta.ru (RSS) | 172 | `article` | реальные страницы |
| Статьи | habr.com (RSS) | 40 | `article` | реальные страницы |
| Вакансии | kaluga.hh.ru (sitemap) | 100 | `job` | реальные страницы |
| Товары | www.regard.ru | 60 | `product` | реальные страницы |
| Товары | www.citilink.ru | 5 | `product` | реальные страницы |
| Синтетика | `scripts/09_make_synthetic.py` | 500 | все 5 типов | шаблонные страницы |
| **Итого** | | **877** | | принято 866 |

Синтетическая часть сбалансирована: по 100 записей на каждый тип, внутри — 60 русских и
40 английских. Реальные данные неравномерны: статьи и вакансии представлены хорошо,
товаров меньше, отдельного реального источника для `listing` и `realestate` пока нет.

## Откуда взяты реальные страницы

Сборщик `scripts/08_collect.py` ходит только на публичные HTML-страницы, соблюдает
`robots.txt`, не использует авторизацию, cookies и обход антибот-защиты. Целевой JSON
строится не вручную и не учителем, а из schema.org JSON-LD, который сайт уже публикует
на странице. JSON-LD вырезается из `input_html`, поэтому модель учится читать видимый
HTML, а не копировать метаданные.

| Источник | Как получены URL | Что даёт |
|---|---|---|
| lenta.ru | `https://lenta.ru/rss` | `NewsArticle`: title, author, published_at, modified_at, section, description |
| habr.com | `https://habr.com/ru/rss/articles/` | `Article`: title, published_at, modified_at, description (без автора) |
| hh.ru | `https://hh.ru/sitemap/vacancy0.xml` | `JobPosting`: title, company, location, salary, posted_at |
| regard.ru | sitemap + категории | `Product`: title, price, currency, availability, brand |
| citilink.ru | `sitemap/main/products_10.xml` | `Product`: title, price, currency, availability, brand, sku, rating |

Что проверено и не подошло: Avito (403), Domclick (401), Cian (SPA без sitemap),
IRR (DNS), Etagi (мёртвые sitemap), Novostroy (нет sitemap), Rabota.ru (нет JSON-LD),
Mvideo/Technopark/Petrovich (401 или antibot), Youla (пустой sitemap, JS-оболочка).
Это не ошибка пайплайна, а сознательный отказ от обхода защиты.

## Формат записи

Каждая строка — объект `Sample` (`src/web2spec/schema.py`):

```json
{
  "id": "crawl-...",
  "group_id": "lenta.ru",
  "task": "extract",
  "instruction": "Извлеки из HTML статьи заголовок, автора, дату публикации и раздел в JSON.",
  "input_html": "<main>...</main>",
  "target": {"title": "...", "author": "...", "published_at": "..."},
  "meta": {
    "lang": "ru",
    "complexity": 1,
    "source_url": "https://...",
    "n_tokens": null,
    "pipeline_version": "collect-0.1"
  }
}
```

`group_id` — хост или синтетический генератор. Сплит group-aware: все записи одного
хоста попадают в один из наборов, чтобы модель не видела почти одинаковые страницы
одного сайта в train и val.

## Бюджет токенов

Реальные страницы после сжатия всё равно длинные. Замеры на Qwen-токенизаторе показали,
что 12 000 символов дают 3 500–4 100 токенов и не влезают в 4 GB-режим. Поэтому:

- `compress_max_chars = 3500` — бюджет видимого HTML на один пример;
- `max_seq_len = 1536` — потолок промпта вместе с инструкцией и целевым JSON;
- для вакансий hh.ru используется отдельный бюджет 2400 символов: их текст плотнее.

После валидации полный датасет даёт p50 ≈ 1 100 токенов, p95 ≈ 1 500, max ≤ 1 535.
Это оставляет запас под целевой JSON и не раздувает активации на 4 GB.

## Контроль качества

`scripts/01_validate.py` для каждой строки проверяет:

- JSON и Pydantic-схему `Sample` (лишние поля запрещены);
- длину `input_html` и наличие видимого заголовка/цены;
- целевой JSON по схеме задачи (`configs/tasks.yaml`);
- длину промпта в токенах относительно `max_seq_len`;
- дубликаты по хэшу;
- карантин: отклонённые строки не удаляются, а пишутся в `data/raw/quarantine.jsonl`
  с кодом причины (`E_TOO_LONG`, `E_NO_PRICE`, `E_DUP`, ...).

Синтетика дополнительно проверяется тестами: якорные значения (title/price) обязаны
оставаться в сжатом HTML, JSON-LD и `<script>` не должны просачиваться в `input_html`,
генерация с seed 42 воспроизводится побайтово.

## Ограничения

1. **Доменов мало.** Реальные товары — только Regard и Citilink, вакансии — только hh.ru,
   статьи — Lenta и Habr. Модель может выучить вёрстку этих сайтов, а не общий навык.
2. **Синтетика шаблонна.** 500 примеров полезны для прогрева и проверки пайплайна, но
   не заменяют реальные страницы. Оценивать качество только на синтетике нельзя.
3. **Test-набор стратифицирован, но не по хостам.** После валидации: train 744,
   val 61, test 61 (`scripts/10_split_dataset.py`). Все хосты есть во всех трёх частях,
   поэтому это оценка на новых страницах, а не на новом сайте. Для проверки переноса
   нужен отдельный host-held-out замер (например, обучаться на Lenta, проверять на Habr).
4. **Разметка из JSON-LD не идеальна.** Цена/наличие могут расходиться с видимой
   карточкой; часть страниц отбраковывается, если цена не опубликована.
5. **Данные меняются.** Sitemap и вёрстка сайтов живут своей жизнью; повторный сбор
   может дать другой набор. Версия пайплайна фиксируется в `meta.pipeline_version`.

## Как расширить

```powershell
# товары: медленно, иначе Citilink отдаёт 403
python scripts/08_collect.py --category products `
  --sitemap https://www.citilink.ru/sitemap/main/products_10.xml `
  --out data/raw/products --max-total 100 --depth 0 --delay 3.0 --max-chars 3500

# статьи: каждый фид отдельно, иначе max_total съедает один хост
python scripts/08_collect.py --category articles --seeds configs/seeds/lenta.txt `
  --out data/raw/articles --max-total 120 --depth 1 --max-per-seed 250

# вакансии: 2400 символов, чтобы влезть в 1536 токенов
python scripts/08_collect.py --category listings `
  --sitemap https://hh.ru/sitemap/vacancy0.xml `
  --out data/raw/listings --max-total 100 --depth 0 --delay 1.5 --max-chars 2400

# синтетика
python scripts/09_make_synthetic.py --out data/raw/synthetic --count 500

# пересобрать train/val и отчёт
python scripts/01_validate.py --input data/raw --output data/processed
```

Перед обучением повторно запустить `01_validate.py` и проверить
`data/reports/validation_full.json`: `failed` должен быть близок к нулю, а p95 — ниже
`max_seq_len`.
