---
name: Web2Spec Research
description: "Use when a task needs current web data, tariff/price comparison, or source-backed facts without reading HTML or spending paid search API tokens. Teaches when to call web2spec research vs collect, how to phrase the task, and how to report facts, conflicts, and gaps."
summary: "web2spec MCP/CLI workflow: research for vague tasks (20-50 pages), collect for known URLs, exact evidence, honest conflicts/missing, 4GB VRAM rules."
---

# Web2Spec Research

## What this tool is for

`web2spec` is a local pipeline:

```text
task -> local search -> pages -> local models -> facts with exact quotes
```

The calling model writes only the goal and constraints. It never receives HTML and
never needs to open dozens of links manually. The local side runs a 4B analyst for
broad research or two 0.5B models for known URLs.

Use it when the answer must be current, comparable across sources, or backed by
quotations. Do not use it for general knowledge you already know reliably.

## Choose one path

| Situation | Tool | Why |
|---|---|---|
| Vague question, market scan, several vendors, unknown sources | `web2spec_research` | The 4B model clarifies questions, searches, reads 20-50 pages, finds gaps |
| You already have 1-8 official URLs | `web2spec_collect` | Skips search, extracts only requested facts, fastest |
| You need only candidate links | `web2spec search` CLI | Rare; do not use it as a first step for a research task |
| You need a repeatable report from a fixed domain | `web2spec_research` with `domains` | Official-domain scoping plus gap rounds |

Never call `search` repeatedly and then feed links into `collect` one by one. That
is the old, expensive workflow. One `research` call replaces it.

## MCP tools

The configured server is named `web2spec`. Native tool names:

- `web2spec_research`
- `web2spec_collect`

In Code Mode they may appear as `tools.web2spec.research(...)` and
`tools.web2spec.collect(...)`. Use the native names unless the client exposes only
the grouped form.

### `web2spec_research`

Required: `task`.

Useful options:

```json
{
  "task": "Сравни тарифы X, Y, Z: цена за час, бесплатный лимит, русский язык, диаризация. Только официальные сайты, ноябрь 2026.",
  "domains": ["vendor-x.ru", "vendor-y.ru"],
  "min_pages": 20,
  "max_pages": 30,
  "max_rounds": 2,
  "include_notes": false
}
```

- `min_pages` is clamped to 20-50. `max_pages` defaults to 30.
- `urls` skips search and uses only those pages; combine with `domains` only if
  you also want source hints.
- `source_hints` accepts domains or page types, not invented URLs.
- `include_notes` returns every raw note. Use only for an audit; it multiplies
  output tokens. Normally read `entities`, `answer`, `gaps`, `sources`.
- A run can take minutes because it loads a 2.38 GB model and reads many pages.
  Do not launch several research calls in parallel on a 4 GB GPU.

### `web2spec_collect`

Required: `task`.

Useful options:

```json
{
  "task": "Извлеки цену за час, бесплатный лимит и поддержку русского языка.",
  "urls": ["https://vendor.example/pricing"],
  "max_pages": 6,
  "include_evidence": false
}
```

- If `urls` is omitted, it performs a short local search (`max_results` default 6,
  `max_pages` default 6). Prefer passing official URLs when you know them.
- `domains` restricts search results to those hosts.
- `include_evidence=true` adds exact quotes. Use it when a number is disputed or
  the user asks for proof; otherwise keep it false to save tokens.
- `include_pages=true` returns per-page facts. Use it only for debugging or when
  you need per-source attribution that `by_source` does not provide.

## How to phrase the task

Write a compact research brief, not a list of tool calls:

1. Goal: what decision the answer must support.
2. Entities: exact vendor/product names.
3. Fields: price, free tier, language, diarization, date, author, etc.
4. Constraints: official domains, region, date, units.
5. Output: table, short report, or JSON.

Good:

```text
Сравни Yandex SpeechKit, Sber SaluteSpeech и VK Cloud за ноябрь 2026:
цена распознавания за час в рублях, бесплатный лимит, поддержка русского,
диаризация. Только официальные сайты. Если точной цены нет — скажи missing,
не подставляй блог.
```

Bad:

```text
Найди ссылки, открой их, потом найди цены.
```

The second forces the caller to orchestrate what the pipeline already does.

## How to read the result

### `research`

Top-level fields:

- `answer` — prose synthesis based only on grounded notes.
- `entities` — per-entity `facts`, `conflicts`, `evidence`.
- `gaps` — what the pages did not answer.
- `sources` — pages actually processed successfully.
- `errors` — blocked/empty/fetch failures with reasons.
- `stats` — `pages_requested`, `pages_ok`, `pages_shortfall`, `notes`, `rounds`.
- `speed` — load time, tokens, tok/s, wall time.

### `collect`

- `facts` — one value per field when all sources agree.
- `conflicts` — several values; each has `value`, `sources`, optionally `evidence`.
- `missing` — field -> URLs where it was absent.
- `by_source` — URL -> field -> value, when multiple sources disagree.
- `sources`, `errors`, `stats`, `search`.

Rules:

1. Never silently choose one side of a `conflict`. Report both values and sources.
2. Never fill a `missing` field from memory. Say it was not found on the pages.
3. Cite the `sources` or `by_source` URL for every number in the final answer.
4. If `pages_shortfall > 0`, say the report is based on fewer than 20 pages.
5. If `errors` contains blocked/WAF/captcha, do not claim the vendor was checked.
6. Relative dates are valid evidence. Do not convert “6 часов назад” into an ISO
   date unless the page also provides one.

## Recommended workflows

### Price/tariff comparison

1. One `web2spec_research` call with all vendors and their official domains.
2. Read `entities` as a table; resolve units from `evidence` if needed.
3. Report `conflicts` and `missing` explicitly.
4. If a vendor is blocked, use `web2spec_collect` with a known official URL or say
   it could not be verified. Do not scrape around a captcha.

### Known official pages

1. One `web2spec_collect` call with all URLs.
2. Keep `include_evidence=false` unless the user asks for quotes.
3. If a field is absent on all pages, return `missing`, not a blog estimate.

### Follow-up question on an existing report

Do not rerun the whole research. Ask `collect` only for the missing URL/field, or
answer from the existing JSON if it is still in context.

## CLI fallback

If MCP is unavailable or the client cannot call tools, use the same pipeline from
the repository root:

```powershell
.\.venv\Scripts\web2spec.exe doctor
.\.venv\Scripts\web2spec.exe research --task "..." --domains vendor-x.ru --min-pages 20 --max-pages 30 --out data\reports\research.json
.\.venv\Scripts\web2spec.exe collect --task "..." --urls urls.txt --out data\reports\collect.json
```

`doctor` must show the analyst, planner, extractor, and `llama-server` as `[OK]`.
If it does not, report the exact missing item instead of pretending the pipeline ran.

## Hardware and cost rules

- GPU: 4 GB VRAM. The 4B analyst is 2381.6 MB; the 0.5B collector models are
  379.4 MB each. Run one pipeline at a time.
- Do not run Ollama 3B/7B and the local pipeline simultaneously. Stop Ollama first
  if it holds VRAM.
- `research` default 20-30 pages; hard range 20-50. More pages cost time, not just
  tokens. Do not raise the budget without a reason.
- One `research` call is usually 2-4 minutes locally. Warn the user before a long
  run; do not spam retries.
- Search is local (DuckDuckGo HTML or SearXNG). There are no paid search API calls.

## Failure handling

- `blocked` / `empty`: the site used a WAF, captcha, or JavaScript. Do not retry
  the same URL many times.
- `fetch_error`: network, 404, robots.txt. Report the URL and move on.
- `pages_shortfall`: fewer than 20 usable pages. State the real coverage.
- Model/server error: run `web2spec doctor`; check that `llama-server` is on PATH
  or set `WEB2SPEC_LLAMA_SERVER`.
- MCP tool missing: verify the `web2spec` server is connected in `/mcps`; then use
  the CLI fallback.

## Do not

- Do not ask the tool for raw HTML. It never returns HTML by design.
- Do not invent a schema and force it before search; `research` is question-first.
- Do not treat a blog as an official price when the official page is blocked.
- Do not hide conflicts or missing fields to make the answer look complete.
- Do not run multiple 4B research calls in parallel on this GPU.
- Do not edit the project while using it; this skill is for using the tool, not
  developing it.
