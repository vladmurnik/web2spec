# Real-page benchmark

This benchmark checks the runtime contract on pages that people actually ask
research agents to read: documentation, API references, pricing, and product
specifications. It is deliberately separate from the synthetic and Habr-only
article evaluations in `docs/BENCHMARKS.md`.

## What is measured

Each line in `cases.jsonl` is one live page with:

- a task that names the expected fields;
- the URL;
- the expected values read from that page on `checked_at`.

`web2spec benchmark` runs the real collector for each case. It passes the
expected field names and kinds as `--field`, so it measures the extractor,
evidence repair and strict validation against a fixed caller schema; planner
quality is covered separately by `eval-plan`. It reports:

- exact cases: every expected field matched;
- field accuracy: matched fields / expected fields;
- wrong and missing fields with the actual model value.

It never counts a value without an exact supporting quote as matched.

## Run

```powershell
web2spec benchmark
web2spec benchmark --limit 2 --no-cache
web2spec benchmark --out data/reports/benchmark.json
```

The command needs the same local models as `collect`; it does not call a paid
API. Each case starts the models separately, so a full run is slower than one
multi-URL collection.

## Current coverage

| Category | Cases |
|---|---|
| Documentation | GitHub Actions limits, Python asyncio |
| API reference | Stripe errors |
| Pricing | Anthropic, Supabase, Vercel |
| Product specs | Apple MacBook Air |

Jobs and real-estate pages are not included yet: their public pages are often
dynamic, blocked, or change too quickly for a small dated gold set. Add a case
only after reading the page and recording the exact value and quote context.

## Updating gold values

1. Open the URL and verify the value manually.
2. Update `fields` and `checked_at` in `cases.jsonl`.
3. Run the case with `--limit` while developing.
4. Record the measured result in `docs/BENCHMARKS.md`; do not copy a result from
   an older run to a newer page snapshot.

Pages can change or block automated fetches. A failed fetch is reported as
`collector_error` or `timeout`, never as a quality win.
