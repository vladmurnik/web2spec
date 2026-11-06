# web2spec

**English** · [Русский](README.ru.md)

[![CI](https://github.com/vladmurnik/web2spec/actions/workflows/ci.yml/badge.svg)](https://github.com/vladmurnik/web2spec/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Local, evidence-first web research for API models.

A large model writes only **what it needs to find**. The pipeline runs on your
machine: a local analyst turns the goal into search queries, finds and reads
pages, and returns compact, quote-backed facts. The caller never receives HTML,
never reads dozens of links, and never pays for a search API.

```text
task -> local search -> pages -> local models -> facts with exact evidence
```

The reference machine is a laptop with **4 GB VRAM** (RTX 3050 Ti). The runtime
uses GGUF models through `llama-server` and keeps exactly one model resident at
a time.

## Why this exists

- **Token efficiency.** A research task can cost hundreds of thousands of
  tokens if the caller reads raw pages. web2spec returns a small JSON document
  instead of HTML and page text.
- **Local search.** SearXNG or DuckDuckGo HTML, no paid API keys, no query
  metering.
- **Evidence or nothing.** Every returned fact must be an exact substring of
  the compressed page. Unverified values are moved to `missing` or `conflicts`;
  the pipeline does not fill gaps from model memory.
- **Small GPU.** The fast `collect` path uses two 0.5B models (~380 MB each).
  The `research` path uses a 4B model (~2.4 GB) and unloads it before the
  collector starts.

## Measured results

These are proxy measurements, not a general benchmark. The collector was
evaluated on **50 held-out task/page pairs from 18 article/Habr URLs only**.
No measured results exist yet for job, product, real-estate, tariff or other
page types.

| Path | Task success | Fact F1 | Evidence | Wrong values after repair | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Collect, Q4_K_M | **100%** (50/50) | **1.0000** | exact substring | **0** | **101 s / 50 pages** (~2 s/page) |
| Collect, PyTorch fp16 | 94% (47/50) | 0.9388 | exact substring | 0 | ~11.7 s/page |
| Planner (test) | — | field F1 **0.8220** | — | — | — |
| Extractor v2 (test) | — | fact F1 **0.9357** | 0.9524 evidence rate | — | — |

The Q4 collector result is high because the test slice is narrow: 18 unique
pages, one host, one entity type. It is useful as a regression check, not as a
claim that the model solves arbitrary websites.

`research` was smoke-tested offline on 20 Habr fixtures with Qwen3-4B Q4_K_M:
20/20 pages processed, 37 quote-backed notes, 20 entities, 35 gaps, 10 honest
note omissions (mostly missing authors), 206 s wall time, 24.7 tok/s generation,
7.0 s model load. Those fixtures are metadata-heavy, so this validates the loop
and the output contract, not multi-domain answer quality.

Full methodology, hardware and caveats: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
and the original Russian lab notes in [`docs/STATS.md`](docs/STATS.md).

## Two paths, one contract

### `research` — vague task, unknown sources

For questions such as “compare the current speech-to-text tariffs of three
vendors” where neither the fields nor the URLs are known in advance.

```text
goal
  -> ResearchBrief: entities, questions, source hints
  -> 5-8 queries -> local metasearch -> dedupe / rank / domain filter
  -> 20-50 pages -> semantic chunks (headings, paragraphs, table rows)
  -> 4B notes: entity / field / value / verbatim quote / URL
  -> gap analysis -> new queries (up to 2 rounds)
  -> deterministic aggregation: facts / conflicts / missing
  -> final answer from the 4B model, grounded only in the notes
```

The analyst is **Qwen3-4B-Instruct-2507 Q4_K_M**. It runs for the whole
research call, then exits before any 0.5B model is loaded.

### `collect` — known URLs, fast path

For “extract title, author and publication date from these five official
pages.”

```text
URLs -> polite fetch + robots.txt -> compressed HTML + metadata header
     -> planner 0.5B -> CollectionPlan (entity + fields)
     -> extractor 0.5B -> FactSet (name / value / evidence)
     -> deterministic repair -> strict validation -> facts / missing
```

`collect` does not crawl a site. It processes exactly the URLs it is given.

### Evidence invariant

A fact is accepted only when its `evidence` is an exact substring of the
compressed page. `repair_factset` tries to re-anchor evidence to a nearby exact
fragment; if it cannot, the field becomes `missing`. Facts outside the plan are
dropped. This is why the output can be used without trusting the model's
memory.

## Quick start

### Requirements

- Python **3.12**
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) with `llama-server` on
  `PATH` (or set `WEB2SPEC_LLAMA_SERVER`)
- ~3.3 GB of disk for the three GGUF models
- A GPU with ~4 GB VRAM is the tested target; CPU works but is much slower
- Windows is the reference platform; Linux and macOS use the same Python code

### Install

```powershell
git clone https://github.com/vladmurnik/web2spec.git
cd web2spec
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
web2spec download-models --only all
web2spec doctor
```

On Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
web2spec download-models --only all
web2spec doctor
```

Install with `-e` (editable). The CLI resolves `scripts/` and `models/` relative
to the repository checkout, so this is a repository-first project rather than a
library-only wheel.

`download-models` verifies SHA-256 hashes from
[`models/checksums.sha256`](models/checksums.sha256). The two 0.5B collector
models are attached to the GitHub release; the 4B analyst comes from the
official Qwen GGUF repository. Model files are not stored in git.

Expected `doctor` output:

```text
[OK  ] analyst Q4_K_M: .../models/gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf (2381.6 MB)
[OK  ] planner Q4_K_M: .../models/gguf/qwen05b-plan-Q4_K_M.gguf (379.4 MB)
[OK  ] extractor Q4_K_M: .../models/gguf/qwen05b-plan-extract-v2-Q4_K_M.gguf (379.4 MB)
[OK  ] llama-server: .../llama-server
[OK  ] Python: 3.12.x
```

### First run: known URLs

```powershell
web2spec collect `
  --task "Collect article titles, authors and publication dates" `
  --url https://habr.com/ru/articles/1103276/ `
  --out data/reports/demo.json
```

### First run: research

```powershell
web2spec research `
  --task "Compare Yandex SpeechKit, Sber SaluteSpeech and VK Cloud speech-to-text tariffs in 2026: price per hour, free tier, Russian language and diarization. Use official pages and report missing fields honestly." `
  --domains developers.sber.ru,cloud.vk.com `
  --min-pages 20 --max-pages 30 `
  --out data/reports/research.json
```

One research call typically takes 2–4 minutes on the reference laptop. It
writes a JSON report with `answer`, `entities`, `sources`, `gaps`, `errors`,
`stats` and `speed`; `--include-notes` is available for audits.

### MCP server

`web2spec mcp` starts a stdio MCP server. Any MCP-capable client can call two
tools:

| Tool | Use it when | Main output |
| --- | --- | --- |
| `research` | The task is vague and sources/fields are unknown | `answer`, `entities`, `sources`, `gaps`, `errors` |
| `collect` | You already have official URLs | aggregated `facts`, `conflicts`, `missing`, `by_source` |

Copy [`configs/mcp.example.json`](configs/mcp.example.json) into the client
configuration and replace the paths. Claude Desktop stores it in
`claude_desktop_config.json`; Cursor uses `~/.cursor/mcp.json`.

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

`WEB2SPEC_LLM_MODEL` is optional and only affects the old `search` path; the
MCP `research` path uses the local 4B analyst. Do not run `research` and
`collect` concurrently on a 4 GB GPU. The server writes diagnostics to stderr;
stdout is protocol-clean JSON-RPC.

The response is compact JSON, never HTML. A typical `collect` result:

```json
{
  "task": "Compare speech-to-text tariffs",
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

When sources disagree, both values are returned. The pipeline does not pick a
winner.

## CLI

```powershell
web2spec doctor
web2spec download-models --only collector
web2spec collect --task "..." --urls data/urls.txt
web2spec search  --task "..." --provider ddg --search-only
web2spec research --task "..." --domains vendor-x.ru --min-pages 20
web2spec mcp
web2spec eval-q4 --limit 50
```

| Command | Purpose |
| --- | --- |
| `download-models` | fetch GGUF models and verify checksums |
| `doctor` | check models, `llama-server` and Python |
| `collect` | task + URLs -> structured facts |
| `search` | task -> queries -> local search -> collector |
| `research` | task -> 20–50 pages -> evidence report |
| `mcp` | MCP stdio server |
| `eval-q4` | Q4_K_M quality/speed evaluation |
| `eval-runtime` | PyTorch runtime cascade evaluation |
| `eval-plan`, `eval-plan-extract` | planner/extractor evaluation |
| `train`, `merge` | QLoRA training and adapter merge |

Every numbered script in [`scripts/`](scripts/) remains runnable on its own.
See [`scripts/README.md`](scripts/README.md) for the full map.

## Search providers

- **SearXNG JSON** (recommended): self-hosted, aggregates public engines, no
  API key. Enable `formats: [html, json]` in `settings.yml` and pass
  `--provider searxng --searxng-url http://localhost:8080`.
- **DuckDuckGo HTML** (fallback): no installation, but public endpoint and
  best-effort; it can rate-limit or return a captcha.

`search` can use an optional local LLM for query generation (Ollama or any
OpenAI-compatible server). Without it, the task text is used as the query.
`research` does not need Ollama: its 4B analyst writes the queries.

## Models

| Role | File | Size | Source |
| --- | --- | ---: | --- |
| Analyst (`research`) | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2381.6 MB | Qwen official GGUF |
| Planner (`collect`) | `qwen05b-plan-Q4_K_M.gguf` | 379.4 MB | fine-tuned Qwen2.5-0.5B |
| Extractor (`collect`) | `qwen05b-plan-extract-v2-Q4_K_M.gguf` | 379.4 MB | fine-tuned Qwen2.5-0.5B |

Base models are Apache-2.0. The fine-tuned GGUFs are released with this
project; see [`docs/MODELS.md`](docs/MODELS.md) for checksums, rebuild steps and
the quantization caveat. Rebuilding the collector requires the training data,
which is not committed.

The PyTorch fallback (`--backend hf`) requires the `hf` extra:

```powershell
pip install -e ".[hf]"
web2spec collect --task "..." --urls data/urls.txt `
  --backend hf `
  --planner models/merged/qwen05b-plan `
  --extractor models/merged/qwen05b-plan-extract-v2
```

Training and full evaluation use `.[train]`:

```powershell
pip install -e ".[train]"
web2spec train --config configs/train_plan_extract_reliable_4gb.yaml
```

## Output contract

`collect` writes one JSON document:

```json
{
  "task": "Collect article titles and authors",
  "generated_at": "2026-11-06T12:00:00+00:00",
  "pages": [
    {
      "url": "https://example.com/article/1",
      "status": "ok",
      "entity": "article",
      "facts": [
        { "name": "title", "value": "How to speed up data collection", "evidence": "How to speed up data collection" }
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
      "facts": { "title": "How to speed up data collection" },
      "missing": ["published_at"],
      "error": null
    }
  },
  "stats": { "requested": 1, "ok": 1, "failed": 0, "by_status": { "ok": 1 } }
}
```

`status` is one of `ok`, `fetch_error`, `plan_error`, `extract_error`.
`missing` is not a failure of the run; it is the honest answer that the page
does not contain a verifiable value. Exit code `0` means at least one page is
`ok`, `1` means all pages failed, `2` means invalid arguments.

`research` adds `goal`, `questions`, `entities` (`facts`, `conflicts`,
`evidence`), `notes`, `gaps`, `sources`, `errors`, `stats` and `speed`.

## Repository layout

```text
src/web2spec/       library: schemas, HTML compression, research loop, CLI
scripts/            numbered pipeline: data -> training -> eval -> runtime -> MCP
configs/            YAML configs, MCP example, seeds
tests/              pytest suite; no GPU or network required
docs/               architecture, benchmarks, model notes, lab history
examples/           small task/URL examples
data/               datasets and reports (gitignored)
models/             GGUF weights (gitignored) and checksum manifest
outputs/runs/       adapters, checkpoints, training logs (gitignored)
```

Key modules:

- `web2spec/html_compress.py` — HTML compression and metadata extraction;
- `web2spec/semantic.py` — semantic chunks: headings, paragraphs, table rows;
- `web2spec/research.py` — question-first loop, notes, gaps, aggregation;
- `web2spec/plan.py` — `CollectionPlan`, `FactSet`, `repair_factset`, validation;
- `web2spec/llama_runtime.py` — `llama-server` lifecycle and Q4 defaults;
- `web2spec/tokens.py` — ChatML prompts without a tokenizer dependency;
- `web2spec/cli.py` — unified `web2spec` entry point.

## Development

```powershell
pip install -e ".[dev]"
python -m ruff check .
python -m pytest
```

The suite is CPU/network-free and covers schemas, compression, plan repair,
research aggregation, search parsing, the MCP protocol and the CLI. CI runs the
same commands on Ubuntu and Windows with Python 3.12.

Training data and reports are not committed. The original experiments, plans
and metrics are preserved in `docs/` and the numbered scripts. Read
[`CONTRIBUTING.md`](CONTRIBUTING.md) before changing the evidence contract or
adding a model path.

## Limitations

- **Narrow quality evidence.** Collector quality is measured only on
  article/Habr pages; research only on Habr fixtures. There is no multi-domain
  benchmark yet. Do not present the numbers as general accuracy.
- **No browser.** The fetcher does not execute JavaScript, wait for SPAs, solve
  captchas, bypass WAFs, log in, or crawl pagination. A blocked page is reported
  as `blocked`/`fetch_error`, not silently skipped.
- **Best-effort search.** Public metasearch endpoints can rate-limit. SearXNG is
  recommended for repeatable use.
- **Untrusted page text.** Fetched pages may contain prompt injection. The
  pipeline returns data, never instructions or tool calls, and requires exact
  evidence, but a malicious page can still bias what the model chooses to note.
  Treat outputs as untrusted data and verify high-stakes numbers at the source.
- **One model at a time.** On 4 GB VRAM, running the 4B analyst and the 0.5B
  collector together will exhaust memory. The CLI enforces sequencing by
  starting and stopping `llama-server` around each stage.
- **Not a crawler.** `collect` processes exactly the supplied URLs. It does not
  discover a site's full catalogue or deduplicate entities across pages.
- **Model weights are separate.** A clean clone contains no `.gguf` files. Run
  `web2spec download-models` or build them from the training pipeline.

## License

MIT — see [`LICENSE`](LICENSE). The base Qwen models are Apache-2.0; model
licenses and redistribution notes are in [`docs/MODELS.md`](docs/MODELS.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Datasets and scraped pages
are not redistributed.
