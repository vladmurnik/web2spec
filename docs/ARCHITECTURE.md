# Architecture

This document explains how web2spec keeps a large API model away from raw web
pages while still returning auditable facts. It describes the design as of
v0.4.0; the historical planning notes in this directory are not required to
understand the runtime.

## Design constraints

1. **Evidence first.** A fact without an exact quote from the compressed page is
   not a fact. It becomes `missing` or `conflict`.
2. **Small hardware.** The reference GPU has 4 GB of VRAM. One model is resident
   at a time, and the 4B analyst is unloaded before the 0.5B collector starts.
3. **No paid search API.** Search is local metasearch: SearXNG JSON or the
   DuckDuckGo HTML endpoint.
4. **The caller sees only data.** MCP tools return compact JSON. HTML and page
   text never enter the caller's context.
5. **Honest failure.** Blocked pages, short page budgets and absent fields are
   explicit output, not silently repaired with model memory.

## Two execution paths

### Research: unknown sources and unknown fields

```text
task
  |
  v
ResearchBrief (4B): entities, questions, source hints
  |
  v
queries (5-8) --> metasearch --> dedupe / rank / domain filter
  |
  v
20-50 pages --> fetch + compress --> semantic chunks
  |
  v
notes (4B): entity / field / value / verbatim quote / URL
  |
  v
gap analysis --> new queries (up to 2 rounds)
  |
  v
deterministic aggregation --> facts / conflicts / missing
  |
  v
final answer (4B), grounded only in notes
```

The research path is question-first: it does not force a schema before it has
seen the pages. It is implemented in `web2spec/research.py` and exposed by
`web2spec research` and the MCP `research` tool.

### Collect: known URLs

```text
URLs
  |
  v
polite fetch (robots.txt, delay) --> compressed HTML + metadata header
  |
  v
planner (0.5B) --> CollectionPlan: entity + requested fields
  |
  v
extractor (0.5B) --> FactSet: name / value / evidence
  |
  v
repair_factset --> strict validate --> JSON report
```

The collect path is implemented by `scripts/21_collect_urls.py`, with the
schema and repair logic in `web2spec/plan.py`. It is the fast path for official
pricing pages, documentation or a fixed list of articles.

## Components

| Component | Responsibility |
| --- | --- |
| `web2spec/collect.py` | HTTP fetch, gzip, redirects, robots.txt and crawl delay |
| `web2spec/html_compress.py` | strip scripts/styles, keep content, extract metadata |
| `web2spec/semantic.py` | split pages into headings, paragraphs and table rows |
| `web2spec/plan.py` | `CollectionPlan`, `FactSet`, `repair_factset`, lenient parsing |
| `web2spec/research.py` | brief, notes, gap rounds, aggregation, answer synthesis |
| `web2spec/llama_runtime.py` | start/stop `llama-server`, health checks, batched calls |
| `web2spec/tokens.py` | ChatML prompts without importing a tokenizer |
| `web2spec/cli.py` | unified command dispatcher and `doctor` |
| `scripts/23_mcp_server.py` | MCP stdio JSON-RPC server |
| `scripts/00_download_models.py` | model download and SHA-256 verification |

## Evidence lifecycle

The collector never trusts the model's output directly:

1. The planner proposes a plan with an entity and fields.
2. The extractor returns values plus claimed evidence.
3. `repair_factset` tries to find the evidence as an exact substring of the
   compressed page. A near match may be re-anchored; otherwise the value is
   dropped and the field is marked missing.
4. Facts whose names are not in the plan are moved to `dropped_foreign`.
5. `validate_factset` checks coverage and exact evidence a second time.
6. Only the validated `FactSet` is written to the report.

For research, each note must contain a quote. Aggregation is deterministic:
one value becomes a fact, several values become a conflict with per-source
attribution, and no value becomes a missing field.

This is not a formal proof that a page is truthful. It proves that the value
appears on the page that was fetched. The source URL is always retained.

## Model lifecycle

`llama-server` is started with:

- `-ngl 99` to offload layers to the GPU;
- Vulkan device `Vulkan0` by default on the reference laptop;
- an 8192-token context;
- one parallel slot (`-np 1`);
- a random free localhost port.

The runtime stops the server after each stage. This is deliberate: the 4B
analyst and the 0.5B planner/extractor are never resident together. The cost is
model load time (seconds), paid once per command rather than per page.

The PyTorch path is still available for evaluation and for machines without
`llama-server`, but it is optional and slower. `--backend auto` chooses GGUF
when the model path ends in `.gguf`.

## Search layer

`web2spec/search` and the research loop share the same search adapters:

- **SearXNG JSON**: recommended, self-hosted, no key. The instance must enable
  `formats: [html, json]`.
- **DuckDuckGo HTML**: no setup, but the public endpoint can rate-limit and is
  not a stable API.

The optional query-generation LLM is external (Ollama or OpenAI-compatible).
It is not bundled, because a 3B/7B model would compete with the pipeline for
VRAM. Without it, the task text is used as the query. The research path does
not use this optional model: the 4B analyst writes its own queries.

## MCP transport

`scripts/23_mcp_server.py` implements the MCP stdio transport directly:

- newline-delimited JSON-RPC 2.0, with a `Content-Length` fallback for clients
  that use framed messages;
- `initialize`, `ping`, `tools/list`, `tools/call`;
- notifications are accepted without a response;
- diagnostics go to stderr only;
- tool failures are returned as `isError: true` content, not as a broken
  protocol stream.

The server exposes `research` and `collect`. It deliberately does not expose a
raw `fetch` or `html` tool.

## Failure taxonomy

| Status | Meaning |
| --- | --- |
| `ok` | page fetched, plan and facts validated |
| `fetch_error` | network, HTTP error, robots.txt or empty body |
| `blocked` | WAF, captcha, login wall or anti-bot response |
| `plan_error` | planner did not produce a valid plan |
| `extract_error` | extractor output failed strict validation |

Research additionally reports `pages_shortfall` when fewer than the requested
minimum pages were usable. It never inflates the page count.

## Security and privacy

- No API keys are required for the default path.
- Search queries and page requests go directly from the local machine.
- Page content is untrusted. The pipeline returns data, never executes page
  instructions or tool calls.
- Prompt injection is still possible in the sense that a malicious page can
  bias which values the model notices. Exact evidence and source URLs make the
  result auditable; high-stakes numbers should still be checked at the source.
- Model weights, datasets and reports are not committed.

## Extension points

- Add a search provider by implementing the small adapter used by
  `web2spec/research.py` and `scripts/22_search_and_collect.py`.
- Add a page type by extending the task schema/planner data, not by weakening
  the evidence validator.
- Add an MCP client by pointing it at `python scripts/23_mcp_server.py`; no
  network transport is required.
- Add a model by adding a GGUF path and a checksum entry; keep one model per
  stage on 4 GB.

## Intentionally absent

- a JavaScript browser and SPA rendering;
- captcha/WAF bypass, authentication and paywall handling;
- site crawling, pagination discovery and cross-page entity deduplication;
- a hosted service or a paid search API;
- a claim of general extraction accuracy from the narrow benchmark.
