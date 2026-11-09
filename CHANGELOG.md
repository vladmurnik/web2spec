# Changelog

All notable changes are documented here. The project follows a lightweight
version of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- One-command setup scripts for Windows (`scripts/00_install.ps1`) and
  Linux/macOS (`scripts/00_install.sh`): venv, editable install, optional
  llama.cpp install, verified model download, and `doctor`.
- SQLite page/search cache with a 24-hour default TTL, `--cache-ttl`,
  `--no-cache`, and cache hit statistics.
- Incremental collector and search runs through `--previous`: unchanged pages
  are reused by content hash without starting either model.
- Experimental Q3_K_M and Q2_K collector GGUFs, generated locally with hashes
  in `models/checksums-extra.sha256`; they are not distributed.
- Optional f16 GGUFs and merged HF-format archives for both collector models,
  kept local with hashes in `models/checksums-extra.sha256`.
- `upload-models` and `docs/HF_MODEL_CARD.md` for mirroring only the Q4_K_M
  runtime models; the actual upload requires an `HF_TOKEN` and is not performed
  automatically.
- Dated real-page benchmark harness (`web2spec benchmark`) covering docs,
  API reference, pricing and product specs.
- `content_hash` and `cache_hit` fields in collector reports.
- Explicit caller-provided schemas: `collect --field name[:kind]` and MCP
  `collect.fields` skip the planner and preserve field names exactly.

### Changed

- `download-models --only all` fetches the three default runtime models; the
  Q3_K_M/Q2_K variants are local builds and are no longer download groups.
- The public release asset set is limited to the Q4_K_M planner/extractor and
  `checksums.sha256`; optional artifacts are not hosted.

### Removed

- Optional Q3_K_M/Q2_K, f16 and merged assets from the `v0.4.0` release. The
  public release now contains only the models used by the CLI/MCP runtime.

## [0.4.0] - 2026-11-06

### Added

- `research` path: a local Qwen3-4B analyst builds a brief, searches, reads
  20–50 pages, keeps quote-backed notes, detects gaps and synthesizes an answer.
- MCP stdio server with two tools: `research` for vague tasks and `collect` for
  known URLs. Works with Claude Desktop, Cursor, Cline and other MCP clients.
- One-call `collect`: task + optional URLs/domains -> search -> fetch -> facts.
- Local search through SearXNG JSON or DuckDuckGo HTML; no paid search APIs.
- Semantic HTML chunks, deterministic aggregation of facts, conflicts and
  missing fields.
- Q4_K_M GGUF runtime through `llama-server`; `web2spec doctor` checks models,
  server and Python.
- `web2spec download-models` with SHA-256 verification.
- Unified `web2spec` CLI for runtime, evaluation and training stages.

### Changed

- Q4_K_M is the default runtime; PyTorch is an optional fallback.
- Runtime dependencies no longer include the training stack. Use `[hf]` for the
  PyTorch path and `[train]` for QLoRA training.

### Measured limits

- Collector quality is measured on 50 held-out article/Habr pairs only:
  Q4_K_M task success 100%, fact F1 1.0, 0 wrong values after repair.
- Research quality is an offline smoke test on 20 Habr fixtures, not a
  multi-domain benchmark. See `docs/BENCHMARKS.md`.

## [0.3.0]

- Plan-conditioned extractor v2, runtime cascade with evidence repair buffer,
  Q4_K_M evaluation, metadata-preserving HTML compression.

## [0.2.0]

- Task-conditioned planner and plan-conditioned extractor training pipeline.

## [0.1.0]

- Initial QLoRA extraction experiments and synthetic data generator.
