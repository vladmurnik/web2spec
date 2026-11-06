# Changelog

All notable changes are documented here. The project follows a lightweight
version of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
