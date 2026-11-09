---
license: apache-2.0
base_model:
  - Qwen/Qwen2.5-0.5B-Instruct
language:
  - ru
  - en
library_name: llama.cpp
tags:
  - web2spec
  - structured-extraction
  - gguf
  - qwen
---

# web2spec collector models

Fine-tuned Qwen2.5-0.5B models for the local web2spec planner/extractor cascade.
The runtime uses only the Q4_K_M GGUF files.

## Files

| Path | Role | Notes |
| --- | --- | --- |
| `gguf/qwen05b-plan-Q4_K_M.gguf` | planner | default runtime |
| `gguf/qwen05b-plan-extract-v2-Q4_K_M.gguf` | extractor | default runtime |

The Q4_K_M files are the measured default. Experimental Q3_K_M/Q2_K, f16 and
merged artifacts are local-only and are not published in this repository.

## Usage

Install web2spec from the repository and point it at the downloaded files:

```powershell
web2spec collect --task "Collect article titles and authors" `
  --url https://example.com/article `
  --planner models/gguf/qwen05b-plan-Q4_K_M.gguf `
  --extractor models/gguf/qwen05b-plan-extract-v2-Q4_K_M.gguf
```

The models return plans and evidence-backed fact JSON; the runtime validates
every quote before exposing a value. They are not general chat models.

## Provenance and limits

Both fine-tuned models start from `Qwen/Qwen2.5-0.5B-Instruct` and inherit its
Apache-2.0 license. The analyst used by `research` is the separate
`Qwen3-4B-Instruct-2507-Q4_K_M` file from the official Qwen GGUF repository.

Measurements are limited to the protocols in
[`BENCHMARKS.md`](https://github.com/vladmurnik/web2spec/blob/main/docs/BENCHMARKS.md).
Do not treat a single benchmark as a general accuracy claim. The training data
is not included in this repository.
