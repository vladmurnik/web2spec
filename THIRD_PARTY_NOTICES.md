# Third-party notices

web2spec's own source code is MIT-licensed (see `LICENSE`). The runtime models
are derived from Qwen models and are distributed under their upstream licenses.

## Qwen2.5-0.5B-Instruct

- Upstream: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- License: Apache License 2.0
- Used by: planner and plan-conditioned extractor v2 fine-tunes

## Qwen3-4B-Instruct-2507

- Upstream: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- GGUF build: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF
- License: Apache License 2.0
- Used by: research analyst

## llama.cpp

- Upstream: https://github.com/ggml-org/llama.cpp
- License: MIT
- Used by: local GGUF inference (`llama-server`)

## Python dependencies

The packages listed in `pyproject.toml` retain their own licenses. See the
metadata of each installed package for details.

Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
