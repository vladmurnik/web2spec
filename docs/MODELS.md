# Models

web2spec does not train a model at runtime. It downloads or builds three GGUF
files and starts them through `llama-server`.

## Runtime artifacts

| Role | File | Size | SHA-256 |
| --- | --- | ---: | --- |
| Analyst (`research`) | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2381.6 MB | `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597` |
| Planner (`collect`) | `qwen05b-plan-Q4_K_M.gguf` | 379.4 MB | `6579c854eec279f8e0f15e00ed27f10f9fff916318e8a8a1cbe9809450dd0a2a` |
| Extractor (`collect`) | `qwen05b-plan-extract-v2-Q4_K_M.gguf` | 379.4 MB | `aa454fbe3dddb32bd6a306e7a1e1fcc6bebb37819e1d18f8c65d6ba4cbd46209` |

The checksum manifest is [`models/checksums.sha256`](../models/checksums.sha256).
The files are not stored in git.

## Download

```powershell
web2spec download-models --only all
web2spec download-models --check
```

The command writes to `models/gguf/` by default, resumes a partial `.part`
download, verifies SHA-256 and deletes a corrupt file only by refusing to use
it. Use `--dest` to choose another directory, then point `WEB2SPEC_ANALYST`,
`WEB2SPEC_PLANNER` and `WEB2SPEC_EXTRACTOR` at the files.

The analyst is fetched from the official Qwen GGUF repository:

```text
https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF
```

The two 0.5B collector GGUFs are attached to the GitHub release for the current
version. If the release asset is unavailable, build them from the training
pipeline described below.

## Provenance

| Artifact | Base model | License |
| --- | --- | --- |
| Analyst | Qwen3-4B-Instruct-2507 | Apache-2.0 |
| Planner | Qwen2.5-0.5B-Instruct | Apache-2.0 |
| Extractor v2 | Qwen2.5-0.5B-Instruct | Apache-2.0 |

The fine-tuned planner and extractor inherit the base model license. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). Model behavior and outputs
are not warranted; evaluate them on your own pages.

## Rebuilding the collector

The training data is not committed. The original experiment artifacts and
instructions are preserved in [`DATASET.md`](DATASET.md),
[`PLANNER_PLAN.md`](PLANNER_PLAN.md) and the numbered scripts.

At a high level:

```text
raw pages -> compress_html -> task-conditioned plans -> QLoRA planner
           -> plan-conditioned facts -> QLoRA extractor v2
           -> merge adapter -> convert to GGUF Q4_K_M
```

```powershell
pip install -e ".[train]"
web2spec train --config configs/train_plan_extract_reliable_4gb.yaml
web2spec merge --config configs/train_plan_extract_reliable_4gb.yaml
pwsh scripts/04_convert_gguf.ps1 `
  -ModelDir models/merged/qwen05b-plan-extract-v2 `
  -Name qwen05b-plan-extract-v2 -Quant Q4_K_M
```

The exact dataset sizes, seeds and evaluation protocol are in
[`BENCHMARKS.md`](BENCHMARKS.md) and [`STATS.md`](STATS.md).

### Quantization caveat

The Qwen2.5-0.5B hidden size is 896, which is not divisible by 256. The
quantizer therefore leaves some tensors at q5_0/q6_K/q8_0. The effective size is
about 6.35 bits per weight, not a pure 4-bit file. This is expected behavior,
not a conversion failure.

## llama.cpp

The runtime needs `llama-server` from
[`llama.cpp`](https://github.com/ggml-org/llama.cpp). On Windows:

```powershell
winget install ggml.llamacpp
llama-server --version
```

On Linux, build with the backend you need (Vulkan, CUDA or CPU) and put
`llama-server` on `PATH`. The runtime resolves the binary from, in order:

1. `--llama-server`;
2. `WEB2SPEC_LLAMA_SERVER`;
3. `PATH`.

The default device is `Vulkan0`; override it with `--device` or
`WEB2SPEC_LLAMA_DEVICE`. The default context is 8192 tokens. On a 4 GB GPU, do
not increase the context or run two servers at once.

## Memory budget

| Process | Approximate VRAM |
| --- | ---: |
| `llama-server` + 4B Q4_K_M | 2.4 GB weights + KV cache |
| `llama-server` + 0.5B Q4_K_M | 0.4 GB weights + KV cache |

The 4B path leaves little headroom on a 4 GB card. Close other GPU workloads and
stop Ollama before running research. The CLI starts and stops one server per
stage rather than keeping both models alive.

## PyTorch fallback

For `--backend hf`, install the optional stack:

```powershell
pip install -e ".[hf]"
```

The fallback expects merged Hugging Face directories, not GGUF files. It is
useful for evaluation and debugging; the Q4 path is faster and is the default.
