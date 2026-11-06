# Benchmarks

The numbers below are the measurements that exist in this repository. They are
proxy evaluations on a narrow slice, not a general accuracy claim. Read the
threats to validity before quoting them.

## Hardware and software

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX 3050 Ti Laptop, 4 GB VRAM |
| Power | AC |
| Runtime | `llama.cpp` `llama-server`, Vulkan backend |
| Analyst | Qwen3-4B-Instruct-2507 Q4_K_M, 2381.6 MB |
| Collector | Qwen2.5-0.5B planner + plan-conditioned extractor v2, Q4_K_M, 379.4 MB each |
| OS | Windows 11 |
| Python | 3.12 |

## Collector: held-out article/Habr pairs

### Protocol

- 50 task/page pairs from the held-out plan-extract test set;
- 18 unique article/Habr URLs;
- gold plan and gold facts are fixed before the run;
- the model must return a valid plan, a valid FactSet and exact evidence;
- a human/task-level success check asks whether the page produced the requested
  information without a wrong value.

The test slice is deliberately narrow: article pages, mostly `title` and
`author`, with some fields absent. No job, product, real-estate or tariff pages
were measured.

### Results

| Metric | Q4_K_M | PyTorch fp16 |
| --- | ---: | ---: |
| Task success (proxy) | **100%** (50/50) | 94% (47/50) |
| Fact F1 / precision / recall | **1.0 / 1.0 / 1.0** | 0.9388 / — / — |
| Strict valid FactSet after repair | **100%** | 100% |
| Wrong values after repair | **0** | 0 |
| Wall time, 50 pages | **101.3 s** | 586.7 s |
| Per page | **~2.0 s** | ~11.7 s |
| Planner throughput | ~172 tok/s | ~24 tok/s |
| Extractor throughput | ~161 tok/s | ~24 tok/s |

The perfect Q4 score is explained by the test composition, not by general
superiority:

- the gold plans are narrow and mostly ask for `title`;
- 60 gold fields are absent from the pages and correctly returned as `missing`;
- 51 fields are present and matched;
- the model therefore has little room to invent values on this slice.

This is a regression benchmark. A result on one host does not transfer to
pricing tables, job listings or product specifications.

### Planner and extractor alone

| Stage | Test set | Metric | Value |
| --- | --- | --- | ---: |
| Planner | 136 unseen Habr tasks | field F1 | **0.8220** (val), **0.7714** (test) |
| Extractor v2 | 90 held-out Habr pages | fact F1 | **0.9357** |
| Extractor v2 | same | evidence rate | **0.9524** |
| Extractor v2 | same | missing accuracy | **0.9118** |
| Extractor v2 | same | JSON valid | **0.9444** |
| Extractor v2 | same | exact FactSet | **0.90** |

Errors in the extractor test: 4 wrong values and 5 invalid responses out of 90.
The repair stage removed wrong values from the runtime report; it cannot recover
information the page did not contain.

## Research: offline smoke test

### Protocol

- 20 saved Habr pages, offline fixtures;
- `--min-pages 20 --max-pages 20 --max-rounds 1`;
- Qwen3-4B-Instruct-2507 Q4_K_M;
- one 4B server for the whole run;
- no network and no paid API.

The fixtures are mostly metadata-only: relative dates such as “6 hours ago”,
and many pages have no author. This tests the loop, evidence handling and
aggregation, not answer quality across domains.

### Results

| Metric | Value |
| --- | ---: |
| Pages ok | **20/20** |
| `pages_shortfall` | 0 |
| Notes | 37 |
| Entities | 20 |
| Page errors | 0 |
| Gaps | 35 |
| Honest note omissions | 10 (mostly missing authors) |
| 4B calls | 22 |
| Prompt / generated tokens | 14,116 / 4,629 |
| Generation speed | **24.70 tok/s** |
| Model load | 7.026 s |
| Full run | **205.597 s** |
| Model size | 2381.6 MB |

A multi-domain research benchmark (tariffs, products, jobs) does not exist yet.
Do not describe the 20/20 result as “research accuracy”.

## What is not measured

- extraction quality on job, product, real-estate, news or tariff pages;
- research answer correctness on anything other than Habr fixtures;
- recall against the open web (search coverage);
- behavior under WAF, captcha, JavaScript-only sites or authentication;
- long-running stability across thousands of pages;
- energy consumption, CPU-only latency, or GPUs other than the reference one.

## Reproduction

The evaluation scripts and configs are committed; the datasets and model
weights are not. With the models present:

```powershell
web2spec eval-q4 --limit 50
web2spec eval-runtime --limit 50
web2spec eval-plan
web2spec eval-plan-extract
```

The original Russian lab notes, including dataset sizes and training curves, are
in [`STATS.md`](STATS.md) and [`PLANNER_PLAN.md`](PLANNER_PLAN.md). They are
historical records and may contain outdated setup details.
