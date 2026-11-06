# Contributing

Thanks for taking the time to improve web2spec. The project is a local,
evidence-first research pipeline; changes are judged mainly by whether they
preserve that contract.

## Development setup

```powershell
git clone https://github.com/vladmurnik/web2spec.git
cd web2spec
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest
```

The runtime tests do not need a GPU, network access or model weights. For the
optional PyTorch path install `.[hf]`; for training and evaluation install
`.[train]`.

## Checks

```powershell
python -m ruff check .
python -m pytest
```

Both commands must pass on Windows and Linux. Add tests for behaviour changes;
do not weaken the evidence validator to make a test pass.

## Contracts

1. **Evidence is mandatory.** A fact is only valid when its `evidence` is an
   exact substring of the compressed page. If it cannot be verified, the field
   belongs in `missing` or `conflicts`.
2. **No invented values.** Never fill a gap from model memory, a blog snippet or
   a different page.
3. **No HTML to the caller.** The pipeline returns compact facts and source
   URLs. Do not add a tool that dumps full pages into the caller's context.
4. **One model at a time.** The reference machine has 4 GB VRAM. Do not add a
   path that keeps the 4B analyst and the 0.5B collector resident together.
5. **No paid APIs.** Search uses SearXNG or DuckDuckGo HTML. Do not add keys or
   metered services to the default path.
6. **Do not commit models, datasets, reports or secrets.** `models/`,
   `data/`, `outputs/` and `.env` are ignored. Only the checksum manifest is
   tracked.

## Commits and pull requests

Use short imperative commit subjects, for example:

```text
fix: reject evidence outside the compressed page
docs: explain SearXNG JSON setup
```

A pull request should state the problem, the change, the verification command
and any remaining limitation. Keep unrelated refactors out of the same commit.

## Reporting bugs

Include the command, Python version, OS, GPU/VRAM, model files, and the smallest
JSON output that shows the problem. Remove URLs and text you cannot share.
Security issues should be reported privately to the repository owner rather
than in a public issue.
