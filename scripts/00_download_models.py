"""Download the runtime GGUF models and verify their SHA-256 hashes.

The analyst is published by the Qwen team.  The two Q4_K_M collector models are
fine-tuned in this repository and attached to the GitHub release.  Nothing is
downloaded implicitly by the runtime: this command makes the model step
explicit and verifiable.

Examples::

    web2spec download-models --only all
    web2spec download-models --only analyst --dest D:/models
    web2spec download-models --check
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "models/gguf"
DEFAULT_CHECKSUMS = REPO_ROOT / "models/checksums.sha256"
RELEASE_BASE = os.environ.get(
    "WEB2SPEC_MODEL_BASE_URL",
    "https://github.com/vladmurnik/web2spec/releases/download/v0.4.0",
)
HF_BASE = "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main"


@dataclass(frozen=True)
class Model:
    name: str
    filename: str
    url: str
    approx_size: str


MODELS = {
    "analyst": Model(
        "analyst",
        "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        f"{HF_BASE}/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "2.4 GB",
    ),
    "planner": Model(
        "planner",
        "qwen05b-plan-Q4_K_M.gguf",
        f"{RELEASE_BASE}/qwen05b-plan-Q4_K_M.gguf",
        "380 MB",
    ),
    "extractor": Model(
        "extractor",
        "qwen05b-plan-extract-v2-Q4_K_M.gguf",
        f"{RELEASE_BASE}/qwen05b-plan-extract-v2-Q4_K_M.gguf",
        "380 MB",
    ),
}

DEFAULT_MODELS = ("analyst", "planner", "extractor")
GROUPS = {
    "all": DEFAULT_MODELS,
    "collector": ("planner", "extractor"),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksums(path: Path = DEFAULT_CHECKSUMS) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"checksum manifest not found: {path}")
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"invalid checksum line: {line!r}")
        digest, filename = parts
        checksums[filename] = digest.lower()
    return checksums


def verify(path: Path, expected: str) -> bool:
    return path.is_file() and sha256_file(path).lower() == expected.lower()


def _human_mb(size: int) -> str:
    return f"{size / (1024**2):.1f} MB"


def download(
    url: str,
    dest: Path,
    *,
    timeout: float = 60.0,
    progress: bool = True,
) -> None:
    """Download ``url`` to ``dest`` with a resumable ``.part`` file."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "web2spec/0.4"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset:
            part.unlink(missing_ok=True)
            offset = 0
            request = urllib.request.Request(url, headers={"User-Agent": "web2spec/0.4"})
            response = urllib.request.urlopen(request, timeout=timeout)
        else:
            raise

    with response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        length = response.headers.get("Content-Length")
        total = (offset + int(length)) if length else None
        done = offset
        if progress:
            print(
                f"  downloading {dest.name} ({_human_mb(done) if done else '0 MB'}"
                f"{f' of {_human_mb(total)}' if total else ''})",
                flush=True,
            )
        with part.open(mode) as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                done += len(chunk)
                if progress and total and done // (20 * 1024 * 1024) != (done - len(chunk)) // (
                    20 * 1024 * 1024
                ):
                    print(f"  {done * 100 // total:3d}%", flush=True)
    part.replace(dest)


def _selected(only: str) -> list[Model]:
    if only in GROUPS:
        return [MODELS[name] for name in GROUPS[only]]
    return [MODELS[only]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify web2spec GGUF models")
    parser.add_argument(
        "--only",
        choices=(*GROUPS, *MODELS),
        default="all",
        help=(
            "model or group: all (default runtime), collector (Q4_K_M), "
            "or one model name"
        ),
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUMS)
    parser.add_argument("--check", action="store_true", help="verify existing files only")
    parser.add_argument("--force", action="store_true", help="re-download even if valid")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    try:
        checksums = load_checksums(args.checksums)
    except (OSError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    failures = 0
    for model in _selected(args.only):
        target = args.dest / model.filename
        expected = checksums.get(model.filename)
        if expected is None:
            print(f"[error] no checksum for {model.filename}", file=sys.stderr)
            failures += 1
            continue
        if verify(target, expected) and not args.force:
            print(f"[ok]   {model.name}: {target} (verified)")
            continue
        if args.check:
            state = "missing" if not target.exists() else "checksum mismatch"
            print(f"[fail] {model.name}: {target} ({state})", file=sys.stderr)
            failures += 1
            continue
        print(f"[get]  {model.name}: {model.url} -> {target} (~{model.approx_size})")
        try:
            download(model.url, target, timeout=args.timeout)
        except (OSError, urllib.error.URLError) as exc:
            print(f"[error] download failed for {model.name}: {exc}", file=sys.stderr)
            failures += 1
            continue
        if not verify(target, expected):
            print(
                f"[error] checksum mismatch for {target}; delete it and retry",
                file=sys.stderr,
            )
            failures += 1
            continue
        print(f"[ok]   {model.name}: {target} (verified)")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
