"""Publish the Q4_K_M runtime collector models to the Hugging Face Hub.

GitHub release assets remain the primary download source used by
``web2spec download-models``.  This script mirrors only the two Q4_K_M models
that the MCP/CLI runtime uses.  Experimental Q3_K_M/Q2_K, f16 and merged
artifacts stay local; they are not uploaded by this command.

Authentication is read from ``HF_TOKEN`` or the token passed with ``--token``.
Never commit a token or put it in a config file.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = os.environ.get("WEB2SPEC_HF_REPO", "vladmurnik/web2spec-models")
GGUF_DIR = REPO_ROOT / "models" / "gguf"
MODEL_CARD = REPO_ROOT / "docs" / "HF_MODEL_CARD.md"

GGUF_FILES = (
    "qwen05b-plan-Q4_K_M.gguf",
    "qwen05b-plan-extract-v2-Q4_K_M.gguf",
)


def selected_files() -> list[Path]:
    return [GGUF_DIR / name for name in GGUF_FILES]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload web2spec models to Hugging Face")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--token", default=None, help="defaults to HF_TOKEN")
    args = parser.parse_args(argv)

    files = selected_files()
    missing = [str(path) for path in [*files, MODEL_CARD] if not path.exists()]
    if missing:
        print("[error] missing artifacts:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    print(f"[plan] repo={args.repo_id} private={args.private}")
    for path in files:
        print(f"[plan] file   {path.relative_to(REPO_ROOT)}")
    print(f"[plan] card   {MODEL_CARD.relative_to(REPO_ROOT)} -> README.md")
    if args.dry_run:
        return 0

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print(
            "[error] HF_TOKEN is not set. Create a write token at "
            "https://huggingface.co/settings/tokens and export it for this command.",
            file=sys.stderr,
        )
        return 2
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print(
            "[error] install the upload extra: pip install -e '.[upload]'",
            file=sys.stderr,
        )
        return 2

    api = HfApi(token=token)
    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=token,
    )
    print(f"[repo] https://huggingface.co/{args.repo_id}")
    api.upload_file(
        path_or_fileobj=MODEL_CARD,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="docs: add web2spec model card",
    )
    for path in files:
        print(f"[upload] {path.name}")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"gguf/{path.name}",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"add {path.name}",
        )
    print("[done] upload complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
