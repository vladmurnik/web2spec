"""Publish fine-tuned artifacts to the Hugging Face Hub.

GitHub release assets remain the primary download source used by
``web2spec download-models``.  This script is the mirror path selected for the
project: it uploads the two collector models in Q4_K_M, Q3_K_M and Q2_K, the f16
GGUFs, and the merged Hugging Face directories.

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
MERGED_DIR = REPO_ROOT / "models" / "merged"
MODEL_CARD = REPO_ROOT / "docs" / "HF_MODEL_CARD.md"

GGUF_FILES = (
    "qwen05b-plan-Q4_K_M.gguf",
    "qwen05b-plan-Q3_K_M.gguf",
    "qwen05b-plan-Q2_K.gguf",
    "qwen05b-plan-extract-v2-Q4_K_M.gguf",
    "qwen05b-plan-extract-v2-Q3_K_M.gguf",
    "qwen05b-plan-extract-v2-Q2_K.gguf",
    "qwen05b-plan-f16.gguf",
    "qwen05b-plan-extract-v2-f16.gguf",
)
MERGED_FOLDERS = (
    "qwen05b-plan",
    "qwen05b-plan-extract-v2",
)


def selected_files(*, include_gguf: bool, include_f16: bool, include_merged: bool) -> list[Path]:
    files: list[Path] = []
    if include_gguf:
        for name in GGUF_FILES:
            if not include_f16 and "-f16." in name:
                continue
            files.append(GGUF_DIR / name)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload web2spec models to Hugging Face")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--token", default=None, help="defaults to HF_TOKEN")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--skip-f16", action="store_true")
    parser.add_argument("--skip-merged", action="store_true")
    args = parser.parse_args(argv)

    files = selected_files(
        include_gguf=not args.skip_gguf,
        include_f16=not args.skip_f16,
        include_merged=not args.skip_merged,
    )
    folders = (
        [MERGED_DIR / name for name in MERGED_FOLDERS]
        if not args.skip_merged
        else []
    )
    missing = [str(path) for path in [*files, *folders, MODEL_CARD] if not path.exists()]
    if missing:
        print("[error] missing artifacts:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    print(f"[plan] repo={args.repo_id} private={args.private}")
    for path in files:
        print(f"[plan] file   {path.relative_to(REPO_ROOT)}")
    for path in folders:
        print(f"[plan] folder {path.relative_to(REPO_ROOT)}")
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
    for path in folders:
        print(f"[upload] {path.name}/")
        api.upload_folder(
            folder_path=path,
            path_in_repo=f"merged/{path.name}",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"add merged {path.name}",
        )
    print("[done] upload complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
