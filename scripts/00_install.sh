#!/usr/bin/env bash
# One-command local setup for web2spec on Linux/macOS.
#
# Creates .venv, installs the package in editable mode, checks llama.cpp,
# downloads verified GGUF models, and runs the doctor check.
#
# Usage:
#   bash scripts/00_install.sh
#   bash scripts/00_install.sh --dev --skip-models
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEV=0
SKIP_MODELS=0
for arg in "$@"; do
    case "$arg" in
        --dev) DEV=1 ;;
        --skip-models) SKIP_MODELS=1 ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

find_python() {
    for candidate in python3.12 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && \
            "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

PYTHON="$(find_python || true)"
if [[ -z "$PYTHON" ]]; then
    echo "Python 3.12+ not found. Install it and rerun." >&2
    exit 1
fi

if [[ ! -d .venv ]]; then
    echo "==> creating .venv"
    "$PYTHON" -m venv .venv
fi

if [[ -x .venv/bin/python ]]; then
    VENV_PYTHON="$ROOT/.venv/bin/python"
elif [[ -x .venv/Scripts/python.exe ]]; then
    VENV_PYTHON="$ROOT/.venv/Scripts/python.exe"
else
    echo "virtual environment is incomplete" >&2
    exit 1
fi

echo "==> installing Python package"
"$VENV_PYTHON" -m pip install --upgrade pip
if [[ "$DEV" -eq 1 ]]; then
    "$VENV_PYTHON" -m pip install -e ".[dev]"
else
    "$VENV_PYTHON" -m pip install -e .
fi

if command -v llama-server >/dev/null 2>&1; then
    echo "==> llama-server: $(command -v llama-server)"
else
    echo "WARNING: llama-server not found." >&2
    echo "  Install llama.cpp: https://github.com/ggml-org/llama.cpp" >&2
    echo "  Then open a new terminal and run this script again." >&2
fi

if [[ "$SKIP_MODELS" -eq 0 ]]; then
    echo "==> downloading verified models"
    "$VENV_PYTHON" -m web2spec.cli download-models --only all
fi

echo "==> checking installation"
set +e
"$VENV_PYTHON" -m web2spec.cli doctor
DOCTOR_CODE=$?
set -e

echo
echo "Next steps:"
echo "  .venv/bin/web2spec collect --task \"...\" --url https://example.com"
echo "  .venv/bin/web2spec research --task \"...\""
echo "  .venv/bin/web2spec mcp   # stdio MCP server"
echo
echo "MCP config for an MCP client (use an absolute path):"
cat <<EOF
"mcp": {
  "web2spec": {
    "type": "local",
    "command": ["$VENV_PYTHON", "-m", "web2spec.cli", "mcp"],
    "enabled": true
  }
}
EOF

if [[ "$DOCTOR_CODE" -ne 0 ]]; then
    echo "WARNING: doctor found missing pieces; fix them and rerun." >&2
fi
exit "$DOCTOR_CODE"
