from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "26_research.py"
_spec = importlib.util.spec_from_file_location("research_cli", SCRIPT)
assert _spec is not None and _spec.loader is not None
research_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(research_cli)


def test_domains_normalizes_and_dedupes() -> None:
    assert research_cli._domains(
        ["https://www.Example.com/path", "example.com, docs.example.com:443"]
    ) == ["example.com", "docs.example.com"]


def test_read_urls_strips_bom_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "urls.txt"
    path.write_text("\ufeffhttps://a.example\n\n https://b.example \n", encoding="utf-8")
    assert research_cli._read_urls(path) == ["https://a.example", "https://b.example"]
