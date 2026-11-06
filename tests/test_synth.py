from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from web2spec.config import REPO_ROOT
from web2spec.synth import build_messages, synthesize

HTML = (
    "<html><body><div class='product'><h1>Товар</h1>"
    "<span class='price'>1000</span><p>Описание товара на странице</p></div></body></html>"
)
SCHEMA = {
    "type": "object",
    "required": ["title"],
    "properties": {"title": {"type": "string"}},
}


class MockTeacher:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls = 0

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def make_input(**overrides) -> dict:
    record = {
        "html": HTML,
        "instruction": "Извлеки название товара",
        "source_url": "https://shop.ru/p/1",
    }
    record.update(overrides)
    return record


def test_accepts_valid_output() -> None:
    samples, stats = synthesize([make_input()], MockTeacher('{"title": "Товар"}'), schema=SCHEMA)
    assert len(samples) == 1
    assert samples[0]["target"] == {"title": "Товар"}
    assert stats["accepted"] == 1
    assert stats["rejected"] == 0


def test_rejects_invalid_json() -> None:
    samples, stats = synthesize([make_input()], MockTeacher("не json"), schema=SCHEMA)
    assert samples == []
    assert stats["by_reason"]["E_JSON"] == 1


def test_rejects_schema_mismatch() -> None:
    samples, stats = synthesize([make_input()], MockTeacher('{"price": 10}'), schema=SCHEMA)
    assert samples == []
    assert stats["by_reason"]["E_TARGET"] == 1


def test_parses_fenced_output() -> None:
    teacher = MockTeacher('```json\n{"title": "Товар"}\n```')
    samples, _ = synthesize([make_input()], teacher, schema=SCHEMA)
    assert samples[0]["target"]["title"] == "Товар"


def test_dedup_drops_repeats() -> None:
    samples, stats = synthesize(
        [make_input(), make_input()], MockTeacher('{"title": "Товар"}'), schema=SCHEMA
    )
    assert len(samples) == 1
    assert stats["by_reason"]["E_DUP"] == 1


def test_teacher_error_is_counted() -> None:
    samples, stats = synthesize(
        [make_input()], MockTeacher(RuntimeError("boom")), schema=SCHEMA
    )
    assert samples == []
    assert stats["by_reason"]["E_TEACHER"] == 1


def test_short_html_rejected() -> None:
    samples, stats = synthesize(
        [make_input(html="<p>x</p>")], MockTeacher('{"title": "x"}'), schema=SCHEMA
    )
    assert samples == []
    assert stats["by_reason"]["E_TOO_SHORT"] == 1


def test_build_messages_truncates() -> None:
    system, user = build_messages(make_input(html="<div>" + "текст " * 5000 + "</div>"), SCHEMA)
    assert "JSON Schema" in system
    assert len(user) <= 6000


def test_stats_shape() -> None:
    _, stats = synthesize([make_input()], MockTeacher('{"title": "Товар"}'), schema=SCHEMA)
    assert set(stats) == {"total", "accepted", "rejected", "by_reason", "duration_s"}


def test_cli_dry_run(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(make_input(), ensure_ascii=False) + "\n", encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "PYTHONIOENCODING": "utf-8",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/07_synth.py"),
            "--input",
            str(source),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "SYSTEM" in result.stdout
    assert "Сеть не использовалась" in result.stdout


def test_teacher_requires_base_url(monkeypatch) -> None:
    from web2spec.synth import OpenAICompatTeacher

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        OpenAICompatTeacher(base_url="")
