from __future__ import annotations

import sys
from pathlib import Path

import pytest

from web2spec import gguf
from web2spec.merge import merge_adapter

PS1_CONVERT = Path("scripts/04_convert_gguf.ps1")
PS1_SERVE = Path("scripts/06_serve.ps1")


def test_find_binary_prefers_release(tmp_path: Path) -> None:
    release = tmp_path / "build" / "bin" / "Release"
    release.mkdir(parents=True)
    binary = release / "llama-quantize.exe"
    binary.write_bytes(b"")
    assert gguf.find_binary(tmp_path, "llama-quantize") == binary.resolve()


def test_find_binary_is_case_insensitive(tmp_path: Path) -> None:
    root = tmp_path / "bin"
    root.mkdir()
    (root / "LLAMA-QUANTIZE.EXE").write_bytes(b"")
    assert gguf.find_binary(tmp_path, "llama-quantize").exists()


def test_find_binary_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        gguf.find_binary(tmp_path, "llama-quantize")


def test_find_llama_cpp_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path))
    assert gguf.find_llama_cpp() == tmp_path.resolve()


def test_find_llama_cpp_requires_setting(monkeypatch) -> None:
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    with pytest.raises(FileNotFoundError):
        gguf.find_llama_cpp()


def test_convert_requires_script(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        gguf.convert(model, tmp_path / "out.gguf", llama_cpp=tmp_path)


def test_merge_requires_adapter_config(tmp_path: Path) -> None:
    before = "torch" in sys.modules
    with pytest.raises(FileNotFoundError):
        merge_adapter("base", tmp_path / "missing", tmp_path / "out")
    assert ("torch" in sys.modules) == before


def test_convert_script_flags() -> None:
    text = PS1_CONVERT.read_text(encoding="utf-8")
    assert "convert_and_quantize" in text
    assert "Q4_K_M" in text
    assert "$ErrorActionPreference = \"Stop\"" in text


def test_serve_script_flags() -> None:
    text = PS1_SERVE.read_text(encoding="utf-8")
    assert "--jinja" in text
    assert "-ngl" in text
    assert "llama-server" in text


def test_ps1_files_are_utf8() -> None:
    for path in (PS1_CONVERT, PS1_SERVE):
        path.read_bytes().decode("utf-8")
