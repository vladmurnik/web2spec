from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "00_download_models.py"
_spec = importlib.util.spec_from_file_location("download_models", SCRIPT)
assert _spec is not None and _spec.loader is not None
dl = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = dl
_spec.loader.exec_module(dl)


class FakeResponse(io.BytesIO):
    status = 200

    def __init__(self, data: bytes, status: int = 200) -> None:
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_load_checksums_ignores_comments(tmp_path: Path) -> None:
    manifest = tmp_path / "checksums.sha256"
    manifest.write_text("# comment\nABC  file.gguf\n\n", encoding="utf-8")
    assert dl.load_checksums(manifest) == {"file.gguf": "abc"}


def test_sha256_and_verify(tmp_path: Path) -> None:
    target = tmp_path / "model.gguf"
    target.write_bytes(b"web2spec")
    expected = hashlib.sha256(b"web2spec").hexdigest()
    assert dl.sha256_file(target) == expected
    assert dl.verify(target, expected)
    assert not dl.verify(target, "0" * 64)
    assert not dl.verify(tmp_path / "missing.gguf", expected)


def test_selected_groups() -> None:
    assert [m.name for m in dl._selected("collector")] == ["planner", "extractor"]
    assert [m.name for m in dl._selected("analyst")] == ["analyst"]
    assert len(dl._selected("all")) == 3
    assert "collector-q3" not in dl.GROUPS
    assert "collector-q2" not in dl.GROUPS
    assert "planner-q3" not in dl.MODELS
    assert "planner-q2" not in dl.MODELS


def test_download_writes_and_replaces_part(tmp_path: Path, monkeypatch) -> None:
    payload = b"model-bytes" * 100
    part = tmp_path / "model.gguf.part"
    part.write_bytes(b"stale")

    def fake_urlopen(request, timeout):
        assert request.full_url == "https://example.test/model.gguf"
        return FakeResponse(payload, status=200)

    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
    dl.download("https://example.test/model.gguf", tmp_path / "model.gguf", progress=False)
    assert (tmp_path / "model.gguf").read_bytes() == payload
    assert not part.exists()


def test_download_resumes_with_range(tmp_path: Path, monkeypatch) -> None:
    payload = b"0123456789"
    part = tmp_path / "model.gguf.part"
    part.write_bytes(payload[:4])

    def fake_urlopen(request, timeout):
        assert request.headers["Range"] == "bytes=4-"
        return FakeResponse(payload[4:], status=206)

    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
    dl.download("https://example.test/model.gguf", tmp_path / "model.gguf", progress=False)
    assert (tmp_path / "model.gguf").read_bytes() == payload


def test_check_mode_reports_missing_and_valid(tmp_path: Path, monkeypatch) -> None:
    data = b"model"
    digest = hashlib.sha256(data).hexdigest()
    manifest = tmp_path / "checksums.sha256"
    manifest.write_text(f"{digest}  tiny.gguf\n", encoding="utf-8")
    model = dl.Model("tiny", "tiny.gguf", "https://example.test/tiny.gguf", "1 MB")
    monkeypatch.setitem(dl.MODELS, "tiny", model)

    args = ["--only", "tiny", "--check", "--dest", str(tmp_path), "--checksums", str(manifest)]
    assert dl.main(args) == 1
    (tmp_path / "tiny.gguf").write_bytes(data)
    assert dl.main(args) == 0
