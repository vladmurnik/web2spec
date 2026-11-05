from __future__ import annotations

import sys
from pathlib import Path

from web2spec import llama_runtime
from web2spec.llama_runtime import (
    DEFAULT_EXTRACTOR,
    DEFAULT_PLANNER,
    free_port,
    is_gguf,
    server_command,
)


def test_default_models_are_compressed_gguf() -> None:
    assert DEFAULT_PLANNER.name.endswith("Q4_K_M.gguf")
    assert DEFAULT_EXTRACTOR.name.endswith("Q4_K_M.gguf")
    assert DEFAULT_PLANNER != DEFAULT_EXTRACTOR


def test_is_gguf_case_insensitive() -> None:
    assert is_gguf("model.gguf")
    assert is_gguf(Path("model.GGUF"))
    assert not is_gguf("models/merged/model")


def test_free_port_is_usable() -> None:
    port = free_port()
    assert 0 < port < 65536


def test_server_command_has_runtime_flags(monkeypatch) -> None:
    monkeypatch.setattr(llama_runtime, "resolve_server", lambda server=None: server)
    command = server_command(
        "model.gguf",
        server=sys.executable,
        port=12345,
        context=4096,
        device="Vulkan0",
    )
    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "model.gguf"]
    assert command[command.index("--port") + 1] == "12345"
    assert command[command.index("-c") + 1] == "4096"
    assert command[command.index("-ngl") + 1] == "99"
    assert command[command.index("--device") + 1] == "Vulkan0"


def test_server_command_can_disable_device(monkeypatch) -> None:
    monkeypatch.setattr(llama_runtime, "resolve_server", lambda server=None: server)
    command = server_command("model.gguf", server=sys.executable, device=None)
    assert "--device" not in command
