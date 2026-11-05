"""Local llama.cpp runtime for the compressed GGUF models.

The production default is the Q4_K_M build of both 0.5B models.  The models are
small enough that a single ``llama-server`` process replaces the whole PyTorch
stack for inference: no torch import, no fp16 weights, and roughly 160-170
tokens/s on the 4 GB RTX 3050 Ti.

The module owns process lifecycle, health checks and HTTP calls.  Prompt
building stays in :mod:`web2spec.tokens`, and callers pass plain prompt strings.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLANNER = REPO_ROOT / "models/gguf/qwen05b-plan-Q4_K_M.gguf"
DEFAULT_EXTRACTOR = REPO_ROOT / "models/gguf/qwen05b-plan-extract-v2-Q4_K_M.gguf"
DEFAULT_DEVICE = os.environ.get("WEB2SPEC_LLAMA_DEVICE", "Vulkan0")
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]


def is_gguf(path: str | Path) -> bool:
    return str(path).lower().endswith(".gguf")


def resolve_server(server: str | None = None) -> str:
    """Find ``llama-server`` from an explicit path, env var, or ``PATH``."""

    candidate = server or os.environ.get("WEB2SPEC_LLAMA_SERVER")
    if candidate:
        path = Path(candidate)
        if not path.is_file():
            raise FileNotFoundError(f"llama-server не найден: {candidate}")
        return str(path)
    found = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if not found:
        raise FileNotFoundError(
            "llama-server не найден в PATH. Установите llama.cpp или задайте "
            "WEB2SPEC_LLAMA_SERVER."
        )
    return found


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_command(
    model: str | Path,
    *,
    server: str | None = None,
    port: int | None = None,
    context: int = 8192,
    device: str | None = DEFAULT_DEVICE,
) -> list[str]:
    """Build the deterministic llama-server command line."""

    return [
        resolve_server(server),
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port or free_port()),
        "-c",
        str(context),
        "-b",
        "2048",
        "-ub",
        "512",
        "-ngl",
        "99",
        "-np",
        "1",
        "--no-webui",
        *(["--device", device] if device else []),
    ]


class LlamaServer:
    """One resident ``llama-server`` process for a single GGUF model."""

    def __init__(
        self,
        model: str | Path,
        *,
        server: str | None = None,
        device: str | None = DEFAULT_DEVICE,
        context: int = 8192,
        startup_timeout: float = 240.0,
        request_timeout: float = 300.0,
    ) -> None:
        self.model = Path(model)
        self.server = server
        self.device = device
        self.context = context
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.port = free_port()
        self.process: subprocess.Popen[str] | None = None
        self._log_path: Path | None = None
        self._log_handle: Any = None

    def __enter__(self) -> LlamaServer:
        if not self.model.is_file():
            raise FileNotFoundError(
                f"GGUF-модель не найдена: {self.model}. "
                "Соберите её scripts/04_convert_gguf.ps1 или укажите --planner/--extractor."
            )
        self._start(self.device)
        try:
            self._wait_health()
        except RuntimeError:
            # A machine without the configured Vulkan device can still run on
            # CPU; retry once without --device before giving up.
            if not self.device:
                raise
            print(
                f"[llama] устройство {self.device} недоступно, повтор без --device",
                file=sys.stderr,
            )
            self.close()
            self._start(None)
            self._wait_health()
        return self

    def _start(self, device: str | None) -> None:
        command = server_command(
            self.model,
            server=self.server,
            port=self.port,
            context=self.context,
            device=device,
        )
        log = tempfile.NamedTemporaryFile(  # noqa: SIM115 - handle passed to Popen
            prefix="web2spec-llama-", suffix=".log", delete=False
        )
        self._log_path = Path(log.name)
        self._log_handle = log
        self.process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def _wait_health(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        url = f"http://127.0.0.1:{self.port}/health"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server завершился с кодом {self.process.returncode}: "
                    f"{self._log_tail()}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2.0) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(0.2)
        raise RuntimeError(
            f"llama-server не запустился за {self.startup_timeout:.0f} с: "
            f"{self._log_tail()}"
        )

    def _log_tail(self, lines: int = 8) -> str:
        if not self._log_path or not self._log_path.is_file():
            return "нет лога"
        text = self._log_path.read_text(encoding="utf-8", errors="replace")
        return " | ".join(text.strip().splitlines()[-lines:])

    def generate(self, prompt: str, *, max_new_tokens: int = 384) -> str:
        payload = {
            "prompt": prompt,
            "n_predict": max_new_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "seed": 42,
            "cache_prompt": False,
            "stream": False,
            "stop": STOP_TOKENS,
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/completion",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.request_timeout
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"llama-server completion failed: {exc}") from exc
        content = body.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"llama-server вернул неожиданный ответ: {body}")
        return content

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if self._log_path is not None:
            self._log_path.unlink(missing_ok=True)
            self._log_path = None

    def __exit__(self, *_: object) -> None:
        self.close()


def generate_batch(
    prompts: list[tuple[str, str]],
    model: str | Path,
    *,
    max_new_tokens: int,
    label: str,
    server: str | None = None,
    device: str | None = DEFAULT_DEVICE,
    context: int = 8192,
    startup_timeout: float = 240.0,
    request_timeout: float = 300.0,
) -> dict[str, str]:
    """Generate one output per ``(key, prompt)`` pair with a single server."""

    output: dict[str, str] = {}
    with LlamaServer(
        model,
        server=server,
        device=device,
        context=context,
        startup_timeout=startup_timeout,
        request_timeout=request_timeout,
    ) as runtime:
        for index, (key, prompt) in enumerate(prompts, 1):
            text = runtime.generate(prompt, max_new_tokens=max_new_tokens)
            output[key] = text
            print(
                f"[{label} {index}/{len(prompts)}] {key}: {len(text)} chars",
                file=sys.stderr,
            )
    return output
