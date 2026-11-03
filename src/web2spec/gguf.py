"""llama.cpp wrapper: HF -> GGUF -> quantized GGUF."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_BINARY_DIRS = ("build/bin/Release", "build/bin", "build", "bin", ".")


def find_llama_cpp(root: Path | None = None) -> Path:
    candidate = root or os.environ.get("LLAMA_CPP_DIR")
    if not candidate:
        raise FileNotFoundError(
            "llama.cpp не найден: задайте LLAMA_CPP_DIR или аргумент --llama-cpp"
        )
    path = Path(candidate).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"каталог llama.cpp не существует: {path}")
    return path.resolve()


def find_binary(llama_cpp: Path, name: str) -> Path:
    """Find a llama.cpp executable across common build layouts (case-insensitive)."""

    wanted = {name.lower(), f"{name}.exe".lower()}
    for relative in _BINARY_DIRS:
        directory = llama_cpp / relative
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            if entry.is_file() and entry.name.lower() in wanted:
                return entry.resolve()
    raise FileNotFoundError(
        f"бинарник {name} не найден в {llama_cpp}. Соберите llama.cpp "
        "(cmake --build build --config Release)"
    )


def convert(
    model_dir: Path,
    out_file: Path,
    *,
    llama_cpp: Path,
    outtype: str = "f16",
) -> Path:
    script = Path(llama_cpp) / "convert_hf_to_gguf.py"
    if not script.exists():
        raise FileNotFoundError(f"нет скрипта конвертации: {script}")
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"нет config.json в {model_dir}")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(script),
            str(model_dir),
            "--outfile",
            str(out_file),
            "--outtype",
            outtype,
        ],
        cwd=Path(llama_cpp),
    )
    if not out_file.exists():
        raise RuntimeError(f"конвертация завершилась без файла {out_file}")
    return out_file.resolve()


def quantize(
    src: Path,
    dst: Path,
    *,
    llama_cpp: Path,
    quant_type: str = "Q4_K_M",
) -> Path:
    binary = find_binary(Path(llama_cpp), "llama-quantize")
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"нет исходного GGUF: {src}")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run([str(binary), str(src), str(dst), quant_type], cwd=Path(llama_cpp))
    if not dst.exists():
        raise RuntimeError(f"квантование завершилось без файла {dst}")
    return dst.resolve()


def convert_and_quantize(
    model_dir: Path,
    out_dir: Path,
    *,
    llama_cpp: Path,
    quant_type: str = "Q4_K_M",
    name: str = "web2spec",
) -> Path:
    out_dir = Path(out_dir)
    f16 = convert(model_dir, out_dir / f"{name}-f16.gguf", llama_cpp=llama_cpp)
    return quantize(
        f16, out_dir / f"{name}-{quant_type}.gguf", llama_cpp=llama_cpp, quant_type=quant_type
    )


def _run(command: list[str], *, cwd: Path) -> None:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-20:])
        raise RuntimeError(f"команда завершилась с кодом {result.returncode}:\n{tail}")
