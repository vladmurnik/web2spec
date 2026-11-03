"""Merge a LoRA adapter into a full-precision base model.

Never merge into a 4-bit model: NF4 weights are lossy and merging quantized
weights produces a corrupted checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path


def merge_adapter(
    base_model: str | Path,
    adapter_dir: Path,
    out_dir: Path,
    *,
    dtype: str = "float16",
) -> Path:
    """Merge ``adapter_dir`` into ``base_model`` and save a standalone model."""

    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"в {adapter_dir} нет adapter_config.json — это не папка LoRA-адаптера"
        )
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"unsupported dtype: {dtype}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=torch_dtype,
        device_map="auto" if use_cuda else "cpu",
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(str(base_model))
    tokenizer.save_pretrained(out_dir)

    weights = list(out_dir.glob("*.safetensors"))
    if not (out_dir / "config.json").exists() or not weights:
        raise RuntimeError(f"мердж неполный: проверьте {out_dir}")
    manifest = {
        "base_model": str(base_model),
        "adapter": str(adapter_dir),
        "dtype": dtype,
        "weights": [path.name for path in weights],
    }
    (out_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_dir.resolve()
