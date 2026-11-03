"""Typed configuration loading.

Every relative path in YAML is resolved against the repository root
(``transformer/``), never against the current working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[2]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Paths(StrictModel):
    raw: Path = Path("data/raw")
    interim: Path = Path("data/interim")
    processed: Path = Path("data/processed")
    reports: Path = Path("data/reports")
    base_model: Path = Path("models/base/qwen2.5-0.5b-instruct")
    merged: Path = Path("models/merged/qwen2.5-0.5b-ft")
    gguf: Path = Path("models/gguf")
    runs: Path = Path("outputs/runs")

    def resolve(self, root: Path = REPO_ROOT) -> Paths:
        values = {
            name: (root / value if not value.is_absolute() else value).resolve()
            for name, value in self.model_dump().items()
        }
        return Paths(**values)


class DataConfig(StrictModel):
    paths: Paths = Field(default_factory=Paths)
    tokenizer: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_len: int = Field(default=1024, ge=128, le=8192)
    min_chars: int = Field(default=50, ge=1)
    max_chars: int = Field(default=200_000, ge=100)
    compress_max_chars: int = Field(default=12_000, ge=1000)
    val_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    seed: int = 42
    dedup: bool = True
    task_schemas: Path = Path("configs/tasks.yaml")

    @field_validator("max_chars")
    @classmethod
    def _check_bounds(cls, value: int, info) -> int:
        minimum = info.data.get("min_chars", 1)
        if value <= minimum:
            raise ValueError("max_chars must be greater than min_chars")
        return value


class QuantConfig(StrictModel):
    load_in_4bit: bool = True
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"


class LoraConfig(StrictModel):
    r: int = Field(default=16, ge=1, le=128)
    alpha: int = Field(default=32, ge=1, le=256)
    dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: Literal["none", "all", "lora_only"] = "none"

    @field_validator("target_modules")
    @classmethod
    def _check_modules(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("target_modules must not be empty")
        return value


class TrainConfig(StrictModel):
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: Path = Path("outputs/runs/qwen05b-qlora-v1")
    seed: int = 42
    max_seq_len: int = Field(default=1024, ge=128, le=8192)

    quant: QuantConfig = Field(default_factory=QuantConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)

    batch_size: int = Field(default=1, ge=1)
    grad_accum: int = Field(default=16, ge=1)
    epochs: int = Field(default=3, ge=1)
    lr: float = Field(default=2e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.2)
    optim: Literal[
        "paged_adamw_8bit", "adamw_8bit", "adamw_torch", "adafactor"
    ] = "paged_adamw_8bit"
    max_grad_norm: float = Field(default=1.0, gt=0)
    gradient_checkpointing: bool = True
    attn_implementation: Literal["sdpa", "eager", "flash_attention_2"] = "sdpa"

    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    eval_steps: int = Field(default=50, ge=1)
    early_stopping_patience: int = Field(default=2, ge=0)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum


class EvalConfig(StrictModel):
    model_dir: Path = Path("models/merged/qwen2.5-0.5b-ft")
    data: Path = Path("data/processed/test.jsonl")
    report: Path = Path("data/reports/eval_report.json")
    max_new_tokens: int = Field(default=256, ge=16, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    batch_size: int = Field(default=1, ge=1)
    limit: int | None = Field(default=200, ge=1)


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return payload


def load_data_config(path: Path | str | None = None) -> DataConfig:
    path = Path(path) if path else REPO_ROOT / "configs/data.yaml"
    return DataConfig.model_validate(_read_yaml(path))


def load_train_config(path: Path | str | None = None) -> TrainConfig:
    path = Path(path) if path else REPO_ROOT / "configs/train_qlora_4gb.yaml"
    return TrainConfig.model_validate(_read_yaml(path))


def load_eval_config(path: Path | str | None = None) -> EvalConfig:
    path = Path(path) if path else REPO_ROOT / "configs/eval.yaml"
    return EvalConfig.model_validate(_read_yaml(path))
