from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from web2spec.config import (
    REPO_ROOT,
    TrainConfig,
    load_data_config,
    load_eval_config,
    load_train_config,
)


def test_shipped_train_config_matches_4gb_plan():
    cfg = load_train_config()
    assert cfg.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.max_seq_len == 1536
    assert cfg.batch_size == 1
    assert cfg.grad_accum == 16
    assert cfg.effective_batch_size == 16
    assert cfg.lora.r == 16
    assert cfg.lora.alpha == 32
    assert cfg.lora.dropout == pytest.approx(0.05)
    assert cfg.optim == "paged_adamw_8bit"
    assert cfg.attn_implementation == "sdpa"
    assert cfg.gradient_checkpointing is True
    assert len(cfg.lora.target_modules) == 7


def test_shipped_data_config_is_consistent_with_training():
    data = load_data_config()
    train = load_train_config()
    assert data.max_seq_len == train.max_seq_len
    assert data.tokenizer == train.base_model


def test_eval_config_loads():
    cfg = load_eval_config()
    assert cfg.max_new_tokens == 256
    assert cfg.temperature == 0.0
    assert cfg.limit == 200


def test_relative_paths_resolve_against_repo_root():
    paths = load_data_config().paths.resolve()
    assert paths.processed.is_absolute()
    assert paths.processed == (REPO_ROOT / "data/processed").resolve()
    assert paths.gguf == (REPO_ROOT / "models/gguf").resolve()


def test_unknown_key_is_rejected(tmp_path: Path):
    bad = tmp_path / "train.yaml"
    bad.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\nlearning_rate: 0.001\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_train_config(bad)


def test_missing_config_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_data_config(tmp_path / "nope.yaml")


def test_sequence_length_bounds():
    with pytest.raises(ValidationError):
        TrainConfig(max_seq_len=99)

