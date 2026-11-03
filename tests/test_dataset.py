from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from web2spec.config import REPO_ROOT
from web2spec.dataset import ExtractionDataset
from web2spec.schema import validate_sample

HTML = (
    "<html><body><div class='product'><h1>Товар</h1>"
    "<span class='price'>1000</span><p>Описание товара</p></div></body></html>"
)


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True) -> str:
        return " ".join(message["content"] for message in messages)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [5] * max(1, len(text.split()))}


def make_sample(**overrides):
    record = {
        "id": "sample-0001",
        "group_id": "shop.ru",
        "task": "extract",
        "instruction": "Извлеки товар",
        "input_html": HTML,
        "target": {"title": "Товар", "price": 1000},
        "meta": {},
    }
    record.update(overrides)
    return validate_sample(record)


def test_prompt_tokens_are_masked() -> None:
    dataset = ExtractionDataset([make_sample()], FakeTokenizer(), max_seq_len=512)
    item = dataset[0]
    labels = item["labels"]
    first_target = next(index for index, value in enumerate(labels) if value != -100)
    assert first_target > 0
    assert all(value == -100 for value in labels[:first_target])
    assert all(value != -100 for value in labels[first_target:])


def test_answer_tokens_are_kept() -> None:
    dataset = ExtractionDataset([make_sample()], FakeTokenizer(), max_seq_len=512)
    item = dataset[0]
    kept = [value for value in item["labels"] if value != -100]
    assert len(kept) > 1
    assert kept[-1] == FakeTokenizer.eos_token_id


def test_truncation_respects_max_seq_len() -> None:
    sample = make_sample(input_html=HTML + "<p>" + "текст " * 500 + "</p>")
    dataset = ExtractionDataset([sample], FakeTokenizer(), max_seq_len=64)
    item = dataset[0]
    assert len(item["input_ids"]) == 64
    assert len(item["attention_mask"]) == 64
    assert len(item["labels"]) == 64
    assert item["labels"][-1] != -100


def test_short_max_seq_len_rejected() -> None:
    with pytest.raises(ValueError):
        ExtractionDataset([make_sample()], FakeTokenizer(), max_seq_len=32)


def test_dataset_loads_jsonl(tmp_path: Path) -> None:
    import json

    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(make_sample().model_dump(), ensure_ascii=False) + "\n", "utf-8")
    dataset = ExtractionDataset(path, FakeTokenizer(), max_seq_len=256)
    assert len(dataset) == 1
    assert dataset[0]["input_ids"]


def test_collator_pads_and_masks() -> None:
    torch = pytest.importorskip("torch")
    from web2spec.dataset import CausalLMCollator

    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5]},
    ]
    batch = CausalLMCollator(pad_token_id=0, pad_to_multiple_of=8)(features)
    assert batch["input_ids"].shape == (2, 8)
    assert torch.equal(batch["attention_mask"][1], torch.tensor([1, 1, 0, 0, 0, 0, 0, 0]))
    assert torch.equal(batch["labels"][1], torch.tensor([4, 5, -100, -100, -100, -100, -100, -100]))


def test_dataset_module_does_not_import_torch() -> None:
    code = "import sys, web2spec.dataset; assert 'torch' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
