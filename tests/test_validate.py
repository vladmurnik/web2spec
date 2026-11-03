from __future__ import annotations

import json
from pathlib import Path

import pytest

from web2spec.config import load_data_config
from web2spec.schema import SampleError, load_task_schemas
from web2spec.validate import (
    iter_jsonl,
    load_samples,
    normalize_record,
    split_samples,
    validate_file,
    write_jsonl,
)

HTML = (
    "<html><body><div class='product'><h1>Товар</h1>"
    "<span class='price'>1000</span><p>Описание товара</p></div></body></html>"
)


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1] * max(1, len(text.split()))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True) -> str:
        return " ".join(message["content"] for message in messages)


class BigTokenizer(FakeTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1] * 2000


def make_record(**overrides) -> dict:
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
    return record


def write_lines(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def cfg():
    return load_data_config()


@pytest.fixture()
def schemas():
    return load_task_schemas(Path("configs/tasks.yaml"))


def test_valid_sample_passes(tmp_path: Path, cfg, schemas) -> None:
    path = write_lines(tmp_path / "in.jsonl", [make_record()])
    samples, report = validate_file(
        path, cfg=cfg, task_schemas=schemas, tokenizer=FakeTokenizer()
    )
    assert len(samples) == 1
    assert report.passed == 1
    assert report.failed == 0
    assert samples[0].meta.n_tokens and samples[0].meta.n_tokens > 0


def test_invalid_json_is_reported(tmp_path: Path, cfg, schemas) -> None:
    path = tmp_path / "in.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    samples, report = validate_file(path, cfg=cfg, task_schemas=schemas)
    assert samples == []
    assert report.failed == 1
    assert report.by_reason["E_JSON"] == 1
    assert report.issues[0].line == 1


def test_short_html_rejected(tmp_path: Path, cfg, schemas) -> None:
    path = write_lines(tmp_path / "in.jsonl", [make_record(input_html="<p>x</p>")])
    _, report = validate_file(path, cfg=cfg, task_schemas=schemas)
    assert report.by_reason["E_TOO_SHORT"] == 1


def test_duplicate_rejected(tmp_path: Path, cfg, schemas) -> None:
    duplicate = make_record(id="sample-0002")
    path = write_lines(tmp_path / "in.jsonl", [make_record(), duplicate])
    samples, report = validate_file(path, cfg=cfg, task_schemas=schemas)
    assert len(samples) == 1
    assert report.by_reason["E_DUP"] == 1


def test_schema_violation_rejected(tmp_path: Path, cfg, schemas) -> None:
    path = write_lines(tmp_path / "in.jsonl", [make_record(target=[])])
    _, report = validate_file(path, cfg=cfg, task_schemas=schemas)
    assert report.by_reason["E_SCHEMA"] == 1


def test_too_long_rejected(tmp_path: Path, cfg, schemas) -> None:
    path = write_lines(tmp_path / "in.jsonl", [make_record()])
    _, report = validate_file(
        path,
        cfg=cfg,
        task_schemas=schemas,
        tokenizer=BigTokenizer(),
        max_seq_len=128,
    )
    assert report.by_reason["E_TOO_LONG"] == 1


def test_quarantine_receives_bad_lines(tmp_path: Path, cfg, schemas) -> None:
    path = write_lines(tmp_path / "in.jsonl", [make_record(input_html="<p>x</p>")])
    quarantine = tmp_path / "quarantine.jsonl"
    validate_file(path, cfg=cfg, task_schemas=schemas, quarantine=quarantine)
    assert quarantine.exists()
    assert "input_html" in quarantine.read_text(encoding="utf-8")


def test_iter_jsonl_rejects_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b"\xef\xbb\xbf{}\n")
    with pytest.raises(SampleError) as excinfo:
        list(iter_jsonl(path))
    assert excinfo.value.code == "E_ENCODING"


def test_split_is_group_aware() -> None:
    samples = []
    for group in range(10):
        for index in range(2):
            record = make_record(
                id=f"sample-{group:02d}{index}",
                group_id=f"shop{group}.ru",
                input_html=HTML + f"<!--{group}-{index}-->",
            )
            samples.append(normalize_record(record, line=group * 2 + index))
    from web2spec.schema import validate_sample

    validated = [validate_sample(record) for record in samples]
    splits = split_samples(validated, val_ratio=0.2, seed=42)
    assert splits["train"] and splits["val"]
    seen = {}
    for name, items in splits.items():
        for sample in items:
            assert seen.setdefault(sample.group_id, name) == name
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(validated)


def test_split_with_one_group_keeps_train() -> None:
    from web2spec.schema import validate_sample

    samples = [validate_sample(make_record())]
    splits = split_samples(samples, val_ratio=0.05, seed=1)
    assert len(splits["train"]) == 1
    assert splits["val"] == []


def test_split_guarantees_non_empty_train_and_val() -> None:
    from web2spec.schema import validate_sample

    samples = [
        validate_sample(make_record(id=f"sample-000{i}", group_id=f"shop{i}.ru"))
        for i in range(3)
    ]
    splits = split_samples(samples, val_ratio=0.05, seed=1)
    assert len(splits["train"]) == 2
    assert len(splits["val"]) == 1
    assert all(isinstance(sample, type(samples[0])) for sample in splits["val"])


def test_normalize_record_from_loose_fields() -> None:
    record = normalize_record(
        {"html": HTML, "source_url": "https://shop.ru/p/1", "output": {"a": 1}}, line=5
    )
    assert record["group_id"] == "shop.ru"
    assert record["target"] == {"a": 1}
    assert record["id"].startswith("extract-")


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    from web2spec.schema import validate_sample

    sample = validate_sample(make_record())
    path = write_jsonl([sample], tmp_path / "out.jsonl")
    loaded = load_samples(path)
    assert loaded[0].id == sample.id
    assert loaded[0].target == sample.target
