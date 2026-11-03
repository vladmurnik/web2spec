"""Validation pipeline: JSONL in, clean splits plus a report out."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from web2spec.config import DataConfig
from web2spec.schema import (
    Sample,
    SampleError,
    ValidationReport,
    make_id,
    validate_sample,
)
from web2spec.tokens import build_prompt, count_tokens, target_text

_JSONL_SUFFIXES = {".jsonl", ".json"}


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, object)``; raise :class:`SampleError` on bad lines."""

    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise SampleError("E_ENCODING", "file starts with a UTF-8 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SampleError("E_ENCODING", f"not valid UTF-8: {exc}") from exc
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SampleError("E_JSON", str(exc), line=line_no) from exc
        if not isinstance(payload, dict):
            raise SampleError("E_NOT_OBJECT", "line is not a JSON object", line=line_no)
        yield line_no, payload


def normalize_record(
    raw: dict[str, Any], *, line: int, default_task: str = "extract"
) -> dict[str, Any]:
    """Map loose scraper output onto the canonical :class:`Sample` shape."""

    html = raw.get("input_html") or raw.get("html") or raw.get("page")
    instruction = raw.get("instruction")
    target = raw.get("target")
    if target is None and "output" in raw:
        target = raw["output"]
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = {
            "extract": "Извлеки товары и поля страницы в JSON.",
            "spec": "Опиши структуру страницы в JSON-спецификации.",
            "qa": "Ответь на вопрос по содержимому страницы в JSON.",
        }.get(default_task, "Извлеки данные страницы в JSON.")
    task = raw.get("task") or default_task
    source_url = raw.get("source_url") or (raw.get("meta") or {}).get("source_url")
    group_id = raw.get("group_id") or _host(source_url) or f"line-{line}"
    sample_id = raw.get("id") or make_id(str(task), group_id, line, html)
    meta = dict(raw.get("meta") or {})
    if source_url and "source_url" not in meta:
        meta["source_url"] = source_url
    return {
        "id": sample_id,
        "group_id": group_id,
        "task": task,
        "instruction": instruction,
        "input_html": html,
        "target": target,
        "meta": meta,
    }


def _host(url: str | None) -> str | None:
    if not url:
        return None
    without_scheme = url.split("://", 1)[-1]
    host = without_scheme.split("/", 1)[0].split(":", 1)[0]
    return host or None


def validate_file(
    input_path: Path,
    *,
    cfg: DataConfig,
    task_schemas: dict[str, dict],
    tokenizer: Any | None = None,
    max_seq_len: int | None = None,
    dedup: bool = True,
    quarantine: Path | None = None,
) -> tuple[list[Sample], ValidationReport]:
    """Validate one JSONL file and return accepted samples with a report."""

    started = time.perf_counter()
    report = ValidationReport()
    samples: list[Sample] = []
    seen: dict[str, int] = {}
    limit = max_seq_len or cfg.max_seq_len
    lengths: list[int] = []

    raw_text = input_path.read_text(encoding="utf-8-sig")
    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        report.total += 1
        try:
            sample = _validate_line(
                line,
                line_no=line_no,
                cfg=cfg,
                task_schemas=task_schemas,
                tokenizer=tokenizer,
                max_seq_len=limit,
                seen=seen,
                dedup=dedup,
            )
        except SampleError as exc:
            report.record_failure(exc)
            if quarantine is not None:
                _append_quarantine(quarantine, line)
            continue
        samples.append(sample)
        report.passed += 1
        if sample.meta.n_tokens is not None:
            lengths.append(sample.meta.n_tokens)

    report.finalize(started, lengths)
    return samples, report


def _validate_line(
    line: str,
    *,
    line_no: int,
    cfg: DataConfig,
    task_schemas: dict[str, dict],
    tokenizer: Any | None,
    max_seq_len: int,
    seen: dict[str, int],
    dedup: bool,
) -> Sample:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SampleError("E_JSON", str(exc), line=line_no) from exc

    record = normalize_record(payload, line=line_no)
    html = record["input_html"]
    if html is None:
        raise SampleError("E_SCHEMA", "input_html is missing", line=line_no)
    if not isinstance(html, str):
        raise SampleError("E_SCHEMA", "input_html must be a string", line=line_no)
    if len(html) < cfg.min_chars:
        raise SampleError(
            "E_TOO_SHORT", f"input_html has {len(html)} chars < {cfg.min_chars}", line=line_no
        )
    if len(html) > cfg.max_chars:
        raise SampleError(
            "E_TOO_LARGE", f"input_html has {len(html)} chars > {cfg.max_chars}", line=line_no
        )

    try:
        sample = validate_sample(record, task_schemas)
    except SampleError as exc:
        raise SampleError(exc.code, exc.message, line=line_no) from exc

    if tokenizer is not None:
        total = count_tokens(build_prompt(sample, tokenizer), tokenizer) + count_tokens(
            target_text(sample), tokenizer
        )
        sample.meta.n_tokens = total
        if total > max_seq_len:
            raise SampleError(
                "E_TOO_LONG", f"{total} tokens > max_seq_len {max_seq_len}", line=line_no
            )

    if dedup:
        digest = _dedup_key(sample)
        if digest in seen:
            raise SampleError(
                "E_DUP", f"duplicate of line {seen[digest]}", line=line_no
            )
        seen[digest] = line_no
    return sample


def _dedup_key(sample: Sample) -> str:
    canonical = json.dumps(sample.target, sort_keys=True, ensure_ascii=False)
    payload = "\x1f".join((sample.group_id, sample.input_html, canonical))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _append_quarantine(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")


def split_samples(
    samples: list[Sample], *, val_ratio: float, seed: int
) -> dict[str, list[Sample]]:
    """Group-aware split: all samples from one site stay in one split."""

    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    total = len(samples)
    val_target = int(round(total * val_ratio))
    test_target = int(round(total * 0.1)) if val_ratio > 0.1 else 0
    splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}

    if len(keys) < 2:
        splits["train"] = list(samples)
        return splits

    for key in keys:
        bucket = groups[key]
        if test_target and len(splits["test"]) < test_target and len(splits["train"]) == 0:
            splits["test"].extend(bucket)
        elif len(splits["val"]) < val_target and len(splits["train"]) == 0:
            splits["val"].extend(bucket)
        else:
            splits["train"].extend(bucket)

    if not splits["train"]:
        moved = splits["val"].pop()
        splits["train"].append(moved)
    if not splits["val"]:
        moved = splits["train"].pop()
        splits["val"].append(moved)
    return splits


def write_jsonl(samples: list[Sample], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return path


def load_samples(path: Path) -> list[Sample]:
    return [sample for _, sample in _iter_samples(path)]


def _iter_samples(path: Path) -> Iterator[tuple[int, Sample]]:
    for line_no, payload in iter_jsonl(path):
        yield line_no, validate_sample(normalize_record(payload, line=line_no))
