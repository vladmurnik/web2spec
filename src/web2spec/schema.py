"""Data contracts for the web2spec pipeline.

One training sample = one JSONL line: a compressed HTML document, an
instruction, and the expected structured answer.

The module is intentionally dependency-free apart from pydantic/jsonschema/yaml
so that validation can run before the ML stack is installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Literal

import jsonschema
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

TaskName = Literal["extract", "spec", "qa", "plan", "plan_extract"]
TASKS: tuple[str, ...] = ("extract", "spec", "qa", "plan", "plan_extract")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,63}$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
MAX_TARGET_CHARS = 8192
MAX_TARGET_DEPTH = 6
MAX_ISSUES = 200


class SampleError(ValueError):
    """A rejected sample with a machine-readable reason code."""

    def __init__(self, code: str, message: str, *, line: int | None = None) -> None:
        self.code = code
        self.message = message
        self.line = line
        prefix = f"line {line}: " if line is not None else ""
        super().__init__(f"{prefix}[{code}] {message}")


class Meta(BaseModel):
    """Non-supervised metadata used for filtering, slicing and reporting."""

    model_config = ConfigDict(extra="forbid")

    lang: str = "ru"
    complexity: int = Field(default=1, ge=1, le=3)
    source_url: str | None = None
    n_tokens: int | None = Field(default=None, ge=0)
    pipeline_version: str = "0.1.0"
    plan: dict[str, Any] | None = None


class Sample(BaseModel):
    """A validated training example."""

    model_config = ConfigDict(extra="forbid")

    id: str
    group_id: str = Field(min_length=1, max_length=128)
    task: TaskName
    instruction: str = Field(min_length=5, max_length=500)
    input_html: str = Field(min_length=50, max_length=200_000)
    target: Any
    meta: Meta = Field(default_factory=Meta)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError("id must be 8..64 chars of [A-Za-z0-9_.:-] and start alphanumeric")
        return value

    @field_validator("group_id")
    @classmethod
    def _check_group(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("group_id must not be blank")
        return value

    @field_validator("instruction")
    @classmethod
    def _check_instruction(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("instruction must not be blank")
        return value

    @field_validator("input_html")
    @classmethod
    def _check_html(cls, value: str) -> str:
        if _CONTROL_RE.search(value):
            raise ValueError("input_html contains control characters")
        if "\ufeff" in value:
            raise ValueError("input_html contains a BOM")
        if "<" not in value:
            raise ValueError("input_html does not look like HTML (no '<')")
        return value

    @field_validator("target")
    @classmethod
    def _check_target(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("target is null")
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"target is not JSON-serializable: {exc}") from exc
        if encoded in ("{}", "[]", '""', "null"):
            raise ValueError("target must not be empty")
        if len(encoded) > MAX_TARGET_CHARS:
            raise ValueError(f"target too large: {len(encoded)} > {MAX_TARGET_CHARS} chars")
        if _depth(value) > MAX_TARGET_DEPTH:
            raise ValueError(f"target nested deeper than {MAX_TARGET_DEPTH}")
        return value


def _depth(value: Any, level: int = 0) -> int:
    if isinstance(value, dict) and value:
        return max(_depth(item, level + 1) for item in value.values())
    if isinstance(value, list) and value:
        return max(_depth(item, level + 1) for item in value)
    return level


def validate_sample(raw: Any, task_schemas: dict[str, dict] | None = None) -> Sample:
    """Validate one decoded JSON object; raise :class:`SampleError` on failure."""

    if not isinstance(raw, dict):
        raise SampleError("E_NOT_OBJECT", f"expected object, got {type(raw).__name__}")
    try:
        sample = Sample.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "<root>"
        raise SampleError("E_SCHEMA", f"{location}: {first['msg']}") from exc

    schema = (task_schemas or {}).get(sample.task)
    if schema is not None:
        try:
            jsonschema.validate(sample.target, schema)
        except jsonschema.ValidationError as exc:
            raise SampleError("E_TARGET", f"{exc.json_path}: {exc.message}") from exc
    return sample


def make_id(prefix: str, *parts: object) -> str:
    """Deterministic, filesystem-safe id from arbitrary parts."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=12).hexdigest()
    return f"{prefix}-{digest}"


class ValidationIssue(BaseModel):
    code: str
    message: str
    line: int | None = None
    sample_id: str | None = None


class ValidationReport(BaseModel):
    """Aggregate result of a validation run; serialised to data/reports."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    by_reason: dict[str, int] = Field(default_factory=dict)
    token_p50: int | None = None
    token_p95: int | None = None
    token_p99: int | None = None
    token_max: int | None = None
    duration_s: float = 0.0
    issues: list[ValidationIssue] = Field(default_factory=list)

    def record_failure(self, error: SampleError, *, sample_id: str | None = None) -> None:
        self.failed += 1
        self.by_reason[error.code] = self.by_reason.get(error.code, 0) + 1
        if len(self.issues) < MAX_ISSUES:
            self.issues.append(
                ValidationIssue(
                    code=error.code,
                    message=error.message,
                    line=error.line,
                    sample_id=sample_id,
                )
            )

    def finalize(self, started_at: float, lengths: list[int]) -> ValidationReport:
        self.duration_s = round(time.perf_counter() - started_at, 3)
        if lengths:
            ordered = sorted(lengths)
            self.token_p50 = _percentile(ordered, 50)
            self.token_p95 = _percentile(ordered, 95)
            self.token_p99 = _percentile(ordered, 99)
            self.token_max = ordered[-1]
        return self

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


def _percentile(ordered: list[int], q: float) -> int:
    """Nearest-rank percentile over an already sorted list."""

    if not ordered:
        raise ValueError("empty list")
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def length_stats(lengths: list[int]) -> dict[str, int | None]:
    if not lengths:
        return {"p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(lengths)
    return {
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "max": ordered[-1],
    }


def load_task_schemas(path: Path) -> dict[str, dict]:
    """Load {task: json-schema} from YAML; unknown tasks are allowed but unused."""

    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a mapping of task -> JSON schema")
    schemas: dict[str, dict] = {}
    for task, schema in payload.items():
        if task not in TASKS:
            raise ValueError(f"{path}: unknown task {task!r}")
        if not isinstance(schema, dict):
            raise ValueError(f"{path}: schema for {task!r} must be a mapping")
        schemas[str(task)] = schema
    return schemas
