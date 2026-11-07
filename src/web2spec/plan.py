"""Semantic collection plans for task-conditioned scraping.

A plan answers *what* to collect for a user task. It deliberately contains no
CSS selectors: selector generation belongs to the extractor/renderer layer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from web2spec.evidence import find_evidence

PLAN_VERSION = 1
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,39}$")
MAX_FIELDS = 12
MAX_FILTERS = 8
MAX_STOP_CONDITIONS = 5
MAX_FACTS = 12


class PlanError(ValueError):
    """Invalid collection plan."""


class PlanField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=2, max_length=40)
    why: str = Field(min_length=3, max_length=200)
    required: bool = False
    kind: Literal["text", "number", "date", "url", "bool", "null"] = "text"

    @model_validator(mode="after")
    def _validate_name(self) -> PlanField:
        if not _NAME_RE.match(self.name):
            raise ValueError("field name must be snake_case")
        if self.name in {"selector", "xpath", "css"}:
            raise ValueError("field name must not describe selector mechanics")
        return self


class CollectionPlan(BaseModel):
    """A task-specific plan, independent of a concrete site implementation."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = PLAN_VERSION
    goal: str = Field(min_length=5, max_length=300)
    entity: str = Field(min_length=2, max_length=40)
    entry: Literal["listing", "detail", "search", "sitemap"] = "detail"
    fields: list[PlanField] = Field(default_factory=list, max_length=MAX_FIELDS)
    filters: dict[str, str] = Field(default_factory=dict)
    pagination: bool = False
    detail_pages: bool = False
    dedup_key: Literal["url", "id", "sku", "title"] = "url"
    stop_conditions: list[str] = Field(default_factory=list, max_length=MAX_STOP_CONDITIONS)

    @model_validator(mode="after")
    def _validate_plan(self) -> CollectionPlan:
        self.goal = self.goal.strip()
        self.entity = self.entity.strip()
        if not _NAME_RE.match(self.entity) and self.entity != "none":
            raise ValueError("entity must be snake_case or 'none'")
        if len(self.filters) > MAX_FILTERS:
            raise ValueError(f"filters must contain at most {MAX_FILTERS} items")
        for key, value in self.filters.items():
            if not _NAME_RE.match(key):
                raise ValueError(f"filter key must be snake_case: {key!r}")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"filter {key!r} must have a non-empty string value")
        if self.entity == "none":
            if self.fields:
                raise ValueError("empty plan must not contain fields")
            if self.pagination or self.detail_pages:
                raise ValueError("empty plan must not crawl pages")
            return self
        if not self.fields:
            raise ValueError("non-empty plan requires at least one field")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("field names must be unique")
        if len(self.stop_conditions) > MAX_STOP_CONDITIONS:
            raise ValueError(f"at most {MAX_STOP_CONDITIONS} stop conditions")
        for condition in self.stop_conditions:
            if not condition.strip():
                raise ValueError("stop condition must not be blank")
        return self


def plan_from_fields(
    task: str,
    fields: list[tuple[str, str] | tuple[str, str, str]],
) -> CollectionPlan:
    """Build a deterministic plan from caller-provided field definitions.

    Each item is ``(name, kind)`` or ``(name, kind, description)``.  The
    description is passed to the extractor as ``why`` and is the main way to
    disambiguate rows such as "GitHub-hosted" versus "self-hosted".  This is the
    MCP/API-friendly path: the caller already knows the schema, so the 0.5B
    planner is not asked to invent one.
    """

    if not fields:
        raise PlanError("at least one field is required")
    specs: list[dict[str, Any]] = []
    for spec in fields:
        name, kind = spec[0], spec[1]
        why = spec[2] if len(spec) > 2 and spec[2] else "requested by the caller"
        if len(why) > 200:
            raise PlanError(f"field {name!r}: description must be at most 200 characters")
        specs.append(
            {
                "name": name,
                "why": why,
                "required": True,
                "kind": kind or "text",
            }
        )
    payload = {
        "version": PLAN_VERSION,
        "goal": task.strip()[:300],
        "entity": "custom",
        "entry": "detail",
        "fields": specs,
    }
    try:
        plan = CollectionPlan.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "<root>"
        raise PlanError(f"{location}: {first['msg']}") from exc
    return plan


def load_plan(raw: Any) -> CollectionPlan:
    """Parse a plan from a dict, a wrapped ``{"plan": ...}`` dict, or JSON text."""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlanError(f"plan is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PlanError("plan must be a JSON object")
    if set(raw) == {"plan"} and isinstance(raw["plan"], dict):
        raw = raw["plan"]
    try:
        plan = CollectionPlan.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "<root>"
        raise PlanError(f"{location}: {first['msg']}") from exc
    if plan.version != PLAN_VERSION:
        raise PlanError(f"unsupported plan version: {plan.version}")
    return plan


_PLAN_KEYS = {
    "version",
    "goal",
    "entity",
    "entry",
    "fields",
    "filters",
    "pagination",
    "detail_pages",
    "dedup_key",
    "stop_conditions",
}
_KIND_ALIASES = {
    "text": "text",
    "string": "text",
    "str": "text",
    "number": "number",
    "numeric": "number",
    "integer": "number",
    "int": "number",
    "float": "number",
    "date": "date",
    "datetime": "date",
    "time": "date",
    "url": "url",
    "link": "url",
    "bool": "bool",
    "boolean": "bool",
    "null": "null",
    "none": "null",
}


def _plan_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlanError(f"plan is not JSON: {exc}") from exc
    if isinstance(raw, list) and len(raw) == 1:
        raw = raw[0]
    if not isinstance(raw, dict):
        raise PlanError("plan must be a JSON object")
    if set(raw) == {"plan"} and isinstance(raw["plan"], dict):
        raw = raw["plan"]
    return raw


def _identifier(value: Any, fallback: str) -> str:
    """Make a model-provided field/entity name satisfy the plan schema."""

    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").casefold())
    text = re.sub(r"_+", "_", text).strip("_")
    if not text or not text[0].isalpha():
        text = fallback
    return text[:40]


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "1", "yes", "да"}:
            return True
        if lowered in {"false", "0", "no", "нет"}:
            return False
    return default


def load_plan_lenient(
    raw: Any,
    *,
    task: str | None = None,
    dropped: list[dict[str, Any]] | None = None,
) -> CollectionPlan | None:
    """Recover a usable plan from common small-model schema mistakes.

    The runtime should not lose a whole page because a 0.5B model wrote
    ``"kind": "html"``, added an unknown key, or omitted ``goal``.  This
    function normalizes those recoverable cases, then still validates the
    result through :class:`CollectionPlan`.  Unrecoverable output returns
    ``None`` and is recorded in ``dropped`` instead of raising.
    """

    try:
        payload = _plan_payload(raw)
    except PlanError as exc:
        if dropped is not None:
            dropped.append({"reason": "invalid_plan", "error": str(exc)})
        return None

    normalized = {key: payload[key] for key in payload if key in _PLAN_KEYS}
    goal = str(normalized.get("goal") or task or "collect requested fields").strip()
    normalized["goal"] = goal[:300] if len(goal) >= 5 else "collect requested fields"
    normalized["version"] = 1

    fields_raw = normalized.get("fields", [])
    if isinstance(fields_raw, dict):
        fields_raw = [fields_raw]
    if not isinstance(fields_raw, list):
        fields_raw = []
    fields: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(fields_raw, 1):
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            if dropped is not None:
                dropped.append({"reason": "malformed_field", "index": index})
            continue
        raw_name = item.get("name") or item.get("field") or ""
        name = _identifier(raw_name, f"field_{index}")
        if name in {"selector", "xpath", "css"} or len(name) < 2:
            if dropped is not None:
                dropped.append({"reason": "invalid_field_name", "name": raw_name})
            continue
        if name in seen:
            if dropped is not None:
                dropped.append({"reason": "duplicate_field", "name": name})
            continue
        seen.add(name)
        why = str(item.get("why") or item.get("description") or "requested by task")
        why = " ".join(why.split())[:200]
        kind = _KIND_ALIASES.get(str(item.get("kind") or "text").casefold(), "text")
        fields.append(
            {
                "name": name,
                "why": why if len(why) >= 3 else "requested by task",
                "required": _as_bool(item.get("required")),
                "kind": kind,
            }
        )
    normalized["fields"] = fields[:MAX_FIELDS]

    entity = _identifier(normalized.get("entity") or "page", "page")
    if entity == "none" and fields:
        entity = "page"
    if entity != "none" and not normalized["fields"]:
        entity = "none"
    normalized["entity"] = entity

    entry = normalized.get("entry")
    normalized["entry"] = entry if entry in {"listing", "detail", "search", "sitemap"} else "detail"
    dedup_key = normalized.get("dedup_key")
    normalized["dedup_key"] = (
        dedup_key if dedup_key in {"url", "id", "sku", "title"} else "url"
    )
    normalized["pagination"] = _as_bool(normalized.get("pagination"))
    normalized["detail_pages"] = _as_bool(normalized.get("detail_pages"))

    filters_raw = normalized.get("filters", {})
    filters: dict[str, str] = {}
    if isinstance(filters_raw, dict):
        for key, value in list(filters_raw.items())[:MAX_FILTERS]:
            key_text = _identifier(key, "")
            if key_text and isinstance(value, str) and value.strip():
                filters[key_text] = value.strip()
    normalized["filters"] = filters

    stops_raw = normalized.get("stop_conditions", [])
    if not isinstance(stops_raw, list):
        stops_raw = []
    stops = [
        " ".join(str(item).split())[:200]
        for item in stops_raw
        if isinstance(item, str) and str(item).strip()
    ][:MAX_STOP_CONDITIONS]
    normalized["stop_conditions"] = stops

    try:
        return CollectionPlan.model_validate(normalized)
    except ValidationError as exc:
        if dropped is not None:
            dropped.append({"reason": "invalid_plan", "error": str(exc)})
        return None


def empty_plan(goal: str) -> CollectionPlan:
    """Canonical negative plan: the page has nothing relevant to the task."""

    return CollectionPlan(
        goal=goal,
        entity="none",
        entry="detail",
        fields=[],
        filters={},
        pagination=False,
        detail_pages=False,
        dedup_key="url",
        stop_conditions=["no relevant data on the page"],
    )


def plan_target(plan: CollectionPlan) -> dict[str, Any]:
    return {"plan": plan.model_dump(mode="json")}


def plan_key_set(plan: CollectionPlan) -> set[str]:
    return {field.name for field in plan.fields}


class Fact(BaseModel):
    """One extracted value with a verbatim evidence span from the input HTML."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=2, max_length=40)
    value: Any
    evidence: str = Field(min_length=1, max_length=1000)

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not _NAME_RE.match(value):
            raise ValueError("fact name must be snake_case")
        if value in {"selector", "xpath", "css"}:
            raise ValueError("fact name must not describe selector mechanics")
        return value

    @field_validator("value")
    @classmethod
    def _check_value(cls, value: Any) -> Any:
        if isinstance(value, (dict, list, tuple, set)):
            raise ValueError("fact value must be a scalar")
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise ValueError("fact value must be a string, number, boolean or null")

    @field_validator("evidence")
    @classmethod
    def _check_evidence(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("evidence must not be blank")
        return value


class FactSet(BaseModel):
    """Facts found on a page plus plan fields that are absent from it."""

    model_config = ConfigDict(extra="forbid")

    facts: list[Fact] = Field(default_factory=list, max_length=MAX_FACTS)
    missing: list[str] = Field(default_factory=list, max_length=MAX_FACTS)

    @model_validator(mode="after")
    def _validate_factset(self) -> FactSet:
        names = [fact.name for fact in self.facts]
        if len(names) != len(set(names)):
            raise ValueError("fact names must be unique")
        if len(self.missing) != len(set(self.missing)):
            raise ValueError("missing field names must be unique")
        overlap = set(names) & set(self.missing)
        if overlap:
            raise ValueError(f"fields cannot be both facts and missing: {sorted(overlap)}")
        for name in self.missing:
            if not _NAME_RE.match(name):
                raise ValueError(f"missing field name must be snake_case: {name!r}")
        return self


def _factset_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlanError(f"fact set is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PlanError("fact set must be a JSON object")
    if set(raw) == {"factset"} and isinstance(raw["factset"], dict):
        raw = raw["factset"]
    return raw


def load_factset(raw: Any) -> FactSet:
    """Parse a FactSet from a dict, a wrapped ``{"factset": ...}`` dict, or JSON."""

    raw = _factset_payload(raw)
    try:
        return FactSet.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "<root>"
        raise PlanError(f"{location}: {first['msg']}") from exc


def load_factset_lenient(
    raw: Any, dropped: list[dict[str, Any]] | None = None
) -> FactSet:
    """Parse model output while removing contradictions before validation.

    Small quantized models often put the same field in both ``facts`` and
    ``missing``, or emit a placeholder instead of a plan field.  The fact is
    authoritative; overlapping ``missing`` entries are removed.  Facts with
    invalid names are dropped and optionally recorded.  The strict
    ``validate_factset`` still checks plan coverage and exact evidence later.
    """

    if isinstance(raw, list):
        raw = {"facts": raw, "missing": []}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlanError(f"fact set is not JSON: {exc}") from exc
        if isinstance(parsed, list):
            parsed = {"facts": parsed, "missing": []}
        raw = parsed
    raw = _factset_payload(raw)
    if isinstance(raw, list):
        raw = {"facts": raw, "missing": []}
    if isinstance(raw, dict) and "facts" not in raw and "name" in raw:
        # The extractor occasionally returns a bare fact object instead of a
        # FactSet; treat it as a single-fact set.
        raw = {"facts": [raw], "missing": []}
    facts_raw = raw.get("facts", [])
    missing_raw = raw.get("missing", [])
    if not isinstance(facts_raw, list):
        facts_raw = []
    if not isinstance(missing_raw, list):
        missing_raw = []

    facts: list[dict[str, Any]] = []
    seen: set[str] = set()
    missing: list[str] = []
    for item in facts_raw:
        if not isinstance(item, dict):
            if dropped is not None:
                dropped.append({"name": None, "reason": "malformed_fact"})
            continue
        name = item.get("name")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            if dropped is not None:
                dropped.append(
                    {
                        "name": name,
                        "value": item.get("value"),
                        "evidence": item.get("evidence"),
                        "reason": "invalid_name",
                    }
                )
            continue
        if name in seen:
            if dropped is not None:
                dropped.append(
                    {
                        "name": name,
                        "value": item.get("value"),
                        "evidence": item.get("evidence"),
                        "reason": "duplicate",
                    }
                )
            continue
        seen.add(name)
        evidence = item.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            # The extractor sometimes omits evidence.  Without a verbatim span
            # the value cannot be verified, so the field becomes ``missing``.
            if dropped is not None:
                dropped.append(
                    {
                        "name": name,
                        "value": item.get("value"),
                        "evidence": evidence,
                        "reason": "missing_evidence",
                    }
                )
            if name not in missing:
                missing.append(name)
            continue
        facts.append({k: item[k] for k in ("name", "value", "evidence") if k in item})

    for name in missing_raw:
        if isinstance(name, dict):
            name = name.get("name")
        if (
            isinstance(name, str)
            and _NAME_RE.match(name)
            and name not in seen
            and name not in missing
        ):
            missing.append(name)

    try:
        return FactSet.model_validate({**raw, "facts": facts, "missing": missing})
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "<root>"
        raise PlanError(f"{location}: {first['msg']}") from exc


def fact_target(factset: FactSet) -> dict[str, Any]:
    return factset.model_dump(mode="json")


def validate_factset(factset: FactSet, plan: CollectionPlan, html: str) -> FactSet:
    """Strict contract check: coverage, plan membership and exact evidence."""

    plan_names = plan_key_set(plan)
    fact_names = {fact.name for fact in factset.facts}
    missing_names = set(factset.missing)
    unknown = (fact_names | missing_names) - plan_names
    if unknown:
        raise PlanError(f"fact set contains fields outside the plan: {sorted(unknown)}")
    uncovered = plan_names - fact_names - missing_names
    if uncovered:
        raise PlanError(f"plan fields without a fact or missing entry: {sorted(uncovered)}")
    for fact in factset.facts:
        if fact.evidence not in html:
            raise PlanError(
                f"evidence for {fact.name!r} is not an exact substring of the input HTML"
            )
    return factset


def fact_key_set(factset: FactSet) -> set[str]:
    return {fact.name for fact in factset.facts}


def repair_factset(
    factset: FactSet, plan: CollectionPlan, html: str
) -> tuple[FactSet, list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically repair model evidence, then re-validate.

    A fact whose evidence is not an exact HTML substring keeps its value only
    when ``find_evidence`` can ground it again.  Otherwise the field becomes a
    ``missing`` entry, so the final FactSet is always verifiable and never
    contains fabricated evidence.
    """

    plan_names = {field.name for field in plan.fields}
    kept: list[Fact] = []
    missing = list(factset.missing)
    repaired: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fact in factset.facts:
        if fact.name not in plan_names:
            dropped.append(
                {
                    "name": fact.name,
                    "value": fact.value,
                    "evidence": fact.evidence,
                    "reason": "outside_plan",
                }
            )
            continue
        if fact.name in seen:
            dropped.append(
                {
                    "name": fact.name,
                    "value": fact.value,
                    "evidence": fact.evidence,
                    "reason": "duplicate",
                }
            )
            continue
        seen.add(fact.name)
        if fact.evidence in html:
            kept.append(fact)
            continue
        found = find_evidence(html, fact.value, key=fact.name)
        if found is not None:
            repaired.append({"name": fact.name, "old": fact.evidence, "new": found})
            kept.append(fact.model_copy(update={"evidence": found}))
            continue
        dropped.append({"name": fact.name, "value": fact.value, "evidence": fact.evidence})
        if fact.name not in missing:
            missing.append(fact.name)
    fact_names = {fact.name for fact in kept}
    # The model sometimes lists a found field in both facts and missing.  The
    # fact is authoritative, so drop the contradictory missing entry instead of
    # rejecting an otherwise valid FactSet.
    missing = [
        name for name in missing if name in plan_names and name not in fact_names
    ]
    return FactSet(facts=kept, missing=missing), repaired, dropped
