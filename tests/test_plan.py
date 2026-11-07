from __future__ import annotations

from pathlib import Path

import pytest

from web2spec.evidence import find_evidence
from web2spec.plan import (
    CollectionPlan,
    Fact,
    FactSet,
    PlanError,
    empty_plan,
    fact_target,
    load_factset,
    load_factset_lenient,
    load_plan,
    load_plan_lenient,
    plan_from_fields,
    plan_target,
    repair_factset,
    validate_factset,
)
from web2spec.schema import SampleError, load_task_schemas, validate_sample
from web2spec.tokens import (
    PLAN_EXTRACT_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    build_plan_extract_prompt,
    build_plan_prompt,
)


def _plan_dict() -> dict:
    return {
        "version": 1,
        "goal": "Сравнить цены на видеокарты",
        "entity": "product",
        "entry": "listing",
        "fields": [
            {
                "name": "model",
                "why": "сопоставление товаров",
                "required": True,
                "kind": "text",
            },
            {
                "name": "price",
                "why": "сравнение цен",
                "required": True,
                "kind": "number",
            },
        ],
        "filters": {"model": "RTX 5060 Ti"},
        "pagination": True,
        "detail_pages": True,
        "dedup_key": "url",
        "stop_conditions": ["нет следующей страницы"],
    }


def _sample(target: dict) -> dict:
    return {
        "id": "plan-00000001",
        "group_id": "example.ru",
        "task": "plan",
        "instruction": "Сравни цены на видеокарты",
        "input_html": "<main><h1>Видеокарта</h1><span>100 ₽</span></main>",
        "target": target,
        "meta": {"lang": "ru", "complexity": 2, "source_url": "https://example.ru"},
    }


def test_plan_from_fields_preserves_names_and_kinds() -> None:
    plan = plan_from_fields(
        "Collect price and stock",
        [
            ("price", "number", "price in RUB without discounts"),
            ("in_stock", "bool"),
        ],
    )
    assert [field.name for field in plan.fields] == ["price", "in_stock"]
    assert [field.kind for field in plan.fields] == ["number", "bool"]
    assert plan.fields[0].why == "price in RUB without discounts"
    assert plan.fields[1].why == "requested by the caller"
    assert all(field.required for field in plan.fields)
    assert plan.entity == "custom"


def test_plan_from_fields_rejects_empty_and_bad_names() -> None:
    with pytest.raises(PlanError):
        plan_from_fields("Collect", [])
    with pytest.raises(PlanError):
        plan_from_fields("Collect", [("Bad Name", "text")])


def test_valid_plan() -> None:
    plan = load_plan(_plan_dict())
    assert plan.entity == "product"
    assert [field.name for field in plan.fields] == ["model", "price"]
    assert plan.pagination is True


def test_wrapped_plan() -> None:
    plan = load_plan({"plan": _plan_dict()})
    assert plan.goal == "Сравнить цены на видеокарты"


def test_empty_plan() -> None:
    plan = empty_plan("Найти вакансии")
    assert plan.entity == "none"
    assert plan.fields == []
    assert plan.pagination is False
    assert plan_target(plan)["plan"]["entity"] == "none"


def test_empty_plan_rejects_fields() -> None:
    raw = _plan_dict()
    raw["entity"] = "none"
    with pytest.raises(PlanError):
        load_plan(raw)


def test_duplicate_fields_rejected() -> None:
    raw = _plan_dict()
    raw["fields"].append(raw["fields"][0])
    with pytest.raises(PlanError):
        load_plan(raw)


def test_selector_in_field_name_rejected() -> None:
    raw = _plan_dict()
    raw["fields"][0]["name"] = "selector"
    with pytest.raises(PlanError):
        load_plan(raw)


def test_bad_entity_rejected() -> None:
    raw = _plan_dict()
    raw["entity"] = "Product Card"
    with pytest.raises(PlanError):
        load_plan(raw)


def test_bad_kind_rejected() -> None:
    raw = _plan_dict()
    raw["fields"][0]["kind"] = "html"
    with pytest.raises(PlanError):
        load_plan(raw)


def test_too_many_fields_rejected() -> None:
    raw = _plan_dict()
    raw["fields"] = [
        {"name": f"field_{i}", "why": "test", "required": False, "kind": "text"} for i in range(13)
    ]
    with pytest.raises(PlanError):
        load_plan(raw)


def test_sample_accepts_plan_schema() -> None:
    schemas = load_task_schemas(Path("configs/tasks.yaml"))
    sample = validate_sample(_sample(plan_target(load_plan(_plan_dict()))), schemas)
    assert sample.task == "plan"


def test_sample_rejects_empty_non_none_plan() -> None:
    schemas = load_task_schemas(Path("configs/tasks.yaml"))
    raw = _plan_dict()
    raw["fields"] = []
    with pytest.raises(SampleError):
        validate_sample(_sample({"plan": raw}), schemas)


def test_plan_prompt_uses_planner_system_prompt() -> None:
    schemas = load_task_schemas(Path("configs/tasks.yaml"))
    sample = validate_sample(_sample(plan_target(load_plan(_plan_dict()))), schemas)
    prompt = build_plan_prompt(sample, tokenizer=None)
    assert PLAN_SYSTEM_PROMPT in prompt
    assert "Сравни цены" in prompt
    assert "HTML-фрагмент" in prompt


def test_plan_roundtrip_model_dump() -> None:
    plan = CollectionPlan.model_validate(_plan_dict())
    assert plan.version == 1
    assert plan.model_dump()["dedup_key"] == "url"


def _fact_sample(target: dict, plan: dict) -> dict:
    return {
        "id": "pex-00000001",
        "group_id": "example.ru",
        "task": "plan_extract",
        "instruction": "Сравни цены на видеокарты",
        "input_html": "<main><h1>RTX 5060 Ti</h1><span>89 560 ₽</span></main>",
        "target": target,
        "meta": {
            "lang": "ru",
            "complexity": 2,
            "source_url": "https://example.ru",
            "plan": plan,
        },
    }


def test_factset_roundtrip() -> None:
    factset = load_factset(
        {
            "facts": [{"name": "price", "value": 89560, "evidence": "89 560"}],
            "missing": ["availability"],
        }
    )
    assert factset.facts[0].name == "price"
    assert fact_target(factset)["missing"] == ["availability"]


def test_factset_rejects_overlap() -> None:
    with pytest.raises(PlanError):
        load_factset(
            {
                "facts": [{"name": "price", "value": 1, "evidence": "1"}],
                "missing": ["price"],
            }
        )


def test_load_factset_lenient_removes_overlap_and_placeholder() -> None:
    dropped: list[dict] = []
    factset = load_factset_lenient(
        {
            "facts": [
                {"name": "price", "value": 1, "evidence": "1"},
                {"name": "поле_плана", "value": "x", "evidence": "x"},
            ],
            "missing": ["price", "model"],
        },
        dropped,
    )
    assert [fact.name for fact in factset.facts] == ["price"]
    assert factset.missing == ["model"]
    assert dropped[0]["reason"] == "invalid_name"


def test_repair_factset_drops_field_outside_plan() -> None:
    plan = load_plan(_plan_dict())
    html = "<span>89 560</span>"
    factset = FactSet(
        facts=[Fact(name="unknown_field", value="x", evidence="x")],
        missing=[],
    )
    repaired, changes, dropped = repair_factset(factset, plan, html)
    assert changes == []
    assert dropped[0]["reason"] == "outside_plan"
    assert repaired.facts == []


def test_load_plan_lenient_recovers_small_model_mistakes() -> None:
    dropped: list[dict] = []
    plan = load_plan_lenient(
        {
            "goal": "Собрать цены",
            "entity": "none",
            "entry": "catalog",
            "dedup_key": "slug",
            "mindset": "extra",
            "fields": [
                {"name": "Price USD", "kind": "currency", "required": "yes"},
                {"name": "цена", "kind": "string"},
            ],
        },
        task="найди цены",
        dropped=dropped,
    )
    assert plan is not None
    assert plan.entity == "page"
    assert plan.entry == "detail"
    assert plan.dedup_key == "url"
    assert [field.name for field in plan.fields] == ["price_usd", "field_2"]
    assert plan.fields[0].kind == "text"
    assert plan.fields[0].required is True
    assert dropped == []


def test_load_plan_lenient_fills_missing_goal_and_accepts_wrappers() -> None:
    plan = load_plan_lenient(
        [{"fields": [{"name": "title"}]}],
        task="найди заголовки статей",
    )
    assert plan is not None
    assert plan.goal == "найди заголовки статей"
    assert [field.name for field in plan.fields] == ["title"]
    wrapped = load_plan_lenient(
        {"plan": {"goal": "Собрать заголовки", "fields": [{"name": "title"}]}}
    )
    assert wrapped is not None
    assert wrapped.goal == "Собрать заголовки"


def test_load_plan_lenient_returns_none_for_unrecoverable_json() -> None:
    dropped: list[dict] = []
    assert load_plan_lenient("not-json", dropped=dropped) is None
    assert dropped[0]["reason"] == "invalid_plan"


def test_factset_rejects_duplicate_facts() -> None:
    with pytest.raises(PlanError):
        load_factset(
            {
                "facts": [
                    {"name": "price", "value": 1, "evidence": "1"},
                    {"name": "price", "value": 2, "evidence": "2"},
                ],
                "missing": [],
            }
        )


def test_validate_factset_requires_coverage() -> None:
    plan = load_plan(_plan_dict())
    factset = load_factset(
        {"facts": [{"name": "model", "value": "RTX", "evidence": "RTX"}], "missing": []}
    )
    with pytest.raises(PlanError):
        validate_factset(factset, plan, "<span>RTX</span>")


def test_validate_factset_rejects_foreign_field() -> None:
    plan = load_plan(_plan_dict())
    factset = load_factset(
        {
            "facts": [
                {"name": "model", "value": "RTX", "evidence": "RTX"},
                {"name": "price", "value": 1, "evidence": "1"},
                {"name": "other", "value": "x", "evidence": "x"},
            ],
            "missing": [],
        }
    )
    with pytest.raises(PlanError):
        validate_factset(factset, plan, "<span>RTX 1 x</span>")


def test_validate_factset_rejects_bad_evidence() -> None:
    plan = load_plan(_plan_dict())
    factset = load_factset(
        {
            "facts": [
                {"name": "model", "value": "RTX", "evidence": "not-on-page"},
                {"name": "price", "value": 1, "evidence": "1"},
            ],
            "missing": [],
        }
    )
    with pytest.raises(PlanError):
        validate_factset(factset, plan, "<span>RTX 1</span>")


def test_find_evidence_number_formatting() -> None:
    assert find_evidence("<span>89 560 ₽</span>", 89560) == "89 560"
    assert find_evidence("<span>89&nbsp;560 ₽</span>", 89560) == "89&nbsp;560"


def test_find_evidence_date_formatting() -> None:
    html = "<time>3 ноября 2026, 17:57</time>"
    assert find_evidence(html, "2026-11-03T17:57:17+03:00") == "3 ноября 2026"
    assert find_evidence(html, "2026-11-03", key="published_at") == "3 ноября 2026"


def test_find_evidence_currency_and_availability() -> None:
    assert find_evidence("<span>89 560 ₽</span>", "RUB", key="currency") == "₽"
    assert find_evidence("<div>В наличии</div>", "in_stock", key="availability") == "В наличии"


def test_find_evidence_description_prefix() -> None:
    text = "Очень длинное описание товара, которое не помещается в сжатый HTML целиком"
    html = "<p>Очень длинное описание товара, которое не помещается в сжатый HTML...</p>"
    found = find_evidence(html, text, key="description")
    assert found is not None and found in html


def test_sample_accepts_plan_extract_schema() -> None:
    schemas = load_task_schemas(Path("configs/tasks.yaml"))
    plan = load_plan(_plan_dict())
    target = fact_target(
        FactSet(
            facts=[Fact(name="model", value="RTX", evidence="RTX 5060 Ti")],
            missing=["price"],
        )
    )
    # The helper HTML contains RTX, so the evidence is valid.
    sample = validate_sample(
        _fact_sample(target, plan.model_dump(mode="json")),
        schemas,
    )
    assert sample.task == "plan_extract"
    assert sample.meta.plan is not None


def test_plan_extract_prompt_uses_plan() -> None:
    schemas = load_task_schemas(Path("configs/tasks.yaml"))
    plan = load_plan(_plan_dict())
    target = fact_target(
        FactSet(
            facts=[Fact(name="model", value="RTX", evidence="RTX 5060 Ti")],
            missing=["price"],
        )
    )
    sample = validate_sample(_fact_sample(target, plan.model_dump(mode="json")), schemas)
    prompt = build_plan_extract_prompt(sample, tokenizer=None)
    assert PLAN_EXTRACT_SYSTEM_PROMPT in prompt
    assert '"entity":"product"' in prompt
    assert "HTML:" in prompt


def test_repair_factset_regrounds_value() -> None:
    plan = load_plan(_plan_dict())
    html = "<span>89 560 ₽</span>"
    factset = FactSet(
        facts=[Fact(name="price", value=89560, evidence="89560")],
        missing=["model"],
    )
    repaired, changes, dropped = repair_factset(factset, plan, html)
    assert changes and dropped == []
    assert repaired.facts[0].evidence in html
    validate_factset(repaired, plan, html)


def test_repair_factset_moves_ungrounded_fact_to_missing() -> None:
    plan = load_plan(_plan_dict())
    html = "<span>RTX 5060 Ti</span>"
    factset = FactSet(
        facts=[
            Fact(name="model", value="RTX", evidence="RTX 5060 Ti"),
            Fact(name="price", value=1, evidence="not-on-page"),
        ],
        missing=[],
    )
    repaired, changes, dropped = repair_factset(factset, plan, html)
    assert changes == []
    assert [row["name"] for row in dropped] == ["price"]
    assert repaired.missing == ["price"]
    assert [fact.name for fact in repaired.facts] == ["model"]
    validate_factset(repaired, plan, html)
