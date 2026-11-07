from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from web2spec.plan import plan_from_fields

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "21_collect_urls.py"
_spec = importlib.util.spec_from_file_location("collect_urls_tasks", SCRIPT)
assert _spec is not None and _spec.loader is not None
collect_urls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect_urls)


@dataclass(frozen=True)
class Scenario:
    """One reproducible task over a small synthetic page."""

    id: str
    url: str
    task: str
    html: str
    fields: tuple[tuple[str, str, str], ...]
    facts: tuple[dict[str, Any], ...]
    expected: dict[str, Any]


SCENARIOS = (
    Scenario(
        id="pricing",
        url="https://saas.example/pricing",
        task="Collect monthly prices for every published plan",
        html="""<html><body>
            <h1>Simple pricing</h1>
            <p>Free plan: $0 per month.</p>
            <p>Pro plan: $20 per month.</p>
            <p>Team plan: $99 per month.</p>
        </body></html>""",
        fields=(
            ("free_monthly_usd", "number", "Free plan monthly price in USD"),
            ("pro_monthly_usd", "number", "Pro plan monthly price in USD"),
            ("team_monthly_usd", "number", "Team plan monthly price in USD"),
        ),
        facts=(
            {"name": "free_monthly_usd", "value": 0, "evidence": "$0 per month"},
            {"name": "pro_monthly_usd", "value": 20, "evidence": "$20 per month"},
            {"name": "team_monthly_usd", "value": 99, "evidence": "$99 per month"},
        ),
        expected={"free_monthly_usd": 0, "pro_monthly_usd": 20, "team_monthly_usd": 99},
    ),
    Scenario(
        id="api-errors",
        url="https://api.example/docs/errors",
        task="Collect the HTTP error reference and retry guidance",
        html="""<html><body>
            <h1>API error reference</h1>
            <table>
              <tr><td>401</td><td>Unauthorized: missing or invalid API key.</td></tr>
              <tr><td>429</td><td>Too Many Requests: rate limit exceeded.</td></tr>
            </table>
            <p>Retry after a delay with exponential backoff.</p>
        </body></html>""",
        fields=(
            ("unauthorized_http_status", "number", "HTTP status for an invalid API key"),
            ("rate_limit_http_status", "number", "HTTP status for rate limiting"),
            ("retry_guidance", "text", "Recommended retry strategy"),
        ),
        facts=(
            {
                "name": "unauthorized_http_status",
                "value": 401,
                "evidence": "401",
            },
            {"name": "rate_limit_http_status", "value": 429, "evidence": "429"},
            {
                "name": "retry_guidance",
                "value": "exponential backoff",
                "evidence": "exponential backoff",
            },
        ),
        expected={
            "unauthorized_http_status": 401,
            "rate_limit_http_status": 429,
            "retry_guidance": "exponential backoff",
        },
    ),
    Scenario(
        id="product-specs",
        url="https://shop.example/laptop/specs",
        task="Collect the chip, display size and battery life",
        html="""<html><body>
            <h1>MacBook Air 13</h1>
            <ul>
              <li>Chip: Apple M5</li>
              <li>Display: 13.6-inch</li>
              <li>Battery: up to 18 hours of video playback</li>
            </ul>
        </body></html>""",
        fields=(
            ("chip", "text", "Processor or chip name"),
            ("display_size_inches", "number", "Display size in inches"),
            ("battery_video_playback_hours", "number", "Video playback battery life"),
        ),
        facts=(
            {"name": "chip", "value": "Apple M5", "evidence": "Apple M5"},
            {
                "name": "display_size_inches",
                "value": 13.6,
                "evidence": "13.6-inch",
            },
            {
                "name": "battery_video_playback_hours",
                "value": 18,
                "evidence": "18 hours",
            },
        ),
        expected={
            "chip": "Apple M5",
            "display_size_inches": 13.6,
            "battery_video_playback_hours": 18,
        },
    ),
    Scenario(
        id="release-notes",
        url="https://app.example/releases/2.4.0",
        task="Collect release version, date, breaking change and notes URL",
        html="""<html><body>
            <h1>Release 2.4.0</h1>
            <p>Released on <time datetime="2026-10-14">14 October 2026</time>.</p>
            <p>Breaking change: yes.</p>
            <a href="https://example.com/releases/2.4.0">
              https://example.com/releases/2.4.0
            </a>
        </body></html>""",
        fields=(
            ("release_version", "text", "Released product version"),
            ("released_on", "date", "Release date"),
            ("breaking_change", "bool", "Whether the release has breaking changes"),
            ("release_notes_url", "url", "Canonical release notes URL"),
        ),
        facts=(
            {"name": "release_version", "value": "2.4.0", "evidence": "2.4.0"},
            {"name": "released_on", "value": "2026-10-14", "evidence": "2026-10-14"},
            {"name": "breaking_change", "value": True, "evidence": "yes"},
            {
                "name": "release_notes_url",
                "value": "https://example.com/releases/2.4.0",
                "evidence": "https://example.com/releases/2.4.0",
            },
        ),
        expected={
            "release_version": "2.4.0",
            "released_on": "2026-10-14",
            "breaking_change": True,
            "release_notes_url": "https://example.com/releases/2.4.0",
        },
    ),
    Scenario(
        id="job-posting",
        url="https://jobs.example/senior-python",
        task="Collect salary, remote policy and location from the job page",
        html="""<html><body>
            <h1>Senior Python Engineer</h1>
            <p>Location: Berlin, Germany</p>
            <p>Salary: $120,000 per year</p>
            <p>Remote: yes</p>
        </body></html>""",
        fields=(
            ("salary_min_usd", "number", "Minimum annual salary in USD"),
            ("remote", "bool", "Whether the role is remote"),
            ("location", "text", "Primary office location"),
        ),
        facts=(
            {"name": "salary_min_usd", "value": 120000, "evidence": "$120,000"},
            {"name": "remote", "value": True, "evidence": "yes"},
            {"name": "location", "value": "Berlin, Germany", "evidence": "Berlin, Germany"},
        ),
        expected={
            "salary_min_usd": 120000,
            "remote": True,
            "location": "Berlin, Germany",
        },
    ),
    Scenario(
        id="event",
        url="https://events.example/pydata-berlin",
        task="Collect the event date, venue and ticket price",
        html="""<html><body>
            <h1>PyData Berlin</h1>
            <p>Date: <time datetime="2026-12-03">3 December 2026</time></p>
            <p>Venue: Kraftwerk Berlin</p>
            <p>Tickets: EUR 49</p>
        </body></html>""",
        fields=(
            ("event_date", "date", "Event start date"),
            ("venue", "text", "Event venue name"),
            ("ticket_price_eur", "number", "Ticket price in EUR"),
        ),
        facts=(
            {"name": "event_date", "value": "2026-12-03", "evidence": "2026-12-03"},
            {"name": "venue", "value": "Kraftwerk Berlin", "evidence": "Kraftwerk Berlin"},
            {"name": "ticket_price_eur", "value": 49, "evidence": "EUR 49"},
        ),
        expected={
            "event_date": "2026-12-03",
            "venue": "Kraftwerk Berlin",
            "ticket_price_eur": 49,
        },
    ),
    Scenario(
        id="configuration-docs",
        url="https://docs.example/client/configuration",
        task="Collect timeout, streaming support and Python requirement",
        html="""<html><body>
            <h1>Client configuration</h1>
            <p>Default timeout: 30 seconds.</p>
            <p>Streaming: supported</p>
            <p>Python: 3.11 or newer</p>
        </body></html>""",
        fields=(
            ("default_timeout_seconds", "number", "Default request timeout in seconds"),
            ("supports_streaming", "bool", "Whether streaming responses are supported"),
            ("python_requires", "text", "Minimum supported Python version"),
        ),
        facts=(
            {
                "name": "default_timeout_seconds",
                "value": 30,
                "evidence": "30 seconds",
            },
            {"name": "supports_streaming", "value": True, "evidence": "supported"},
            {
                "name": "python_requires",
                "value": "3.11 or newer",
                "evidence": "3.11 or newer",
            },
        ),
        expected={
            "default_timeout_seconds": 30,
            "supports_streaming": True,
            "python_requires": "3.11 or newer",
        },
    ),
    Scenario(
        id="repository",
        url="https://code.example/web2spec",
        task="Collect repository license, popularity and language",
        html="""<html><body>
            <h1>web2spec</h1>
            <p>License: MIT</p>
            <p>Stars: 12,800</p>
            <p>Language: Python</p>
        </body></html>""",
        fields=(
            ("license", "text", "Repository license identifier"),
            ("stars", "number", "Number of GitHub stars"),
            ("language", "text", "Primary programming language"),
        ),
        facts=(
            {"name": "license", "value": "MIT", "evidence": "MIT"},
            {"name": "stars", "value": 12800, "evidence": "12,800"},
            {"name": "language", "value": "Python", "evidence": "Python"},
        ),
        expected={"license": "MIT", "stars": 12800, "language": "Python"},
    ),
)

SCENARIO_BY_ID = {scenario.id: scenario for scenario in SCENARIOS}
SCENARIO_BY_URL = {scenario.url: scenario for scenario in SCENARIOS}


def _plan_json(scenario: Scenario) -> str:
    plan = plan_from_fields(scenario.task, list(scenario.fields))
    return plan.model_dump_json()


def _facts_json(
    facts: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    missing: list[str] | None = None,
) -> str:
    return json.dumps(
        {"facts": list(facts), "missing": missing or []},
        ensure_ascii=False,
    )


def _assemble(
    scenario: Scenario,
    *,
    facts: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    return collect_urls.assemble_page(
        scenario.url,
        scenario.html,
        _plan_json(scenario),
        _facts_json(scenario.facts if facts is None else facts),
        task=scenario.task,
        allow_partial=allow_partial,
    )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_domain_scenarios_return_exact_typed_facts(scenario: Scenario) -> None:
    page = _assemble(scenario)

    assert page["status"] == "ok"
    assert page["entity"] == "custom"
    assert page["missing"] == []
    assert page["repairs"] == {"evidence": [], "dropped": []}
    assert page["dropped_foreign"] == []
    assert {fact["name"]: fact["value"] for fact in page["facts"]} == scenario.expected


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_domain_scenarios_fill_missing_fields(scenario: Scenario) -> None:
    first = scenario.facts[0]
    page = _assemble(scenario, facts=[first], allow_partial=True)

    assert page["status"] == "ok"
    assert {fact["name"] for fact in page["facts"]} == {first["name"]}
    assert set(page["missing"]) == {
        name for name, _kind, _description in scenario.fields[1:]
    }


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_domain_scenarios_strict_mode_rejects_partial_coverage(
    scenario: Scenario,
) -> None:
    page = _assemble(scenario, facts=[scenario.facts[0]], allow_partial=False)

    assert page["status"] == "extract_error"
    assert page["facts"] == []
    assert page["missing"] == []
    assert "without a fact or missing" in page["error"]


def test_evidence_repair_handles_number_date_and_bool_across_tasks() -> None:
    scenario = Scenario(
        id="repair",
        url="https://enterprise.example/terms",
        task="Collect enterprise terms",
        html="""<html><body>
            <h1>Enterprise plan</h1>
            <p>Annual price: $1,200 per year.</p>
            <p>Launch date: 14 October 2026.</p>
            <p>SSO enabled: yes.</p>
        </body></html>""",
        fields=(
            ("annual_price_usd", "number", "Annual price in USD"),
            ("launch_date", "date", "Plan launch date"),
            ("sso_enabled", "bool", "Whether SSO is enabled"),
        ),
        facts=(
            {"name": "annual_price_usd", "value": 1200, "evidence": "1200 dollars"},
            {"name": "launch_date", "value": "2026-10-14", "evidence": "14.10.2026"},
            {"name": "sso_enabled", "value": True, "evidence": "SSO is enabled"},
        ),
        expected={
            "annual_price_usd": 1200,
            "launch_date": "2026-10-14",
            "sso_enabled": True,
        },
    )
    page = _assemble(scenario)
    compressed = collect_urls.compress_html(scenario.html, max_chars=12_000)

    assert page["status"] == "ok"
    assert {fact["name"]: fact["value"] for fact in page["facts"]} == scenario.expected
    assert {row["name"] for row in page["repairs"]["evidence"]} == {
        "annual_price_usd",
        "launch_date",
        "sso_enabled",
    }
    for row in page["repairs"]["evidence"]:
        assert row["new"] in compressed
    assert page["missing"] == []


def test_cli_multi_domain_plans_are_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    for scenario in SCENARIOS:
        digest = hashlib.sha1(scenario.url.encode("utf-8")).hexdigest()
        (html_dir / f"{digest}.html").write_text(scenario.html, encoding="utf-8")

    def fake_plans(items, model_dir, max_new_tokens, task, **kwargs):
        return {
            item_url: _plan_json(SCENARIO_BY_URL[item_url])
            for item_url, _ in items
        }

    def fake_facts(items, model_dir, max_new_tokens, task, **kwargs):
        return {
            item_url: _facts_json(SCENARIO_BY_URL[item_url].facts)
            for item_url, _, _ in items
        }

    monkeypatch.setattr(collect_urls, "generate_plans", fake_plans)
    monkeypatch.setattr(collect_urls, "generate_facts", fake_facts)
    out = tmp_path / "multi-domain.json"
    argv = ["--task", "Collect selected facts from each page"]
    for scenario in SCENARIOS:
        argv += ["--url", scenario.url]
    argv += [
        "--html-dir",
        str(html_dir),
        "--out",
        str(out),
        "--planner",
        "unused",
        "--extractor",
        "unused",
        "--no-cache",
    ]

    assert collect_urls.main(argv) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    stats = report["stats"]
    assert stats["requested"] == len(SCENARIOS)
    assert stats["ok"] == len(SCENARIOS)
    assert stats["failed"] == 0
    assert stats["by_status"] == {"ok": len(SCENARIOS)}
    for scenario in SCENARIOS:
        aggregate = report["aggregate"][scenario.url]
        assert aggregate["status"] == "ok"
        assert aggregate["facts"] == scenario.expected
        assert aggregate["missing"] == []
    assert "input_html" not in json.dumps(report)


def test_cli_explicit_fields_support_date_bool_and_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = SCENARIO_BY_ID["release-notes"]
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    digest = hashlib.sha1(scenario.url.encode("utf-8")).hexdigest()
    (html_dir / f"{digest}.html").write_text(scenario.html, encoding="utf-8")

    def no_planner(*args, **kwargs):
        raise AssertionError("planner must not run with explicit fields")

    monkeypatch.setattr(collect_urls, "generate_plans", no_planner)
    monkeypatch.setattr(
        collect_urls,
        "generate_facts",
        lambda items, model_dir, max_new_tokens, task, **kwargs: {
            item_url: _facts_json(scenario.facts) for item_url, _, _ in items
        },
    )
    out = tmp_path / "explicit-kinds.json"
    argv = [
        "--task",
        scenario.task,
        "--url",
        scenario.url,
        "--html-dir",
        str(html_dir),
        "--out",
        str(out),
        "--planner",
        "unused",
        "--extractor",
        "unused",
        "--field",
        "release_version:text:Released product version",
        "--field",
        "released_on:date:Release date",
        "--field",
        "breaking_change:bool:Whether the release has breaking changes",
        "--field",
        "release_notes_url:url:Canonical release notes URL",
        "--no-cache",
    ]

    assert collect_urls.main(argv) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    page = report["pages"][0]
    assert report["stats"]["explicit_plan"] is True
    assert page["entity"] == "custom"
    assert {fact["name"]: fact["value"] for fact in page["facts"]} == scenario.expected
