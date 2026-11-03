"""Teacher-model synthetic data generation with strict validation."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from web2spec.eval import extract_json
from web2spec.html_compress import compress_html
from web2spec.schema import SampleError, make_id, validate_sample

SYSTEM_TEMPLATE = (
    "Ты — эксперт по извлечению данных из HTML. "
    "Верни ТОЛЬКО валидный JSON, соответствующий этой JSON Schema:\n{schema}\n"
    "Без markdown и пояснений. Отсутствующие поля — null."
)


class Teacher(Protocol):
    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str: ...


class OpenAICompatTeacher:
    """Minimal OpenAI-compatible client (vLLM, LM Studio, OpenRouter, ...)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "qwen2.5-coder-32b-instruct",
        timeout: float = 90.0,
        retries: int = 2,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        if not self.base_url:
            raise ValueError("не задан OPENAI_BASE_URL")

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return str(data["choices"][0]["message"]["content"])
            except (urllib.error.HTTPError, urllib.error.URLError, KeyError, ValueError) as exc:
                last_error = exc
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                if not retryable or attempt == self.retries:
                    break
                time.sleep(min(2 * (attempt + 1), 8))
        raise RuntimeError(f"teacher request failed: {last_error}")


def build_messages(record: dict[str, Any], schema: dict[str, Any]) -> tuple[str, str]:
    system = SYSTEM_TEMPLATE.format(schema=json.dumps(schema, ensure_ascii=False))
    html = str(record.get("input_html") or record.get("html") or "")
    html = compress_html(html, max_chars=5000)
    instruction = str(record.get("instruction") or "Извлеки данные страницы в JSON.")
    user = f"{instruction}\n\nHTML:\n{html}"
    return system, user[:6000]


def synthesize(
    inputs: list[dict[str, Any]],
    teacher: Teacher,
    *,
    schema: dict[str, Any],
    task: str = "extract",
    min_chars: int = 50,
    dedup: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    by_reason: dict[str, int] = {}

    for index, record in enumerate(inputs):
        try:
            system, user = build_messages(record, schema)
            raw = teacher.complete(system, user)
            target = extract_json(raw)
            if target is None:
                raise SampleError("E_JSON", "teacher did not return a JSON object")
            html = str(record.get("input_html") or record.get("html") or "")
            if len(html) < min_chars:
                raise SampleError("E_TOO_SHORT", f"html has {len(html)} chars")
            source_url = record.get("source_url")
            group_id = record.get("group_id") or _host(source_url) or f"synth-{index}"
            payload = {
                "id": make_id(task, group_id, index, html),
                "group_id": group_id,
                "task": task,
                "instruction": record.get("instruction") or "Извлеки данные страницы в JSON.",
                "input_html": html,
                "target": target,
                "meta": {"source_url": source_url, "complexity": record.get("complexity", 1)},
            }
            sample = validate_sample(payload, {task: schema})
            digest = hashlib.sha256(
                json.dumps(target, sort_keys=True, ensure_ascii=False).encode("utf-8")
                + html.encode("utf-8")
            ).hexdigest()
            if dedup and digest in seen:
                raise SampleError("E_DUP", "duplicate synthetic sample")
            seen.add(digest)
            accepted.append(sample.model_dump(mode="json"))
        except SampleError as exc:
            by_reason[exc.code] = by_reason.get(exc.code, 0) + 1
        except (RuntimeError, ValueError) as exc:
            code = "E_TEACHER"
            by_reason[code] = by_reason.get(code, 0) + 1
            _ = exc

    stats = {
        "total": len(inputs),
        "accepted": len(accepted),
        "rejected": len(inputs) - len(accepted),
        "by_reason": by_reason,
        "duration_s": round(time.perf_counter() - started, 3),
    }
    return accepted, stats


def _host(url: str | None) -> str | None:
    if not url:
        return None
    return url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0] or None
