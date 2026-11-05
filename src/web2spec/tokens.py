"""Prompt building and token counting.

The tokenizer is duck-typed: anything exposing ``apply_chat_template``,
``encode`` or a callable ``__call__`` works. This keeps validation usable
before the transformers stack is installed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from web2spec.schema import Sample

SYSTEM_PROMPT = (
    "Ты извлекаешь структурированные данные из HTML. "
    "Верни ТОЛЬКО валидный JSON, соответствующий требуемой схеме. "
    "Без markdown, без пояснений, без комментариев. "
    "Если поля нет на странице — используй null."
)

PLAN_SYSTEM_PROMPT = (
    "Ты планировщик сбора данных. По задаче пользователя и образцу страницы "
    "верни ТОЛЬКО валидный JSON CollectionPlan. Не добавляй CSS-селекторы, "
    "XPath или код. Поля должны быть семантическими, соответствовать задаче "
    "и только ей. Если задача явно просит дату, автора или другой атрибут — "
    "включи его в fields; если не просит — не добавляй. "
    'Если на странице нет данных для задачи, верни entity="none" и пустые fields.'
)

PLAN_EXTRACT_SYSTEM_PROMPT = (
    "Ты извлекаешь факты из HTML по заданному CollectionPlan. "
    "Верни ТОЛЬКО валидный JSON FactSet: объект с ключами facts (массив) и missing (массив). "
    "Каждый элемент facts — объект ровно с тремя ключами: name, value, evidence. "
    "name — строго одно из имён полей плана, скопированное из fields. "
    "value — найденное значение. "
    "evidence — точная подстрока HTML, скопированная дословно. "
    "Не нормализуй, не переводи и не вычисляй даты: если на странице написано "
    "«5 часов назад», evidence и value должны содержать именно эту строку. "
    "Если точного значения нет — перечисли имя поля в missing. "
    "Не выдумывай значения и не добавляй поля вне плана."
)

ASSISTANT_SUFFIX = "<|im_start|>assistant\n"


def build_user_message(sample: Sample) -> str:
    return f"{sample.instruction}\n\nHTML:\n{sample.input_html}"


def build_plan_user_message(sample: Sample) -> str:
    source = sample.meta.source_url or "unknown"
    return f"Задача: {sample.instruction}\nСайт: {source}\nHTML-фрагмент:\n{sample.input_html}"


def build_plan_prompt(sample: Sample, tokenizer: Any) -> str:
    """Render the planner prompt; the target is a semantic CollectionPlan."""

    return _render_messages(_plan_messages(sample), tokenizer)


def build_plan_prompt_text(sample: Sample) -> str:
    """Render the planner prompt without loading a tokenizer (GGUF runtime)."""

    return _manual_chatml(_plan_messages(sample))


def _plan_messages(sample: Sample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": build_plan_user_message(sample)},
    ]


def build_plan_extract_user_message(sample: Sample) -> str:
    plan = sample.meta.plan
    if not isinstance(plan, dict) or not plan:
        raise ValueError(f"sample {sample.id}: plan_extract requires meta.plan")
    plan_json = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
    return f"План:\n{plan_json}\n\nЗадача: {sample.instruction}\n\nHTML:\n{sample.input_html}"


def build_plan_extract_prompt(sample: Sample, tokenizer: Any) -> str:
    """Render the plan-conditioned extraction prompt."""

    return _render_messages(_plan_extract_messages(sample), tokenizer)


def build_plan_extract_prompt_text(sample: Sample) -> str:
    """Render the extraction prompt without a tokenizer (GGUF runtime)."""

    return _manual_chatml(_plan_extract_messages(sample))


def _plan_extract_messages(sample: Sample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PLAN_EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": build_plan_extract_user_message(sample)},
    ]


def build_prompt_text(sample: Sample) -> str:
    """Render any prompt without a tokenizer for the llama.cpp runtime."""

    if sample.task == "plan":
        return build_plan_prompt_text(sample)
    if sample.task == "plan_extract":
        return build_plan_extract_prompt_text(sample)
    return _manual_chatml(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(sample)},
        ]
    )


def build_prompt(sample: Sample, tokenizer: Any) -> str:
    """Render the chat prompt exactly as it will be seen during training."""

    if sample.task == "plan":
        return build_plan_prompt(sample, tokenizer)
    if sample.task == "plan_extract":
        return build_plan_extract_prompt(sample, tokenizer)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(sample)},
    ]
    return _render_messages(messages, tokenizer)


def _render_messages(messages: list[dict[str, str]], tokenizer: Any) -> str:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            return apply_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return _manual_chatml(messages)


def target_text(sample: Sample) -> str:
    return json.dumps(sample.target, ensure_ascii=False, separators=(",", ":"))


def count_tokens(text: str, tokenizer: Any) -> int:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        ids = encode(text, add_special_tokens=False)
    elif callable(tokenizer):
        output = tokenizer(text, add_special_tokens=False)
        ids = output["input_ids"] if isinstance(output, dict) else output.input_ids
    else:
        raise TypeError("tokenizer must expose encode() or be callable")
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def token_lengths(samples: Iterable[Sample], tokenizer: Any) -> list[int]:
    return [count_tokens(build_prompt(sample, tokenizer), tokenizer) for sample in samples]


def load_tokenizer(name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _manual_chatml(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    parts.append(ASSISTANT_SUFFIX)
    return "".join(parts)
