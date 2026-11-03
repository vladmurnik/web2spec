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

ASSISTANT_SUFFIX = "<|im_start|>assistant\n"


def build_user_message(sample: Sample) -> str:
    return f"{sample.instruction}\n\nHTML:\n{sample.input_html}"


def build_prompt(sample: Sample, tokenizer: Any) -> str:
    """Render the chat prompt exactly as it will be seen during training."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(sample)},
    ]
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
