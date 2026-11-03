"""Torch-free dataset layer; tensors are created only in the collator."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from web2spec.schema import Sample
from web2spec.tokens import build_prompt, target_text
from web2spec.validate import iter_jsonl, normalize_record, validate_sample


class ExtractionDataset:
    """Prompt + answer pairs with prompt tokens masked out of the loss."""

    def __init__(
        self,
        samples: list[Sample] | Path,
        tokenizer: Any,
        max_seq_len: int = 1024,
    ) -> None:
        if max_seq_len < 64:
            raise ValueError("max_seq_len must be at least 64")
        if isinstance(samples, Path):
            samples = [
                validate_sample(normalize_record(payload, line=line_no))
                for line_no, payload in iter_jsonl(samples)
            ]
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        sample = self.samples[index]
        prompt_ids = self._encode(build_prompt(sample, self.tokenizer))
        answer_ids = self._encode(target_text(sample))
        if self.eos_token_id is not None:
            answer_ids = [*answer_ids, int(self.eos_token_id)]

        budget = self.max_seq_len
        if len(answer_ids) >= budget:
            answer_ids = answer_ids[:budget]
            prompt_ids = []
            warnings.warn(
                f"sample {sample.id}: target alone exceeds max_seq_len, truncated",
                RuntimeWarning,
                stacklevel=2,
            )
        elif len(prompt_ids) + len(answer_ids) > budget:
            keep = budget - len(answer_ids)
            prompt_ids = prompt_ids[:keep]
            warnings.warn(
                f"sample {sample.id}: prompt truncated to {keep} tokens",
                RuntimeWarning,
                stacklevel=2,
            )

        input_ids = [*prompt_ids, *answer_ids]
        labels = [-100] * len(prompt_ids) + list(answer_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def _encode(self, text: str) -> list[int]:
        output = self.tokenizer(text, add_special_tokens=False)
        ids = output["input_ids"] if isinstance(output, dict) else output.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(token) for token in ids]


class CausalLMCollator:
    """Right-padding collator; pad labels with -100 so they are ignored."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        longest = max(len(feature["input_ids"]) for feature in features)
        multiple = self.pad_to_multiple_of
        if multiple > 1:
            longest = ((longest + multiple - 1) // multiple) * multiple

        input_ids, attention, labels = [], [], []
        for feature in features:
            pad = longest - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            attention.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
