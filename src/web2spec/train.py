"""QLoRA training for a 4 GB GPU.

Heavy dependencies are imported inside functions so that configuration and
dataset checks work before the ML stack is installed.
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from web2spec.config import REPO_ROOT, DataConfig, TrainConfig
from web2spec.dataset import CausalLMCollator, ExtractionDataset
from web2spec.tokens import load_tokenizer
from web2spec.validate import load_samples


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _dtype_kwarg(dtype: Any) -> dict[str, Any]:
    """Transformers v5 renamed ``torch_dtype`` to ``dtype``."""
    import transformers

    major = int(transformers.__version__.split(".", 1)[0])
    return {"dtype" if major >= 5 else "torch_dtype": dtype}


def build_model_and_tokenizer(cfg: TrainConfig) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = load_tokenizer(cfg.base_model)
    use_cuda = torch.cuda.is_available()
    compute_dtype = _torch_dtype(cfg.quant.compute_dtype)

    quantization_config = None
    if cfg.quant.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "bitsandbytes не установлен. Выполните: pip install bitsandbytes"
            ) from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.quant.quant_type,
            bnb_4bit_use_double_quant=cfg.quant.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quantization_config,
        **_dtype_kwarg(compute_dtype),
        device_map={"": 0} if use_cuda else None,
        attn_implementation=cfg.attn_implementation if use_cuda else "eager",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


def build_training_arguments(
    cfg: TrainConfig,
    *,
    output_dir: Path,
    has_eval: bool,
    warmup_steps: int = 0,
    smoke: bool = False,
) -> Any:
    import torch
    from transformers import TrainingArguments

    use_cuda = torch.cuda.is_available()
    bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": cfg.grad_accum,
        "num_train_epochs": 1 if smoke else cfg.epochs,
        "learning_rate": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "lr_scheduler_type": cfg.scheduler,
        "warmup_steps": warmup_steps,
        "optim": cfg.optim,
        "max_grad_norm": cfg.max_grad_norm,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": bf16,
        "fp16": bool(use_cuda and not bf16),
        "logging_steps": 1 if smoke else cfg.logging_steps,
        "save_steps": cfg.eval_steps if has_eval else cfg.save_steps,
        "eval_steps": cfg.eval_steps,
        "eval_strategy": "steps" if has_eval else "no",
        "save_strategy": "no" if smoke else "steps",
        "metric_for_best_model": "eval_loss" if has_eval else None,
        "greater_is_better": False,
        "load_best_model_at_end": bool(has_eval),
        "save_total_limit": 3,
        "report_to": [],
        "seed": cfg.seed,
        "data_seed": cfg.seed,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "max_steps": 20 if smoke else -1,
        "logging_first_step": True,
    }
    return TrainingArguments(**kwargs)


def train(cfg: TrainConfig, data_cfg: DataConfig, *, smoke: bool = False) -> Path:
    import torch
    from transformers import EarlyStoppingCallback, Trainer

    paths = data_cfg.paths.resolve()
    train_path = paths.processed / "train.jsonl"
    val_path = paths.processed / "val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"нет обучающего файла: {train_path}")

    train_samples = load_samples(train_path)
    val_samples = load_samples(val_path) if val_path.exists() else []
    if smoke:
        train_samples = train_samples[:32]
        val_samples = val_samples[:8]
    if not train_samples:
        raise ValueError("обучающий набор пуст")

    out_dir = (REPO_ROOT / cfg.output_dir).resolve()
    if smoke:
        out_dir = (REPO_ROOT / "outputs/runs/_smoke").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(cfg)
    train_ds = ExtractionDataset(train_samples, tokenizer, cfg.max_seq_len)
    val_ds = ExtractionDataset(val_samples, tokenizer, cfg.max_seq_len) if val_samples else None
    collator = CausalLMCollator(tokenizer.pad_token_id or tokenizer.eos_token_id)

    steps_per_epoch = math.ceil(len(train_ds) / max(1, cfg.batch_size * cfg.grad_accum))
    total_steps = 20 if smoke else max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = 1 if smoke else max(1, round(total_steps * cfg.warmup_ratio))
    args = build_training_arguments(
        cfg,
        output_dir=out_dir,
        has_eval=val_ds is not None,
        warmup_steps=warmup_steps,
        smoke=smoke,
    )
    callbacks = []
    if val_ds is not None and cfg.early_stopping_patience > 0 and not smoke:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    started = datetime.now(UTC)
    trainer.train()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    peak_vram = None
    if torch.cuda.is_available():
        peak_vram = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    manifest = {
        "created_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "smoke": smoke,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "effective_batch_size": cfg.effective_batch_size,
        "max_seq_len": cfg.max_seq_len,
        "peak_vram_gb": peak_vram,
        "git_commit": _git_commit(),
        "train_config": cfg.model_dump(mode="json"),
        "data_config": data_cfg.model_dump(mode="json"),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return adapter_dir


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() or None
