from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_calling_lora_dpo.config import load_config
from tool_calling_lora_dpo.inference import load_hf_causal_lm
from tool_calling_lora_dpo.sft_training import (
    AssistantOnlyDataCollator,
    build_sft_training_plan,
    build_training_result,
    configure_tokenizer_for_training,
    limit_records,
    read_sft_records,
    records_to_trainer_rows,
    resolve_resume_checkpoint,
    write_training_report,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "sft_train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "models" / "sft_lora"
DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "training" / "sft_train_report.json"
DEFAULT_HF_HOME = PROJECT_ROOT / "data" / "hf_cache"


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = load_config(args.config)
    records = limit_records(read_sft_records(args.input), args.limit)
    plan = build_sft_training_plan(
        config=config,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        model_id=args.model_id,
    )

    if args.dry_run:
        print(json.dumps(dry_run_summary(plan=plan, record_count=len(records)), indent=2))
        return 0

    require_cuda()
    resume = resolve_resume_checkpoint(args.resume, output_dir=plan.output_dir)
    result_metrics = run_training(config=config, plan=plan, records=records, resume=resume)
    result = build_training_result(
        config=config,
        plan=plan,
        records=records,
        metrics=result_metrics,
    )
    write_training_report(args.report, result)

    print(f"Wrote SFT adapter to {args.output_dir}")
    print(f"Wrote training report to {args.report}")
    print(json.dumps(result_metrics, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SFT LoRA adapter.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_cuda() -> None:
    """Stop training early if the CUDA stack is not visible."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 4-bit SFT training script")


def run_training(
    *,
    config: Any,
    plan: Any,
    records: list[Any],
    resume: str | None,
) -> dict[str, Any]:
    """Run SFT with LoRA adapters and return Trainer metrics."""

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(config.seed)
    tokenizer = configure_tokenizer_for_training(AutoTokenizer.from_pretrained(plan.model_id))
    model = load_hf_causal_lm(plan.model_id, load_in_4bit=config.model.load_in_4bit)

    if hasattr(model, "config"):
        model.config.use_cache = False

    if config.model.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=plan.gradient_checkpointing,
        )

    lora_config = LoraConfig(
        r=plan.lora_rank,
        lora_alpha=plan.lora_alpha,
        lora_dropout=plan.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = Dataset.from_list(records_to_trainer_rows(records))
    collator = AssistantOnlyDataCollator(
        tokenizer,
        max_seq_length=plan.max_seq_length,
        assistant_only_loss=plan.assistant_only_loss,
    )

    training_args = TrainingArguments(
        output_dir=str(plan.output_dir),
        per_device_train_batch_size=plan.micro_batch_size,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        num_train_epochs=float(plan.epochs),
        max_steps=plan.max_steps,
        learning_rate=plan.learning_rate,
        lr_scheduler_type=plan.scheduler,
        warmup_ratio=plan.warmup_ratio,
        optim=plan.optimizer,
        fp16=True,
        bf16=False,
        gradient_checkpointing=plan.gradient_checkpointing,
        logging_strategy="steps",
        logging_steps=plan.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=plan.save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=config.seed,
        data_seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    train_output = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(plan.output_dir))
    tokenizer.save_pretrained(str(plan.output_dir))

    metrics = dict(train_output.metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    return metrics


def dry_run_summary(*, plan: Any, record_count: int) -> dict[str, Any]:
    """Return a no-GPU summary of what would be trained."""

    return {
        "stage": "sft",
        "record_count": record_count,
        "plan": {
            "model_id": plan.model_id,
            "output_dir": plan.output_dir,
            "max_seq_length": plan.max_seq_length,
            "micro_batch_size": plan.micro_batch_size,
            "gradient_accumulation_steps": plan.gradient_accumulation_steps,
            "epochs": plan.epochs,
            "learning_rate": plan.learning_rate,
            "scheduler": plan.scheduler,
            "warmup_ratio": plan.warmup_ratio,
            "optimizer": plan.optimizer,
            "assistant_only_loss": plan.assistant_only_loss,
            "gradient_checkpointing": plan.gradient_checkpointing,
            "max_steps": plan.max_steps,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
