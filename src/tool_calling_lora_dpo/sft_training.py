from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from tool_calling_lora_dpo.config import ExperimentConfig, config_hash
from tool_calling_lora_dpo.data import ids_sha256, read_jsonl
from tool_calling_lora_dpo.inference import cuda_memory_summary
from tool_calling_lora_dpo.sft_dataset import SftRecord, sft_record_from_json_dict

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

OPTIMIZER_NAME_MAP = {
    "adamw_8bit": "paged_adamw_8bit",
    "paged_adamw_8bit": "paged_adamw_8bit",
    "adamw_torch": "adamw_torch",
}

TRAINER_STATE_FILE = "trainer_state.json"
TRAINER_CHECKPOINT_WEIGHT_FILES = (
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
)


class TrainingTokenizer(Protocol):
    """Tokenizer interface needed by the SFT collator."""

    eos_token_id: int | None
    pad_token_id: int | None
    eos_token: str | None
    pad_token: str | None
    padding_side: str

    def __call__(
        self,
        text: list[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> Any:
        """Tokenize a batch of already-rendered SFT text."""


@dataclass(frozen=True)
class SftTrainingPlan:
    """Training settings derived from config and CLI overrides."""

    model_id: str
    output_dir: str
    max_seq_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    scheduler: str
    warmup_ratio: float
    optimizer: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    assistant_only_loss: bool
    gradient_checkpointing: bool
    max_steps: int
    logging_steps: int
    save_steps: int


@dataclass(frozen=True)
class SftTrainingResult:
    """Compact result saved after an SFT run."""

    stage: str
    model_id: str
    output_dir: str
    train_record_count: int
    train_ids_hash: str
    config_hash: str
    max_train_steps: int
    metrics: dict[str, Any]
    cuda_memory: dict[str, Any]


class AssistantOnlyDataCollator:
    """Tokenize SFT rows and mask prompt tokens from the language-modeling loss."""

    def __init__(
        self,
        tokenizer: TrainingTokenizer,
        *,
        max_seq_length: int,
        assistant_only_loss: bool,
    ) -> None:
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.assistant_only_loss = assistant_only_loss

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("features must not be empty")

        texts = [str(feature["text"]) for feature in features]
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        pad_token_id = self.tokenizer.pad_token_id

        if pad_token_id is not None:
            labels[input_ids == pad_token_id] = -100

        if self.assistant_only_loss:
            for row_index, feature in enumerate(features):
                prompt_token_length = int(feature["prompt_token_length"])
                labels[row_index, : min(prompt_token_length, labels.shape[1])] = -100

        batch["labels"] = labels
        return batch


def configure_tokenizer_for_training(tokenizer: TrainingTokenizer) -> TrainingTokenizer:
    """Apply training-safe tokenizer defaults."""

    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def read_sft_records(path: str | Path) -> list[SftRecord]:
    """Read SFT JSONL records from disk with trainer-specific validation."""

    return [sft_record_from_json_dict(row) for row in read_jsonl(path)]


def limit_records(records: list[SftRecord], limit: int | None) -> list[SftRecord]:
    """Apply an optional smoke-run limit while preserving deterministic order."""

    if limit is None:
        return records

    if limit <= 0:
        raise ValueError("limit must be positive when provided")

    return records[:limit]


def optimizer_name_for_transformers(name: str) -> str:
    """Map project config names to names accepted by Transformers."""

    try:
        return OPTIMIZER_NAME_MAP[name]
    except KeyError as error:
        supported = ", ".join(sorted(OPTIMIZER_NAME_MAP))
        raise ValueError(f"unsupported SFT optimizer {name!r}; supported: {supported}") from error


def checkpoint_step(path: str | Path) -> int | None:
    """Return the numeric Trainer checkpoint suffix, if present."""

    name = Path(path).name
    prefix = "checkpoint-"
    if not name.startswith(prefix):
        return None

    suffix = name[len(prefix) :]
    if not suffix.isdigit():
        return None

    return int(suffix)


def is_valid_trainer_checkpoint(path: str | Path) -> bool:
    """Return whether a directory has the files needed to resume Trainer training."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_dir():
        return False

    has_state = (checkpoint_path / TRAINER_STATE_FILE).is_file()
    has_weights = any(
        (checkpoint_path / filename).is_file() for filename in TRAINER_CHECKPOINT_WEIGHT_FILES
    )
    return has_state and has_weights


def valid_trainer_checkpoints(output_dir: str | Path) -> list[Path]:
    """List valid checkpoint directories under a Trainer output directory."""

    output_path = Path(output_dir)
    if not output_path.is_dir():
        return []

    checkpoints = [
        candidate
        for candidate in output_path.iterdir()
        if checkpoint_step(candidate) is not None and is_valid_trainer_checkpoint(candidate)
    ]
    return sorted(checkpoints, key=lambda path: checkpoint_step(path) or -1)


def latest_valid_trainer_checkpoint(output_dir: str | Path) -> Path:
    """Return the newest valid checkpoint under a Trainer output directory."""

    checkpoints = valid_trainer_checkpoints(output_dir)
    if not checkpoints:
        raise ValueError(f"No valid Trainer checkpoints found under {Path(output_dir)}")

    return checkpoints[-1]


def resolve_resume_checkpoint(
    resume: str | Path | None,
    *,
    output_dir: str | Path,
) -> str | None:
    """Resolve and validate a Trainer resume checkpoint argument."""

    if resume is None:
        return None

    resume_text = str(resume)
    output_path = Path(output_dir)

    if resume_text.lower() in {"latest", "last", "auto"}:
        return str(latest_valid_trainer_checkpoint(output_path))

    resume_path = Path(resume_text)
    if not resume_path.exists() and len(resume_path.parts) == 1:
        output_checkpoint = output_path / resume_path
        if output_checkpoint.exists():
            resume_path = output_checkpoint

    if is_valid_trainer_checkpoint(resume_path):
        return str(resume_path)

    valid_names = [path.name for path in valid_trainer_checkpoints(output_path)]
    message = f"Can't find a valid Trainer checkpoint at {resume_path}."
    if valid_names:
        message += f" Valid checkpoints under {output_path}: {', '.join(valid_names)}."
    else:
        message += f" No valid checkpoints were found under {output_path}."
    message += " Use --resume latest or pass one of the valid checkpoint directories."
    raise ValueError(message)


def build_sft_training_plan(
    *,
    config: ExperimentConfig,
    output_dir: str | Path,
    max_steps: int,
    logging_steps: int,
    save_steps: int,
    model_id: str | None = None,
) -> SftTrainingPlan:
    """Build the exact settings passed to the HF Trainer."""

    if max_steps < -1 or max_steps == 0:
        raise ValueError("max_steps must be -1 or a positive integer")

    if logging_steps <= 0:
        raise ValueError("logging_steps must be positive")

    if save_steps <= 0:
        raise ValueError("save_steps must be positive")

    gradient_checkpointing = config.sft.gradient_checkpointing.lower() in {
        "true",
        "1",
        "yes",
        "unsloth",
    }

    return SftTrainingPlan(
        model_id=model_id or config.model.id,
        output_dir=str(output_dir),
        max_seq_length=config.prompt.max_seq_length,
        micro_batch_size=config.sft.micro_batch_size,
        gradient_accumulation_steps=config.sft.gradient_accumulation_steps,
        epochs=config.sft.epochs,
        learning_rate=config.sft.learning_rate,
        scheduler=config.sft.scheduler,
        warmup_ratio=config.sft.warmup_ratio,
        optimizer=optimizer_name_for_transformers(config.sft.optimizer),
        lora_rank=config.sft.lora_rank,
        lora_alpha=config.sft.lora_alpha,
        lora_dropout=config.sft.lora_dropout,
        assistant_only_loss=config.sft.assistant_only_loss,
        gradient_checkpointing=gradient_checkpointing,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
    )


def records_to_trainer_rows(records: list[SftRecord]) -> list[dict[str, Any]]:
    """Convert SFT dataclasses into rows consumed by datasets.Dataset."""

    return [asdict(record) for record in records]


def build_training_result(
    *,
    config: ExperimentConfig,
    plan: SftTrainingPlan,
    records: list[SftRecord],
    metrics: dict[str, Any],
) -> SftTrainingResult:
    """Create the report payload saved beside the adapter."""

    return SftTrainingResult(
        stage="sft",
        model_id=plan.model_id,
        output_dir=plan.output_dir,
        train_record_count=len(records),
        train_ids_hash=ids_sha256([record.source_id for record in records]),
        config_hash=config_hash(config),
        max_train_steps=plan.max_steps,
        metrics=metrics,
        cuda_memory=cuda_memory_summary(),
    )


def write_training_report(path: str | Path, result: SftTrainingResult) -> None:
    """Write the compact SFT training report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
