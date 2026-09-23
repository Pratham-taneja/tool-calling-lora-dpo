from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/experiment.yaml")


@dataclass(frozen=True)
class ModelConfig:
    id: str
    fallback_id: str
    load_in_4bit: bool


@dataclass(frozen=True)
class DataConfig:
    id: str
    validation_size: int
    test_size: int


@dataclass(frozen=True)
class PromptConfig:
    max_seq_length: int


@dataclass(frozen=True)
class EvaluationConfig:
    max_new_tokens: int
    do_sample: bool
    batch_size: int


@dataclass(frozen=True)
class SftConfig:
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    micro_batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    scheduler: str
    warmup_ratio: float
    optimizer: str
    assistant_only_loss: bool
    gradient_checkpointing: str


@dataclass(frozen=True)
class DpoConfig:
    source_examples: int
    candidates_per_prompt: int
    sample_temperature: float
    sample_top_p: float
    beta: float
    micro_batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    model: ModelConfig
    data: DataConfig
    prompt: PromptConfig
    evaluation: EvaluationConfig
    sft: SftConfig
    dpo: DpoConfig


class ConfigError(ValueError):
    """Raised when the experiment config is missing or invalid."""


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    """Load and validate the YAML experiment config.

    The config file is the single source of truth for experiment defaults.
    Keeping validation here prevents training and evaluation scripts from
    silently accepting missing or misspelled settings.
    """

    config_path = Path(path)

    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if raw_config is None:
        raise ConfigError(f"Config file is empty: {config_path}")

    root = _require_mapping(raw_config, "root")

    config = ExperimentConfig(
        seed=_require_int(root, "seed"),
        model=_load_model_config(_require_mapping(root.get("model"), "model")),
        data=_load_data_config(_require_mapping(root.get("data"), "data")),
        prompt=_load_prompt_config(_require_mapping(root.get("prompt"), "prompt")),
        evaluation=_load_evaluation_config(_require_mapping(root.get("evaluation"), "evaluation")),
        sft=_load_sft_config(_require_mapping(root.get("sft"), "sft")),
        dpo=_load_dpo_config(_require_mapping(root.get("dpo"), "dpo")),
    )

    _validate_config(config)
    return config


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert the dataclass config into plain Python dictionaries."""

    return asdict(config)


def config_hash(config: ExperimentConfig) -> str:
    """Return a stable SHA-256 hash for the loaded config.

    We use sorted JSON so the same config produces the same hash regardless of
    dictionary insertion order. This hash later goes into manifests, checkpoints,
    and evaluation files so mismatched runs are easy to detect.
    """

    canonical_json = json.dumps(
        config_to_dict(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _load_model_config(data: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        id=_require_str(data, "id"),
        fallback_id=_require_str(data, "fallback_id"),
        load_in_4bit=_require_bool(data, "load_in_4bit"),
    )


def _load_data_config(data: dict[str, Any]) -> DataConfig:
    return DataConfig(
        id=_require_str(data, "id"),
        validation_size=_require_int(data, "validation_size"),
        test_size=_require_int(data, "test_size"),
    )


def _load_prompt_config(data: dict[str, Any]) -> PromptConfig:
    return PromptConfig(
        max_seq_length=_require_int(data, "max_seq_length"),
    )


def _load_evaluation_config(data: dict[str, Any]) -> EvaluationConfig:
    return EvaluationConfig(
        max_new_tokens=_require_int(data, "max_new_tokens"),
        do_sample=_require_bool(data, "do_sample"),
        batch_size=_require_int(data, "batch_size"),
    )


def _load_sft_config(data: dict[str, Any]) -> SftConfig:
    return SftConfig(
        lora_rank=_require_int(data, "lora_rank"),
        lora_alpha=_require_int(data, "lora_alpha"),
        lora_dropout=_require_float(data, "lora_dropout"),
        micro_batch_size=_require_int(data, "micro_batch_size"),
        gradient_accumulation_steps=_require_int(data, "gradient_accumulation_steps"),
        epochs=_require_int(data, "epochs"),
        learning_rate=_require_float(data, "learning_rate"),
        scheduler=_require_str(data, "scheduler"),
        warmup_ratio=_require_float(data, "warmup_ratio"),
        optimizer=_require_str(data, "optimizer"),
        assistant_only_loss=_require_bool(data, "assistant_only_loss"),
        gradient_checkpointing=_require_str(data, "gradient_checkpointing"),
    )


def _load_dpo_config(data: dict[str, Any]) -> DpoConfig:
    return DpoConfig(
        source_examples=_require_int(data, "source_examples"),
        candidates_per_prompt=_require_int(data, "candidates_per_prompt"),
        sample_temperature=_require_float(data, "sample_temperature"),
        sample_top_p=_require_float(data, "sample_top_p"),
        beta=_require_float(data, "beta"),
        micro_batch_size=_require_int(data, "micro_batch_size"),
        gradient_accumulation_steps=_require_int(data, "gradient_accumulation_steps"),
        epochs=_require_int(data, "epochs"),
        learning_rate=_require_float(data, "learning_rate"),
    )


def _validate_config(config: ExperimentConfig) -> None:
    if config.seed < 0:
        raise ConfigError("seed must be non-negative")

    if config.data.validation_size <= 0:
        raise ConfigError("data.validation_size must be positive")

    if config.data.test_size <= 0:
        raise ConfigError("data.test_size must be positive")

    if config.prompt.max_seq_length <= 0:
        raise ConfigError("prompt.max_seq_length must be positive")

    if config.evaluation.max_new_tokens <= 0:
        raise ConfigError("evaluation.max_new_tokens must be positive")

    if config.evaluation.batch_size <= 0:
        raise ConfigError("evaluation.batch_size must be positive")

    if config.evaluation.do_sample:
        raise ConfigError("evaluation.do_sample must be false for frozen greedy evaluation")

    if config.sft.micro_batch_size != 1:
        raise ConfigError("sft.micro_batch_size must remain 1 for the 6 GB VRAM plan")

    if config.dpo.micro_batch_size != 1:
        raise ConfigError("dpo.micro_batch_size must remain 1 for the 6 GB VRAM plan")

    if not 0.0 <= config.sft.lora_dropout <= 1.0:
        raise ConfigError("sft.lora_dropout must be between 0 and 1")

    if not 0.0 <= config.sft.warmup_ratio <= 1.0:
        raise ConfigError("sft.warmup_ratio must be between 0 and 1")

    if not 0.0 < config.dpo.sample_top_p <= 1.0:
        raise ConfigError("dpo.sample_top_p must be greater than 0 and at most 1")

    if config.dpo.sample_temperature <= 0:
        raise ConfigError("dpo.sample_temperature must be positive")

    if config.dpo.beta <= 0:
        raise ConfigError("dpo.beta must be positive")


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)

    # bool is a subclass of int in Python, so reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")

    return value


def _require_float(data: dict[str, Any], key: str) -> float:
    value = data.get(key)

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{key} must be a number")

    return float(value)


def _require_bool(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)

    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be true or false")

    return value
