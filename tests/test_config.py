from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from tool_calling_lora_dpo.config import ConfigError, config_hash, config_to_dict, load_config


def valid_config_dict() -> dict:
    """Return a complete minimal config matching the handoff defaults."""

    return {
        "seed": 42,
        "model": {
            "id": "Qwen/Qwen2.5-3B-Instruct",
            "fallback_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "load_in_4bit": True,
        },
        "data": {
            "id": "Salesforce/xlam-function-calling-60k",
            "validation_size": 500,
            "test_size": 300,
        },
        "prompt": {
            "max_seq_length": 1024,
        },
        "evaluation": {
            "max_new_tokens": 384,
            "do_sample": False,
            "batch_size": 1,
        },
        "sft": {
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "epochs": 1,
            "learning_rate": 0.0002,
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "optimizer": "adamw_8bit",
            "assistant_only_loss": True,
            "gradient_checkpointing": "unsloth",
        },
        "dpo": {
            "source_examples": 2000,
            "candidates_per_prompt": 4,
            "sample_temperature": 0.8,
            "sample_top_p": 0.9,
            "beta": 0.1,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "epochs": 1,
            "learning_rate": 0.000005,
        },
    }


def write_yaml(path: Path, data: dict) -> None:
    """Write YAML in tests without depending on the real project config file."""

    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_load_config_reads_expected_values(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    write_yaml(config_path, valid_config_dict())

    config = load_config(config_path)

    assert config.seed == 42
    assert config.model.id == "Qwen/Qwen2.5-3B-Instruct"
    assert config.model.load_in_4bit is True
    assert config.data.validation_size == 500
    assert config.data.test_size == 300
    assert config.evaluation.do_sample is False
    assert config.sft.micro_batch_size == 1
    assert config.dpo.micro_batch_size == 1


def test_config_hash_is_stable(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    write_yaml(config_path, valid_config_dict())

    first = load_config(config_path)
    second = load_config(config_path)

    assert config_hash(first) == config_hash(second)
    assert len(config_hash(first)) == 64


def test_config_hash_changes_when_config_changes(tmp_path: Path) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"

    first_data = valid_config_dict()
    second_data = copy.deepcopy(first_data)
    second_data["prompt"]["max_seq_length"] = 1536

    write_yaml(first_path, first_data)
    write_yaml(second_path, second_data)

    assert config_hash(load_config(first_path)) != config_hash(load_config(second_path))


def test_config_to_dict_returns_plain_nested_dict(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    write_yaml(config_path, valid_config_dict())

    config = load_config(config_path)
    data = config_to_dict(config)

    assert isinstance(data, dict)
    assert data["model"]["id"] == "Qwen/Qwen2.5-3B-Instruct"
    assert data["dpo"]["beta"] == 0.1


def test_missing_config_file_raises_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigError, match="does not exist"):
        load_config(missing_path)


def test_empty_config_file_raises_clear_error(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="empty"):
        load_config(config_path)


def test_missing_required_section_raises_config_error(tmp_path: Path) -> None:
    data = valid_config_dict()
    del data["model"]

    config_path = tmp_path / "experiment.yaml"
    write_yaml(config_path, data)

    with pytest.raises(ConfigError, match="model must be a mapping"):
        load_config(config_path)


def test_bool_is_not_accepted_as_integer(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["seed"] = True

    config_path = tmp_path / "experiment.yaml"
    write_yaml(config_path, data)

    with pytest.raises(ConfigError, match="seed must be an integer"):
        load_config(config_path)


def test_frozen_evaluation_must_be_greedy(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["evaluation"]["do_sample"] = True

    config_path = tmp_path / "experiment.yaml"
    write_yaml(config_path, data)

    with pytest.raises(ConfigError, match="do_sample must be false"):
        load_config(config_path)


def test_sft_micro_batch_must_stay_one_for_small_gpu(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["sft"]["micro_batch_size"] = 2

    config_path = tmp_path / "experiment.yaml"
    write_yaml(config_path, data)

    with pytest.raises(ConfigError, match="sft.micro_batch_size"):
        load_config(config_path)


def test_dpo_micro_batch_must_stay_one_for_small_gpu(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["dpo"]["micro_batch_size"] = 2

    config_path = tmp_path / "experiment.yaml"
    write_yaml(config_path, data)

    with pytest.raises(ConfigError, match="dpo.micro_batch_size"):
        load_config(config_path)


def test_probability_values_are_validated(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["dpo"]["sample_top_p"] = 1.5

    config_path = tmp_path / "experiment.yaml"
    write_yaml(config_path, data)

    with pytest.raises(ConfigError, match="sample_top_p"):
        load_config(config_path)
