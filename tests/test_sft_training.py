from __future__ import annotations

import json

import pytest
import torch

from tool_calling_lora_dpo.config import (
    DataConfig,
    DpoConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PromptConfig,
    SftConfig,
)
from tool_calling_lora_dpo.sft_dataset import SftRecord
from tool_calling_lora_dpo.sft_training import (
    AssistantOnlyDataCollator,
    build_sft_training_plan,
    build_training_result,
    configure_tokenizer_for_training,
    is_valid_trainer_checkpoint,
    latest_valid_trainer_checkpoint,
    limit_records,
    optimizer_name_for_transformers,
    records_to_trainer_rows,
    resolve_resume_checkpoint,
    write_training_report,
)


class FakeTrainingTokenizer:
    eos_token_id = 9
    pad_token_id = None
    eos_token = "<eos>"
    padding_side = "left"

    def __init__(self) -> None:
        self._pad_token = None

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        if value == self.eos_token:
            self.pad_token_id = self.eos_token_id

    def __call__(
        self,
        text,
        *,
        return_tensors,
        padding,
        truncation,
        max_length,
    ):
        rows = []
        for item in text:
            length = min(len(item.split()), max_length)
            rows.append(list(range(1, length + 1)))

        max_row_length = max(len(row) for row in rows)
        pad_id = self.pad_token_id
        padded = [row + [pad_id] * (max_row_length - len(row)) for row in rows]
        return {"input_ids": torch.tensor(padded)}


def record(source_id: str = "1") -> SftRecord:
    return SftRecord(
        source_id=source_id,
        text="one two three four",
        token_length=4,
        prompt_token_length=2,
    )


def config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=42,
        model=ModelConfig(
            id="model",
            fallback_id="fallback",
            load_in_4bit=True,
        ),
        data=DataConfig(
            id="dataset",
            validation_size=2,
            test_size=2,
        ),
        prompt=PromptConfig(max_seq_length=16),
        evaluation=EvaluationConfig(
            max_new_tokens=8,
            do_sample=False,
            batch_size=1,
        ),
        sft=SftConfig(
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.0,
            micro_batch_size=1,
            gradient_accumulation_steps=16,
            epochs=1,
            learning_rate=0.0002,
            scheduler="cosine",
            warmup_ratio=0.03,
            optimizer="adamw_8bit",
            assistant_only_loss=True,
            gradient_checkpointing="unsloth",
        ),
        dpo=DpoConfig(
            source_examples=10,
            candidates_per_prompt=4,
            sample_temperature=0.8,
            sample_top_p=0.9,
            beta=0.1,
            micro_batch_size=1,
            gradient_accumulation_steps=16,
            epochs=1,
            learning_rate=0.000005,
        ),
    )


def test_configure_tokenizer_for_training_uses_right_padding_and_eos_pad() -> None:
    tokenizer = FakeTrainingTokenizer()

    configured = configure_tokenizer_for_training(tokenizer)

    assert configured.padding_side == "right"
    assert configured.pad_token == "<eos>"


def test_assistant_only_collator_masks_prompt_and_padding_tokens() -> None:
    tokenizer = configure_tokenizer_for_training(FakeTrainingTokenizer())
    collator = AssistantOnlyDataCollator(
        tokenizer,
        max_seq_length=8,
        assistant_only_loss=True,
    )

    batch = collator(
        [
            records_to_trainer_rows([record("1")])[0],
            {
                "source_id": "2",
                "text": "one two three",
                "token_length": 3,
                "prompt_token_length": 1,
            },
        ]
    )

    assert batch["labels"].tolist() == [
        [-100, -100, 3, 4],
        [-100, 2, 3, -100],
    ]


def test_collator_can_train_on_full_text_when_requested() -> None:
    tokenizer = configure_tokenizer_for_training(FakeTrainingTokenizer())
    collator = AssistantOnlyDataCollator(
        tokenizer,
        max_seq_length=8,
        assistant_only_loss=False,
    )

    batch = collator([records_to_trainer_rows([record("1")])[0]])

    assert batch["labels"].tolist() == [[1, 2, 3, 4]]


def test_limit_records_keeps_deterministic_prefix() -> None:
    records = [record("1"), record("2"), record("3")]

    assert [item.source_id for item in limit_records(records, 2)] == ["1", "2"]


def test_limit_records_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        limit_records([record("1")], 0)


def test_optimizer_name_for_transformers_maps_config_name() -> None:
    assert optimizer_name_for_transformers("adamw_8bit") == "paged_adamw_8bit"


def test_optimizer_name_for_transformers_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        optimizer_name_for_transformers("mystery")


def write_checkpoint_file(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def test_is_valid_trainer_checkpoint_rejects_empty_directories(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()

    assert not is_valid_trainer_checkpoint(checkpoint)


def test_is_valid_trainer_checkpoint_accepts_state_and_adapter_weights(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    write_checkpoint_file(checkpoint / "trainer_state.json")
    write_checkpoint_file(checkpoint / "adapter_model.safetensors")

    assert is_valid_trainer_checkpoint(checkpoint)


def test_latest_valid_trainer_checkpoint_skips_empty_later_directories(tmp_path) -> None:
    output_dir = tmp_path / "adapter"
    write_checkpoint_file(output_dir / "checkpoint-100" / "trainer_state.json")
    write_checkpoint_file(output_dir / "checkpoint-100" / "adapter_model.safetensors")
    write_checkpoint_file(output_dir / "checkpoint-200" / "trainer_state.json")
    write_checkpoint_file(output_dir / "checkpoint-200" / "adapter_model.safetensors")
    (output_dir / "checkpoint-300").mkdir()

    assert latest_valid_trainer_checkpoint(output_dir).name == "checkpoint-200"


def test_resolve_resume_checkpoint_supports_latest_and_checkpoint_name(tmp_path) -> None:
    output_dir = tmp_path / "adapter"
    checkpoint = output_dir / "checkpoint-200"
    write_checkpoint_file(checkpoint / "trainer_state.json")
    write_checkpoint_file(checkpoint / "adapter_model.safetensors")

    assert resolve_resume_checkpoint("latest", output_dir=output_dir) == str(checkpoint)
    assert resolve_resume_checkpoint("checkpoint-200", output_dir=output_dir) == str(checkpoint)


def test_resolve_resume_checkpoint_lists_valid_checkpoints_for_empty_directory(tmp_path) -> None:
    output_dir = tmp_path / "adapter"
    empty_checkpoint = output_dir / "checkpoint-100"
    valid_checkpoint = output_dir / "checkpoint-200"
    empty_checkpoint.mkdir(parents=True)
    write_checkpoint_file(valid_checkpoint / "trainer_state.json")
    write_checkpoint_file(valid_checkpoint / "adapter_model.safetensors")

    with pytest.raises(ValueError, match="checkpoint-200"):
        resolve_resume_checkpoint(empty_checkpoint, output_dir=output_dir)


def test_build_sft_training_plan_uses_config_and_overrides() -> None:
    plan = build_sft_training_plan(
        config=config(),
        output_dir="artifacts/models/sft_lora_smoke",
        max_steps=3,
        logging_steps=1,
        save_steps=2,
        model_id="override-model",
    )

    assert plan.model_id == "override-model"
    assert plan.optimizer == "paged_adamw_8bit"
    assert plan.gradient_checkpointing is True
    assert plan.max_steps == 3


def test_write_training_report_writes_json(tmp_path) -> None:
    plan = build_sft_training_plan(
        config=config(),
        output_dir="adapter",
        max_steps=1,
        logging_steps=1,
        save_steps=1,
    )
    result = build_training_result(
        config=config(),
        plan=plan,
        records=[record("1")],
        metrics={"train_loss": 1.23},
    )
    path = tmp_path / "report.json"

    write_training_report(path, result)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["stage"] == "sft"
    assert saved["train_record_count"] == 1
    assert saved["metrics"]["train_loss"] == 1.23
