from __future__ import annotations

import json

import pytest

from tool_calling_lora_dpo.data import PreparedExample
from tool_calling_lora_dpo.sft_dataset import (
    build_exclusion_manifest,
    build_sft_dataset,
    sft_record_from_json_dict,
    write_exclusion_manifest,
    write_sft_records,
)


class LengthTokenizer:
    """Fake chat-template tokenizer controlled by query text like `length=5`."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        user_message = next(
            message["content"] for message in conversation if message["role"] == "user"
        )
        length = int(user_message.split("length=", maxsplit=1)[1])
        if tokenize and add_generation_prompt:
            length = max(1, length - 1)
        rendered = "|".join(f"{message['role']}:{message['content']}" for message in conversation)

        if add_generation_prompt:
            rendered += "|assistant:"

        if tokenize:
            return list(range(length))

        return rendered


def example(source_id: str, length: int) -> PreparedExample:
    return PreparedExample(
        source_id=source_id,
        query=f"synthetic length={length}",
        tools=[{"name": "lookup", "parameters": {}}],
        gold_answers=[{"name": "lookup", "arguments": {"id": source_id}}],
    )


def test_build_sft_dataset_keeps_examples_at_or_below_limit() -> None:
    result = build_sft_dataset(
        [example("1", 5), example("2", 10)],
        LengthTokenizer(),
        max_seq_length=10,
    )

    assert [record.source_id for record in result.records] == ["1", "2"]
    assert result.exclusions == []


def test_build_sft_dataset_excludes_examples_above_limit() -> None:
    result = build_sft_dataset(
        [example("1", 5), example("2", 11)],
        LengthTokenizer(),
        max_seq_length=10,
    )

    assert [record.source_id for record in result.records] == ["1"]
    assert len(result.exclusions) == 1
    assert result.exclusions[0].source_id == "2"
    assert result.exclusions[0].token_length == 11
    assert result.exclusions[0].reason == "training_prompt_exceeds_max_seq_length"


def test_build_sft_dataset_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_sft_dataset([example("1", 5)], LengthTokenizer(), max_seq_length=0)


def test_kept_records_include_rendered_training_text() -> None:
    result = build_sft_dataset([example("1", 5)], LengthTokenizer(), max_seq_length=10)

    assert len(result.records) == 1
    assert result.records[0].token_length == 5
    assert result.records[0].prompt_token_length == 4
    assert "assistant" in result.records[0].text
    assert "lookup" in result.records[0].text


def test_build_exclusion_manifest_records_counts_hashes_and_reasons() -> None:
    result = build_sft_dataset(
        [example("1", 5), example("2", 11)],
        LengthTokenizer(),
        max_seq_length=10,
    )

    manifest = build_exclusion_manifest(
        result=result,
        input_example_count=2,
        max_seq_length=10,
        model_id="model",
        config_hash="abc",
    )

    assert manifest["source_split"] == "train"
    assert manifest["counts"] == {"input": 2, "kept": 1, "excluded": 1}
    assert manifest["max_seq_length"] == 10
    assert len(manifest["hashes"]["kept_ids"]) == 64
    assert manifest["exclusions"][0]["source_id"] == "2"


def test_write_sft_records_writes_jsonl(tmp_path) -> None:
    result = build_sft_dataset([example("1", 5)], LengthTokenizer(), max_seq_length=10)
    path = tmp_path / "sft_train.jsonl"

    write_sft_records(path, result.records)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["source_id"] == "1"
    assert rows[0]["token_length"] == 5
    assert rows[0]["prompt_token_length"] == 4


def test_write_exclusion_manifest_writes_json(tmp_path) -> None:
    path = tmp_path / "sft_exclusions.json"
    manifest = {"counts": {"kept": 1, "excluded": 0}}

    write_exclusion_manifest(path, manifest)

    assert json.loads(path.read_text(encoding="utf-8")) == manifest


def test_sft_record_from_json_dict_validates_prompt_length() -> None:
    row = {
        "source_id": "1",
        "text": "prompt and answer",
        "token_length": 5,
        "prompt_token_length": 4,
    }

    record = sft_record_from_json_dict(row)

    assert record.source_id == "1"
    assert record.prompt_token_length == 4


def test_sft_record_from_json_dict_rejects_missing_prompt_length() -> None:
    with pytest.raises(ValueError, match="prompt_token_length"):
        sft_record_from_json_dict(
            {
                "source_id": "1",
                "text": "prompt and answer",
                "token_length": 5,
            }
        )
