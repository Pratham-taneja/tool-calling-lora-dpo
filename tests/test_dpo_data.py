from __future__ import annotations

import json

import pytest

from tool_calling_lora_dpo.data import PreparedExample
from tool_calling_lora_dpo.dpo_data import (
    CandidateRecord,
    build_candidate_identity,
    build_dpo_dataset,
    read_candidate_checkpoint,
    select_dpo_sources,
    validate_candidate_prefix,
    write_candidate_checkpoint,
)


class FakeTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, *, add_special_tokens=False):
        return {"input_ids": text.split()}


def example(source_id: str) -> PreparedExample:
    return PreparedExample(
        source_id=source_id,
        query="query",
        tools=[{"name": "weather"}],
        gold_answers=[{"name": "weather", "arguments": {"city": "Sydney"}}],
    )


def test_source_selection_is_deterministic_and_training_only() -> None:
    examples = [example(str(index)) for index in range(10)]
    first = select_dpo_sources(examples, eligible_ids={str(i) for i in range(8)}, count=4, seed=42)
    second = select_dpo_sources(examples, eligible_ids={str(i) for i in range(8)}, count=4, seed=42)
    assert [item.source_id for item in first] == [item.source_id for item in second]
    assert {item.source_id for item in first} <= {str(i) for i in range(8)}


def test_candidate_checkpoint_round_trip_and_identity_guard(tmp_path) -> None:
    identity = build_candidate_identity(
        config_hash="config",
        adapter_sha256="adapter",
        source_ids=["1"],
        candidates_per_prompt=4,
        temperature=0.8,
        top_p=0.9,
        max_new_tokens=32,
        seed=42,
    )
    records = [CandidateRecord("1", "prompt ", [], ["answer"])]
    path = tmp_path / "checkpoint.json"
    write_candidate_checkpoint(path, identity=identity, records=records)
    assert read_candidate_checkpoint(path, expected_identity=identity) == records
    with pytest.raises(ValueError, match="identity"):
        read_candidate_checkpoint(path, expected_identity={**identity, "seed": 7})


def test_candidate_prefix_is_validated() -> None:
    with pytest.raises(ValueError, match="prefix"):
        validate_candidate_prefix(
            [CandidateRecord("2", "prompt", [], ["answer"])],
            [example("1"), example("2")],
        )


def test_build_dpo_dataset_filters_long_candidates_and_prevents_leakage() -> None:
    gold = [{"name": "weather", "arguments": {"city": "Sydney"}}]
    bad = json.dumps(
        [{"name": "weather", "arguments": {"city": "Melbourne"}}],
        separators=(",", ":"),
    )
    records = [CandidateRecord("1", "prompt ", gold, ["too many words here", bad])]
    result = build_dpo_dataset(
        records,
        tokenizer=FakeTokenizer(),
        max_seq_length=4,
        train_ids={"1"},
        validation_ids={"2"},
        test_ids={"3"},
    )
    assert result.over_length_candidates == 1
    assert len(result.pairs) == 1
    assert result.pairs[0].source_id == "1"
