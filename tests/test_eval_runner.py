from __future__ import annotations

import json

import pytest

from tests.test_inference import FakeModel, FakeTokenizer
from tool_calling_lora_dpo.data import PreparedExample
from tool_calling_lora_dpo.eval_runner import (
    EvaluationIdentity,
    build_evaluation_identity,
    build_final_evaluation_artifact,
    load_checkpoint,
    run_resumable_evaluation,
    save_checkpoint,
)
from tool_calling_lora_dpo.inference import GenerationConfig


def example(source_id: str) -> PreparedExample:
    return PreparedExample(
        source_id=source_id,
        query=f"Find item {source_id}",
        tools=[{"name": "lookup", "parameters": {}}],
        gold_answers=[{"name": "lookup", "arguments": {"id": "1"}}],
    )


def identity(examples: list[PreparedExample] | None = None) -> EvaluationIdentity:
    examples = examples or [example("1")]
    return build_evaluation_identity(
        stage="baseline",
        model_id="model",
        split_name="test",
        config_hash="config",
        manifest_hash="manifest",
        examples=examples,
        generation_config=GenerationConfig(
            max_new_tokens=8,
            do_sample=False,
            max_seq_length=10,
        ),
    )


def test_build_evaluation_identity_hashes_example_ids() -> None:
    first = identity([example("1"), example("2")])
    second = identity([example("2"), example("1")])

    assert first.example_ids_hash == second.example_ids_hash
    assert first.stage == "baseline"
    assert first.max_new_tokens == 8


def test_run_resumable_evaluation_writes_checkpoint(tmp_path) -> None:
    examples = [example("1"), example("2")]
    checkpoint_path = tmp_path / "checkpoint.json"

    result = run_resumable_evaluation(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        examples=examples,
        generation_config=GenerationConfig(
            max_new_tokens=8,
            do_sample=False,
            max_seq_length=10,
        ),
        identity=identity(examples),
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
    )

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert len(result.records) == 2
    assert len(checkpoint["records"]) == 2


def test_run_resumable_evaluation_skips_completed_checkpoint_records(tmp_path) -> None:
    examples = [example("1"), example("2")]
    checkpoint_path = tmp_path / "checkpoint.json"
    run_identity = identity(examples)

    first = run_resumable_evaluation(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        examples=[example("1")],
        generation_config=GenerationConfig(
            max_new_tokens=8,
            do_sample=False,
            max_seq_length=10,
        ),
        identity=identity([example("1")]),
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
    )
    save_checkpoint(checkpoint_path, identity=run_identity, records=first.records)

    second = run_resumable_evaluation(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        examples=examples,
        generation_config=GenerationConfig(
            max_new_tokens=8,
            do_sample=False,
            max_seq_length=10,
        ),
        identity=run_identity,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
    )

    assert len(second.records) == 2
    assert second.metadata["resumed_records"] == 1
    assert second.metadata["generated_records_this_run"] == 1


def test_load_checkpoint_rejects_identity_mismatch(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    save_checkpoint(checkpoint_path, identity=identity(), records=[])

    mismatched = EvaluationIdentity(
        **{
            **identity().__dict__,
            "model_id": "different",
        }
    )

    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(checkpoint_path, mismatched)


def test_build_final_evaluation_artifact_adds_metrics_and_ci(tmp_path) -> None:
    examples = [example("1")]
    checkpoint_path = tmp_path / "checkpoint.json"
    run_identity = identity(examples)
    run_result = run_resumable_evaluation(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        examples=examples,
        generation_config=GenerationConfig(
            max_new_tokens=8,
            do_sample=False,
            max_seq_length=10,
        ),
        identity=run_identity,
        checkpoint_path=checkpoint_path,
    )

    evaluation = build_final_evaluation_artifact(
        examples=examples,
        run_result=run_result,
        identity=run_identity,
        bootstrap_seed=42,
    )

    assert evaluation["aggregate"]["metrics"]["end_to_end_exact"] == 1.0
    assert "confidence_intervals" in evaluation["aggregate"]
    assert len(evaluation["metadata"]["evaluation_hash"]) == 64


def test_run_resumable_evaluation_validates_checkpoint_interval(tmp_path) -> None:
    with pytest.raises(ValueError, match="checkpoint_every"):
        run_resumable_evaluation(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            examples=[example("1")],
            generation_config=GenerationConfig(
                max_new_tokens=8,
                do_sample=False,
                max_seq_length=10,
            ),
            identity=identity(),
            checkpoint_path=tmp_path / "checkpoint.json",
            checkpoint_every=0,
        )
