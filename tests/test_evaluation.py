from __future__ import annotations

import json

import pytest

from tool_calling_lora_dpo.data import PreparedExample
from tool_calling_lora_dpo.evaluation import (
    evaluate_predictions,
    evaluation_sha256,
    write_evaluation,
)


def example(source_id: str) -> PreparedExample:
    return PreparedExample(
        source_id=source_id,
        query="What is the weather in Sydney?",
        tools=[{"name": "get_weather", "parameters": {}}],
        gold_answers=[
            {"name": "get_weather", "arguments": {"city": "Sydney", "unit": "celsius"}}
        ],
    )


def test_evaluate_predictions_scores_in_example_order() -> None:
    examples = [example("1"), example("2")]
    predictions = [
        {
            "source_id": "2",
            "model_output": '[{"name":"get_weather","arguments":{"city":"Sydney"}}]',
        },
        {
            "source_id": "1",
            "model_output": (
                '[{"name":"get_weather","arguments":{"city":"Sydney","unit":"celsius"}}]'
            ),
        },
    ]

    evaluation = evaluate_predictions(examples, predictions, stage="baseline")

    assert evaluation["stage"] == "baseline"
    assert [row["source_id"] for row in evaluation["per_example"]] == ["1", "2"]
    assert evaluation["aggregate"]["metrics"]["end_to_end_exact"] == 0.5


def test_evaluate_predictions_rejects_missing_prediction() -> None:
    with pytest.raises(ValueError, match="missing prediction"):
        evaluate_predictions([example("1")], [], stage="baseline")


def test_evaluate_predictions_rejects_duplicate_prediction() -> None:
    predictions = [
        {"source_id": "1", "model_output": "[]"},
        {"source_id": "1", "model_output": "[]"},
    ]

    with pytest.raises(ValueError, match="duplicate"):
        evaluate_predictions([example("1")], predictions, stage="baseline")


def test_evaluation_hash_is_stable() -> None:
    evaluation = evaluate_predictions(
        [example("1")],
        [
            {
                "source_id": "1",
                "model_output": (
                    '[{"name":"get_weather","arguments":{"city":"Sydney","unit":"celsius"}}]'
                ),
            }
        ],
        stage="baseline",
    )

    assert evaluation_sha256(evaluation) == evaluation_sha256(evaluation)
    assert len(evaluation_sha256(evaluation)) == 64


def test_write_evaluation_creates_json_file(tmp_path) -> None:
    path = tmp_path / "nested" / "eval.json"
    evaluation = {"stage": "baseline", "aggregate": {}}

    write_evaluation(path, evaluation)

    assert json.loads(path.read_text(encoding="utf-8")) == evaluation
