from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tool_calling_lora_dpo.data import PreparedExample
from tool_calling_lora_dpo.reporting import aggregate_scores
from tool_calling_lora_dpo.scoring import score_model_output, score_to_dict


def evaluate_predictions(
    examples: list[PreparedExample],
    predictions: list[dict[str, Any]],
    *,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score saved model outputs against prepared examples.

    Predictions are keyed by `source_id`. Results are emitted in the exact order
    of the examples list, which should already be the frozen test-ID order.
    """

    prediction_by_id = _prediction_map(predictions)
    per_example = []
    scores = []

    for example in examples:
        if example.source_id not in prediction_by_id:
            raise ValueError(f"missing prediction for source_id={example.source_id}")

        model_output = prediction_by_id[example.source_id]
        score = score_model_output(model_output, example.gold_answers)
        scores.append(score)
        per_example.append(
            {
                "source_id": example.source_id,
                "model_output": model_output,
                "score": score_to_dict(score),
            }
        )

    return {
        "stage": stage,
        "metadata": metadata or {},
        "aggregate": aggregate_scores(scores),
        "per_example": per_example,
    }


def evaluation_sha256(evaluation: dict[str, Any]) -> str:
    """Create a stable hash of an evaluation artifact."""

    payload = json.dumps(evaluation, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_evaluation(path: str | Path, evaluation: dict[str, Any]) -> None:
    """Write an evaluation artifact as stable pretty JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")


def _prediction_map(predictions: list[dict[str, Any]]) -> dict[str, str]:
    prediction_by_id: dict[str, str] = {}

    for prediction in predictions:
        source_id = prediction.get("source_id")
        model_output = prediction.get("model_output")

        if not isinstance(source_id, str) or not source_id:
            raise ValueError("prediction source_id must be a non-empty string")

        if not isinstance(model_output, str):
            raise ValueError(f"prediction model_output must be a string for source_id={source_id}")

        if source_id in prediction_by_id:
            raise ValueError(f"duplicate prediction for source_id={source_id}")

        prediction_by_id[source_id] = model_output

    return prediction_by_id
