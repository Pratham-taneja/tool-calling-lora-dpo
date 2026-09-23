from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tool_calling_lora_dpo.scoring import ScoreResult


@dataclass(frozen=True)
class ConfidenceInterval:
    """Percentile bootstrap confidence interval."""

    mean: float
    lower: float
    upper: float


def aggregate_scores(scores: Sequence[ScoreResult]) -> dict[str, Any]:
    """Aggregate per-example score objects into report-ready metrics."""

    if not scores:
        raise ValueError("cannot aggregate an empty score list")

    total = len(scores)
    tool_correct = [score for score in scores if score.tool_correct]

    metrics = {
        "json_valid": _mean(score.json_valid for score in scores),
        "schema_valid": _mean(score.schema_valid for score in scores),
        "call_count_correct": _mean(score.call_count_correct for score in scores),
        "tool_correct": _mean(score.tool_correct for score in scores),
        "args_exact": _mean(score.args_exact for score in scores),
        "end_to_end_exact": _mean(score.end_to_end_exact for score in scores),
        "args_exact_given_tool": _mean(score.args_exact_given_tool for score in tool_correct)
        if tool_correct
        else None,
    }

    return {
        "example_count": total,
        "metrics": metrics,
        "error_counts": error_counts(scores),
    }


def error_counts(scores: Sequence[ScoreResult]) -> dict[str, int]:
    """Count typed scorer outcomes for error analysis."""

    counts: dict[str, int] = {}
    for score in scores:
        counts[score.error_category] = counts.get(score.error_category, 0) + 1
    return dict(sorted(counts.items()))


def bootstrap_binary_ci(
    values: Sequence[bool],
    *,
    seed: int,
    iterations: int = 1000,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Compute a simple percentile bootstrap CI for a binary metric."""

    if not values:
        raise ValueError("cannot bootstrap an empty value list")

    if iterations <= 0:
        raise ValueError("iterations must be positive")

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    rng = random.Random(seed)
    sample_size = len(values)
    means = []

    for _ in range(iterations):
        sample = [values[rng.randrange(sample_size)] for _ in range(sample_size)]
        means.append(_mean(sample))

    means.sort()
    alpha = 1.0 - confidence
    lower_index = int((alpha / 2.0) * iterations)
    upper_index = min(iterations - 1, int((1.0 - alpha / 2.0) * iterations))

    return ConfidenceInterval(
        mean=_mean(values),
        lower=means[lower_index],
        upper=means[upper_index],
    )


def add_metric_confidence_intervals(
    scores: Sequence[ScoreResult],
    *,
    seed: int,
    iterations: int = 1000,
) -> dict[str, dict[str, float]]:
    """Compute CIs for the main binary portfolio metrics."""

    metric_values = {
        "json_valid": [score.json_valid for score in scores],
        "tool_correct": [score.tool_correct for score in scores],
        "args_exact": [score.args_exact for score in scores],
        "end_to_end_exact": [score.end_to_end_exact for score in scores],
    }

    intervals: dict[str, dict[str, float]] = {}
    for name, values in metric_values.items():
        interval = bootstrap_binary_ci(values, seed=seed, iterations=iterations)
        intervals[name] = {
            "mean": interval.mean,
            "lower": interval.lower,
            "upper": interval.upper,
        }

    return intervals


def assert_matching_manifest_hashes(stage_results: dict[str, dict[str, Any]]) -> None:
    """Reject final comparisons that were not run on the same frozen split."""

    hashes = {
        stage: result.get("metadata", {}).get("manifest_hash")
        for stage, result in stage_results.items()
    }
    missing = sorted(stage for stage, value in hashes.items() if not value)
    if missing:
        raise ValueError(f"missing manifest hash for stages: {missing}")

    unique_hashes = set(hashes.values())
    if len(unique_hashes) != 1:
        raise ValueError(f"stage manifest hashes differ: {hashes}")


def _mean(values: Sequence[bool | None]) -> float:
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        raise ValueError("cannot compute mean for empty values")
    return sum(1 for value in clean_values if value) / len(clean_values)
