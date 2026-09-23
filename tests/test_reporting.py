from __future__ import annotations

import pytest

from tool_calling_lora_dpo.reporting import (
    add_metric_confidence_intervals,
    aggregate_scores,
    assert_matching_manifest_hashes,
    bootstrap_binary_ci,
    error_counts,
)
from tool_calling_lora_dpo.scoring import score_model_output


GOLD = [{"name": "get_weather", "arguments": {"city": "Sydney", "unit": "celsius"}}]


def scored_examples():
    return [
        score_model_output(
            '[{"name":"get_weather","arguments":{"city":"Sydney","unit":"celsius"}}]',
            GOLD,
        ),
        score_model_output(
            '[{"name":"get_weather","arguments":{"city":"Sydney"}}]',
            GOLD,
        ),
        score_model_output(
            '[{"name":"search_web","arguments":{"query":"weather"}}]',
            GOLD,
        ),
        score_model_output("not json", GOLD),
    ]


def test_aggregate_scores_uses_all_examples_for_main_denominators() -> None:
    aggregate = aggregate_scores(scored_examples())

    assert aggregate["example_count"] == 4
    assert aggregate["metrics"]["json_valid"] == 0.75
    assert aggregate["metrics"]["tool_correct"] == 0.5
    assert aggregate["metrics"]["args_exact"] == 0.25
    assert aggregate["metrics"]["end_to_end_exact"] == 0.25


def test_aggregate_scores_uses_tool_correct_denominator_for_conditional_args() -> None:
    aggregate = aggregate_scores(scored_examples())

    assert aggregate["metrics"]["args_exact_given_tool"] == 0.5


def test_error_counts_are_sorted_and_counted() -> None:
    counts = error_counts(scored_examples())

    assert list(counts) == sorted(counts)
    assert counts == {
        "exact": 1,
        "malformed_json": 1,
        "missing_argument": 1,
        "wrong_tool": 1,
    }


def test_aggregate_scores_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        aggregate_scores([])


def test_bootstrap_binary_ci_is_deterministic() -> None:
    first = bootstrap_binary_ci([True, True, False, False], seed=42, iterations=200)
    second = bootstrap_binary_ci([True, True, False, False], seed=42, iterations=200)

    assert first == second
    assert first.mean == 0.5
    assert 0.0 <= first.lower <= first.mean <= first.upper <= 1.0


def test_bootstrap_binary_ci_validates_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_binary_ci([], seed=42)

    with pytest.raises(ValueError, match="iterations"):
        bootstrap_binary_ci([True], seed=42, iterations=0)

    with pytest.raises(ValueError, match="confidence"):
        bootstrap_binary_ci([True], seed=42, confidence=1.5)


def test_add_metric_confidence_intervals_returns_main_metrics() -> None:
    intervals = add_metric_confidence_intervals(scored_examples(), seed=42, iterations=200)

    assert set(intervals) == {"json_valid", "tool_correct", "args_exact", "end_to_end_exact"}
    assert intervals["end_to_end_exact"]["mean"] == 0.25


def test_assert_matching_manifest_hashes_accepts_identical_hashes() -> None:
    assert_matching_manifest_hashes(
        {
            "baseline": {"metadata": {"manifest_hash": "abc"}},
            "sft": {"metadata": {"manifest_hash": "abc"}},
            "dpo": {"metadata": {"manifest_hash": "abc"}},
        }
    )


def test_assert_matching_manifest_hashes_rejects_missing_hash() -> None:
    with pytest.raises(ValueError, match="missing manifest"):
        assert_matching_manifest_hashes(
            {
                "baseline": {"metadata": {"manifest_hash": "abc"}},
                "sft": {"metadata": {}},
            }
        )


def test_assert_matching_manifest_hashes_rejects_mismatched_hashes() -> None:
    with pytest.raises(ValueError, match="differ"):
        assert_matching_manifest_hashes(
            {
                "baseline": {"metadata": {"manifest_hash": "abc"}},
                "sft": {"metadata": {"manifest_hash": "def"}},
            }
        )
