from __future__ import annotations

import pytest

from tool_calling_lora_dpo.analysis import (
    numeric_summary,
    percentile,
    summarize_examples,
    summarize_processed_splits,
)
from tool_calling_lora_dpo.data import PreparedExample, example_to_json_dict, write_jsonl


def example(source_id: str, *, answer_count: int = 1) -> PreparedExample:
    return PreparedExample(
        source_id=source_id,
        query="Find weather for Sydney",
        tools=[{"name": "get_weather", "parameters": {}}],
        gold_answers=[
            {"name": "get_weather", "arguments": {"city": f"Sydney {index}"}}
            for index in range(answer_count)
        ],
    )


def test_summarize_examples_profiles_counts_without_raw_content() -> None:
    summary = summarize_examples([example("1"), example("2", answer_count=2)])

    assert summary["example_count"] == 2
    assert summary["unique_source_ids"] == 2
    assert summary["unique_tool_names"] == 1
    assert summary["unique_answer_names"] == 1
    assert summary["answer_count_distribution"] == {1: 1, 2: 1}


def test_summarize_examples_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize_examples([])


def test_summarize_processed_splits_checks_overlap_counts(tmp_path) -> None:
    processed = tmp_path / "processed"
    write_jsonl(processed / "train.jsonl", [example_to_json_dict(example("1"))])
    write_jsonl(processed / "validation.jsonl", [example_to_json_dict(example("2"))])
    write_jsonl(processed / "test.jsonl", [example_to_json_dict(example("3"))])

    profile = summarize_processed_splits(processed)

    assert profile["total_examples"] == 3
    assert profile["split_overlaps"] == {
        "train_test": 0,
        "train_validation": 0,
        "validation_test": 0,
    }


def test_numeric_summary_reports_descriptive_stats() -> None:
    summary = numeric_summary([1, 2, 3, 4, 5])

    assert summary["min"] == 1
    assert summary["p50"] == 3
    assert summary["max"] == 5
    assert summary["mean"] == 3


def test_percentile_validates_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)

    with pytest.raises(ValueError, match="between"):
        percentile([1], -1)
