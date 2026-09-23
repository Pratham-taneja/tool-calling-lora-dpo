from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from tool_calling_lora_dpo.data import PreparedExample, example_from_json_dict, read_jsonl


def load_prepared_examples(path: str | Path) -> list[PreparedExample]:
    """Load processed examples created by scripts/prepare_data.py."""

    return [example_from_json_dict(row) for row in read_jsonl(path)]


def summarize_examples(examples: list[PreparedExample]) -> dict[str, Any]:
    """Build a non-sensitive aggregate profile of prepared examples."""

    if not examples:
        raise ValueError("cannot summarize an empty example list")

    tool_counts = [len(example.tools) for example in examples]
    answer_counts = [len(example.gold_answers) for example in examples]
    query_lengths = [len(example.query) for example in examples]
    tool_name_counter = Counter(
        _tool_name(tool)
        for example in examples
        for tool in example.tools
        if _tool_name(tool) is not None
    )
    answer_name_counter = Counter(
        answer["name"]
        for example in examples
        for answer in example.gold_answers
        if isinstance(answer.get("name"), str)
    )

    return {
        "example_count": len(examples),
        "unique_source_ids": len({example.source_id for example in examples}),
        "tool_count": numeric_summary(tool_counts),
        "answer_count": numeric_summary(answer_counts),
        "query_character_length": numeric_summary(query_lengths),
        "unique_tool_names": len(tool_name_counter),
        "unique_answer_names": len(answer_name_counter),
        "tool_count_distribution": dict(sorted(Counter(tool_counts).items())),
        "answer_count_distribution": dict(sorted(Counter(answer_counts).items())),
    }


def summarize_processed_splits(processed_dir: str | Path) -> dict[str, Any]:
    """Profile train/validation/test JSONL files and verify ID disjointness."""

    directory = Path(processed_dir)
    split_examples = {
        "train": load_prepared_examples(directory / "train.jsonl"),
        "validation": load_prepared_examples(directory / "validation.jsonl"),
        "test": load_prepared_examples(directory / "test.jsonl"),
    }

    id_sets = {
        split: {example.source_id for example in examples}
        for split, examples in split_examples.items()
    }
    overlaps = {
        "train_validation": sorted(id_sets["train"] & id_sets["validation"]),
        "train_test": sorted(id_sets["train"] & id_sets["test"]),
        "validation_test": sorted(id_sets["validation"] & id_sets["test"]),
    }

    return {
        "splits": {
            split: summarize_examples(examples)
            for split, examples in split_examples.items()
        },
        "total_examples": sum(len(examples) for examples in split_examples.values()),
        "split_overlaps": {name: len(ids) for name, ids in overlaps.items()},
    }


def numeric_summary(values: list[int]) -> dict[str, float | int]:
    """Return compact descriptive statistics for integer values."""

    if not values:
        raise ValueError("cannot summarize empty values")

    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": percentile(ordered, 50),
        "p90": percentile(ordered, 90),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
        "max": ordered[-1],
        "mean": sum(values) / len(values),
    }


def percentile(sorted_values: list[int], percent: int) -> int:
    """Nearest-rank style percentile for simple dataset summaries."""

    if not sorted_values:
        raise ValueError("cannot compute percentile for empty values")

    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")

    index = round((percent / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _tool_name(tool: dict[str, Any]) -> str | None:
    if isinstance(tool.get("name"), str):
        return tool["name"]

    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]

    return None
