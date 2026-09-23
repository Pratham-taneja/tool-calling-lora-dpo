from __future__ import annotations

import json
from pathlib import Path

import pytest

from tool_calling_lora_dpo.data import (
    DataValidationError,
    PreparedExample,
    SplitIds,
    assert_disjoint_splits,
    example_from_json_dict,
    example_to_json_dict,
    ids_sha256,
    make_deterministic_splits,
    parse_source_row,
    parse_source_rows,
    read_jsonl,
    split_manifest,
    write_jsonl,
)


def example_row(
    *,
    source_id: int | str = 123,
    query: str = "What is the weather in Sydney?",
    tools: list[dict] | None = None,
    answers: list[dict] | None = None,
) -> dict:
    """Build a realistic source row with JSON-encoded tools and answers."""

    if tools is None:
        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            }
        ]

    if answers is None:
        answers = [
            {
                "name": "get_weather",
                "arguments": {
                    "city": "Sydney",
                },
            }
        ]

    return {
        "id": source_id,
        "query": query,
        "tools": json.dumps(tools),
        "answers": json.dumps(answers),
    }


def test_parse_source_row_returns_prepared_example() -> None:
    prepared = parse_source_row(example_row())

    assert prepared == PreparedExample(
        source_id="123",
        query="What is the weather in Sydney?",
        tools=[
            {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            }
        ],
        gold_answers=[
            {
                "name": "get_weather",
                "arguments": {
                    "city": "Sydney",
                },
            }
        ],
    )


def test_parse_source_row_supports_nested_function_tool_schema() -> None:
    row = example_row(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_flights",
                    "description": "Search for flights.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string"},
                            "destination": {"type": "string"},
                        },
                    },
                },
            }
        ],
        answers=[
            {
                "name": "search_flights",
                "arguments": {
                    "origin": "SYD",
                    "destination": "MEL",
                },
            }
        ],
    )

    prepared = parse_source_row(row)

    assert prepared.tools[0]["function"]["name"] == "search_flights"
    assert prepared.gold_answers[0]["arguments"]["destination"] == "MEL"


def test_parse_source_rows_rejects_duplicate_ids() -> None:
    rows = [
        example_row(source_id=1),
        example_row(source_id=1),
    ]

    with pytest.raises(DataValidationError, match="unique"):
        parse_source_rows(rows)


@pytest.mark.parametrize(
    ("field_name", "bad_value", "expected_message"),
    [
        ("query", "", "query must be a non-empty string"),
        ("tools", "not-json", "tools is not valid JSON"),
        ("answers", "not-json", "answers is not valid JSON"),
        ("tools", json.dumps({}), "tools must decode to a list"),
        ("answers", json.dumps({}), "answers must decode to a list"),
    ],
)
def test_parse_source_row_rejects_bad_basic_fields(
    field_name: str,
    bad_value: str,
    expected_message: str,
) -> None:
    row = example_row()
    row[field_name] = bad_value

    with pytest.raises(DataValidationError, match=expected_message):
        parse_source_row(row)


def test_parse_source_row_requires_id() -> None:
    row = example_row()
    del row["id"]

    with pytest.raises(DataValidationError, match="id is required"):
        parse_source_row(row)


def test_parse_source_row_rejects_invalid_tool_item() -> None:
    row = example_row(tools=[None])

    with pytest.raises(DataValidationError, match=r"tools\[0\] must be an object"):
        parse_source_row(row)


def test_parse_source_row_rejects_tool_without_name() -> None:
    row = example_row(tools=[{"description": "Missing a name."}])

    with pytest.raises(DataValidationError, match="non-empty name"):
        parse_source_row(row)


def test_parse_source_row_rejects_non_object_tool_parameters() -> None:
    row = example_row(tools=[{"name": "bad_tool", "parameters": []}])

    with pytest.raises(DataValidationError, match="parameters must be an object"):
        parse_source_row(row)


def test_parse_source_row_rejects_invalid_answer_item() -> None:
    row = example_row(answers=[None])

    with pytest.raises(DataValidationError, match=r"answers\[0\] must be an object"):
        parse_source_row(row)


def test_parse_source_row_rejects_answer_without_name() -> None:
    row = example_row(answers=[{"arguments": {}}])

    with pytest.raises(DataValidationError, match="non-empty name"):
        parse_source_row(row)


def test_parse_source_row_rejects_answer_without_object_arguments() -> None:
    row = example_row(answers=[{"name": "get_weather", "arguments": []}])

    with pytest.raises(DataValidationError, match="arguments must be an object"):
        parse_source_row(row)


def test_make_deterministic_splits_returns_expected_sizes() -> None:
    splits = make_deterministic_splits(
        [str(index) for index in range(20)],
        validation_size=5,
        test_size=3,
        seed=42,
    )

    assert len(splits.train_ids) == 12
    assert len(splits.validation_ids) == 5
    assert len(splits.test_ids) == 3


def test_make_deterministic_splits_is_stable() -> None:
    ids = [str(index) for index in range(100)]

    first = make_deterministic_splits(ids, validation_size=10, test_size=5, seed=42)
    second = make_deterministic_splits(ids, validation_size=10, test_size=5, seed=42)

    assert first == second


def test_make_deterministic_splits_does_not_mutate_input() -> None:
    ids = [str(index) for index in range(20)]
    original = ids.copy()

    make_deterministic_splits(ids, validation_size=5, test_size=3, seed=42)

    assert ids == original


def test_make_deterministic_splits_rejects_duplicate_ids() -> None:
    ids = ["1", "1", "2", "3"]

    with pytest.raises(DataValidationError, match="duplicate"):
        make_deterministic_splits(ids, validation_size=1, test_size=1, seed=42)


def test_make_deterministic_splits_requires_training_examples() -> None:
    ids = ["1", "2", "3"]

    with pytest.raises(DataValidationError, match="leave training examples"):
        make_deterministic_splits(ids, validation_size=2, test_size=1, seed=42)


def test_assert_disjoint_splits_rejects_overlap() -> None:
    splits = SplitIds(
        train_ids=["1", "2"],
        validation_ids=["3"],
        test_ids=["2"],
    )

    with pytest.raises(DataValidationError, match="train and test"):
        assert_disjoint_splits(splits)


def test_ids_sha256_is_order_independent() -> None:
    first = ids_sha256(["1", "2", "3"])
    second = ids_sha256(["3", "2", "1"])

    assert first == second
    assert len(first) == 64


def test_split_manifest_contains_counts_hashes_and_ids() -> None:
    splits = SplitIds(
        train_ids=["1", "2", "3"],
        validation_ids=["4"],
        test_ids=["5"],
    )

    manifest = split_manifest(splits, seed=42)

    assert manifest["seed"] == 42
    assert manifest["counts"] == {
        "train": 3,
        "validation": 1,
        "test": 1,
        "total": 5,
    }
    assert set(manifest["hashes"]) == {"train", "validation", "test"}
    assert manifest["ids"]["train"] == ["1", "2", "3"]


def test_example_to_json_dict_is_json_serializable() -> None:
    example = parse_source_row(example_row())
    row = example_to_json_dict(example)

    encoded = json.dumps(row)

    assert json.loads(encoded)["source_id"] == "123"


def test_example_from_json_dict_round_trips_prepared_example() -> None:
    example = parse_source_row(example_row())

    loaded = example_from_json_dict(example_to_json_dict(example))

    assert loaded == example


def test_write_jsonl_creates_parent_directory_and_writes_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "examples.jsonl"

    write_jsonl(
        path,
        [
            {"source_id": "2", "query": "second"},
            {"source_id": "1", "query": "first"},
        ],
    )

    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {"query": "second", "source_id": "2"}
    assert json.loads(lines[1]) == {"query": "first", "source_id": "1"}


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"source_id":"1"}\n\n{"source_id":"2"}\n', encoding="utf-8")

    assert read_jsonl(path) == [{"source_id": "1"}, {"source_id": "2"}]


def test_read_jsonl_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="not valid JSON"):
        read_jsonl(path)
