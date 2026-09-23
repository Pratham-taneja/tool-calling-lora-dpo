from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class DataValidationError(ValueError):
    """Raised when a source dataset row cannot be safely used."""


@dataclass(frozen=True)
class PreparedExample:
    """A validated row after parsing the dataset's JSON-encoded fields."""

    source_id: str
    query: str
    tools: list[dict[str, Any]]
    gold_answers: list[dict[str, Any]]


@dataclass(frozen=True)
class SplitIds:
    """Deterministic train/validation/test ID lists."""

    train_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]


def parse_source_row(row: dict[str, Any]) -> PreparedExample:
    """Parse and validate one raw xLAM dataset row.

    The source dataset stores `tools` and `answers` as JSON strings. We parse
    them exactly once here, then pass structured Python data through the rest of
    the pipeline. That avoids accidental re-parsing differences later.
    """

    source_id = _require_source_id(row)
    query = _require_non_empty_string(row.get("query"), "query")
    tools = _parse_json_list(row.get("tools"), "tools")
    answers = _parse_json_list(row.get("answers"), "answers")

    _validate_tools(tools)
    _validate_answers(answers)

    return PreparedExample(
        source_id=source_id,
        query=query,
        tools=tools,
        gold_answers=answers,
    )


def parse_source_rows(rows: list[dict[str, Any]]) -> list[PreparedExample]:
    """Parse many rows and reject duplicate source IDs."""

    examples = [parse_source_row(row) for row in rows]
    ids = [example.source_id for example in examples]

    if len(ids) != len(set(ids)):
        raise DataValidationError("source IDs must be unique")

    return examples


def make_deterministic_splits(
    source_ids: list[str],
    *,
    validation_size: int,
    test_size: int,
    seed: int,
) -> SplitIds:
    """Create deterministic leakage-free train/validation/test splits.

    We shuffle a copy of the IDs with a local random generator so the function
    never mutates caller-owned lists and never depends on global random state.
    """

    if len(source_ids) != len(set(source_ids)):
        raise DataValidationError("cannot split duplicate source IDs")

    if validation_size <= 0:
        raise DataValidationError("validation_size must be positive")

    if test_size <= 0:
        raise DataValidationError("test_size must be positive")

    if validation_size + test_size >= len(source_ids):
        raise DataValidationError("validation_size + test_size must leave training examples")

    shuffled_ids = [str(source_id) for source_id in source_ids]
    rng = random.Random(seed)
    rng.shuffle(shuffled_ids)

    test_ids = sorted(shuffled_ids[:test_size])
    validation_ids = sorted(shuffled_ids[test_size : test_size + validation_size])
    train_ids = sorted(shuffled_ids[test_size + validation_size :])

    splits = SplitIds(
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
    )

    assert_disjoint_splits(splits)
    return splits


def assert_disjoint_splits(splits: SplitIds) -> None:
    """Fail loudly if any ID appears in more than one split."""

    train = set(splits.train_ids)
    validation = set(splits.validation_ids)
    test = set(splits.test_ids)

    if train & validation:
        raise DataValidationError("train and validation splits overlap")

    if train & test:
        raise DataValidationError("train and test splits overlap")

    if validation & test:
        raise DataValidationError("validation and test splits overlap")


def ids_sha256(source_ids: list[str]) -> str:
    """Hash a sorted list of IDs for reproducible manifests."""

    canonical = json.dumps(sorted(source_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def split_manifest(splits: SplitIds, *, seed: int) -> dict[str, Any]:
    """Build the manifest saved beside processed data."""

    assert_disjoint_splits(splits)

    return {
        "seed": seed,
        "counts": {
            "train": len(splits.train_ids),
            "validation": len(splits.validation_ids),
            "test": len(splits.test_ids),
            "total": len(splits.train_ids) + len(splits.validation_ids) + len(splits.test_ids),
        },
        "hashes": {
            "train": ids_sha256(splits.train_ids),
            "validation": ids_sha256(splits.validation_ids),
            "test": ids_sha256(splits.test_ids),
        },
        "ids": {
            "train": splits.train_ids,
            "validation": splits.validation_ids,
            "test": splits.test_ids,
        },
    }


def example_to_json_dict(example: PreparedExample) -> dict[str, Any]:
    """Convert a prepared example into a JSONL-friendly dictionary."""

    return asdict(example)


def example_from_json_dict(row: dict[str, Any]) -> PreparedExample:
    """Load a prepared example from a processed JSONL row."""

    return PreparedExample(
        source_id=_require_non_empty_string(row.get("source_id"), "source_id"),
        query=_require_non_empty_string(row.get("query"), "query"),
        tools=_require_list_of_dicts(row.get("tools"), "tools"),
        gold_answers=_require_list_of_dicts(row.get("gold_answers"), "gold_answers"),
    )


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL with stable key order and UTF-8 encoding."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL rows from disk."""

    input_path = Path(path)
    rows = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                message = f"{input_path}:{line_number} is not valid JSON"
                raise DataValidationError(message) from error

            if not isinstance(row, dict):
                raise DataValidationError(f"{input_path}:{line_number} must contain a JSON object")

            rows.append(row)

    return rows


def _require_source_id(row: dict[str, Any]) -> str:
    raw_id = row.get("id")

    if raw_id is None:
        raise DataValidationError("id is required")

    source_id = str(raw_id)

    if not source_id:
        raise DataValidationError("id must not be empty")

    return source_id


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{field_name} must be a non-empty string")

    return value


def _parse_json_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, str):
        raise DataValidationError(f"{field_name} must be a JSON-encoded string")

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise DataValidationError(f"{field_name} is not valid JSON") from error

    if not isinstance(parsed, list):
        raise DataValidationError(f"{field_name} must decode to a list")

    return parsed


def _require_list_of_dicts(value: Any, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise DataValidationError(f"{field_name} must be a list")

    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise DataValidationError(f"{field_name}[{index}] must be an object")

    return value


def _validate_tools(tools: list[Any]) -> None:
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise DataValidationError(f"tools[{index}] must be an object")

        name = _tool_name(tool)
        if not isinstance(name, str) or not name.strip():
            raise DataValidationError(f"tools[{index}] must contain a non-empty name")

        parameters = tool.get("parameters")
        if parameters is None and isinstance(tool.get("function"), dict):
            parameters = tool["function"].get("parameters")

        if parameters is not None and not isinstance(parameters, dict):
            raise DataValidationError(f"tools[{index}] parameters must be an object when present")


def _validate_answers(answers: list[Any]) -> None:
    for index, answer in enumerate(answers):
        if not isinstance(answer, dict):
            raise DataValidationError(f"answers[{index}] must be an object")

        name = answer.get("name")
        arguments = answer.get("arguments")

        if not isinstance(name, str) or not name.strip():
            raise DataValidationError(f"answers[{index}] must contain a non-empty name")

        if not isinstance(arguments, dict):
            raise DataValidationError(f"answers[{index}] arguments must be an object")


def _tool_name(tool: dict[str, Any]) -> Any:
    """Support both flat and OpenAI-style nested function schemas."""

    if "name" in tool:
        return tool["name"]

    function = tool.get("function")
    if isinstance(function, dict):
        return function.get("name")

    return None
