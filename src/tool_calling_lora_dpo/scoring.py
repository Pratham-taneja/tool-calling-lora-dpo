from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal


ErrorCategory = Literal[
    "malformed_json",
    "wrong_top_level",
    "invalid_call_item",
    "missing_name",
    "invalid_arguments",
    "wrong_call_count",
    "wrong_tool",
    "missing_argument",
    "extra_argument",
    "wrong_argument_value",
    "exact",
]


@dataclass(frozen=True)
class ToolCall:
    """Validated model or gold tool call."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ScoreResult:
    """All per-example scoring fields required by the experiment spec."""

    json_valid: bool
    schema_valid: bool
    call_count_correct: bool
    tool_correct: bool
    args_exact: bool
    args_exact_given_tool: bool | None
    end_to_end_exact: bool
    error_category: ErrorCategory
    predicted_calls: list[ToolCall]


def score_model_output(model_output: str, gold_calls: list[dict[str, Any]]) -> ScoreResult:
    """Score one model string against validated gold calls without throwing.

    Strict scoring intentionally does not strip Markdown fences, repair broken
    JSON, or coerce odd shapes. Malformed model output should be visible in the
    result instead of hidden by recovery logic.
    """

    gold = _coerce_gold_calls(gold_calls)

    try:
        parsed = json.loads(model_output)
    except Exception:
        return _failure("malformed_json")

    if not isinstance(parsed, list):
        return _failure("wrong_top_level", json_valid=True)

    predicted, schema_error = _validate_predicted_calls(parsed)
    if schema_error is not None:
        return _failure(schema_error, json_valid=True, predicted_calls=predicted)

    call_count_correct = len(predicted) == len(gold)
    tool_correct = _tool_multiset(predicted) == _tool_multiset(gold)
    args_exact_given_tool = _arguments_exact(predicted, gold) if tool_correct else None
    args_exact = bool(tool_correct and args_exact_given_tool)
    end_to_end_exact = bool(call_count_correct and tool_correct and args_exact)
    error_category = _comparison_error(predicted, gold)

    return ScoreResult(
        json_valid=True,
        schema_valid=True,
        call_count_correct=call_count_correct,
        tool_correct=tool_correct,
        args_exact=args_exact,
        args_exact_given_tool=args_exact_given_tool,
        end_to_end_exact=end_to_end_exact,
        error_category=error_category,
        predicted_calls=predicted,
    )


def score_to_dict(score: ScoreResult) -> dict[str, Any]:
    """Convert a score result to a JSON-serializable dictionary."""

    return {
        "json_valid": score.json_valid,
        "schema_valid": score.schema_valid,
        "call_count_correct": score.call_count_correct,
        "tool_correct": score.tool_correct,
        "args_exact": score.args_exact,
        "args_exact_given_tool": score.args_exact_given_tool,
        "end_to_end_exact": score.end_to_end_exact,
        "error_category": score.error_category,
        "predicted_calls": [
            {"name": call.name, "arguments": call.arguments} for call in score.predicted_calls
        ],
    }


def _failure(
    category: ErrorCategory,
    *,
    json_valid: bool = False,
    predicted_calls: list[ToolCall] | None = None,
) -> ScoreResult:
    return ScoreResult(
        json_valid=json_valid,
        schema_valid=False,
        call_count_correct=False,
        tool_correct=False,
        args_exact=False,
        args_exact_given_tool=None,
        end_to_end_exact=False,
        error_category=category,
        predicted_calls=predicted_calls or [],
    )


def _coerce_gold_calls(gold_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Validate trusted gold data before comparison.

    Data prep should already enforce this, so a gold failure means a project bug
    rather than a model-output scoring failure.
    """

    calls, schema_error = _validate_predicted_calls(gold_calls)
    if schema_error is not None:
        raise ValueError(f"invalid gold calls: {schema_error}")
    return calls


def _validate_predicted_calls(raw_calls: list[Any]) -> tuple[list[ToolCall], ErrorCategory | None]:
    calls: list[ToolCall] = []

    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            return calls, "invalid_call_item"

        name = raw_call.get("name")
        if not isinstance(name, str) or not name.strip():
            return calls, "missing_name"

        arguments = raw_call.get("arguments")
        if not isinstance(arguments, dict):
            return calls, "invalid_arguments"

        calls.append(ToolCall(name=name, arguments=arguments))

    return calls, None


def _tool_multiset(calls: list[ToolCall]) -> Counter[str]:
    return Counter(call.name for call in calls)


def _arguments_exact(predicted: list[ToolCall], gold: list[ToolCall]) -> bool:
    pairs = _pair_calls_by_gold_order(predicted, gold)
    return all(
        predicted_call.arguments == gold_call.arguments
        for predicted_call, gold_call in pairs
    )


def _comparison_error(predicted: list[ToolCall], gold: list[ToolCall]) -> ErrorCategory:
    if len(predicted) != len(gold):
        return "wrong_call_count"

    if _tool_multiset(predicted) != _tool_multiset(gold):
        return "wrong_tool"

    for predicted_call, gold_call in _pair_calls_by_gold_order(predicted, gold):
        predicted_keys = set(predicted_call.arguments)
        gold_keys = set(gold_call.arguments)

        if gold_keys - predicted_keys:
            return "missing_argument"

        if predicted_keys - gold_keys:
            return "extra_argument"

        if predicted_call.arguments != gold_call.arguments:
            return "wrong_argument_value"

    return "exact"


def _pair_calls_by_gold_order(
    predicted: list[ToolCall],
    gold: list[ToolCall],
) -> list[tuple[ToolCall, ToolCall]]:
    """Pair calls with the same tool name while ignoring call order."""

    remaining = predicted.copy()
    pairs: list[tuple[ToolCall, ToolCall]] = []

    for gold_call in gold:
        for index, predicted_call in enumerate(remaining):
            if predicted_call.name == gold_call.name:
                pairs.append((predicted_call, gold_call))
                del remaining[index]
                break

    return pairs
