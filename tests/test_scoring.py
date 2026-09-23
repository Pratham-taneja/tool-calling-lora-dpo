from __future__ import annotations

import json

import pytest

from tool_calling_lora_dpo.scoring import score_model_output, score_to_dict


GOLD = [
    {
        "name": "get_weather",
        "arguments": {
            "city": "Sydney",
            "unit": "celsius",
        },
    }
]


def dump(value: object) -> str:
    return json.dumps(value)


def test_exact_answer_scores_true() -> None:
    score = score_model_output(dump(GOLD), GOLD)

    assert score.json_valid is True
    assert score.schema_valid is True
    assert score.call_count_correct is True
    assert score.tool_correct is True
    assert score.args_exact is True
    assert score.args_exact_given_tool is True
    assert score.end_to_end_exact is True
    assert score.error_category == "exact"


def test_malformed_json_is_reported_without_throwing() -> None:
    score = score_model_output("```json\n[]\n```", GOLD)

    assert score.json_valid is False
    assert score.error_category == "malformed_json"


def test_wrong_top_level_is_not_schema_valid() -> None:
    score = score_model_output(dump({"name": "get_weather"}), GOLD)

    assert score.json_valid is True
    assert score.schema_valid is False
    assert score.error_category == "wrong_top_level"


@pytest.mark.parametrize(
    ("prediction", "category"),
    [
        ([None], "invalid_call_item"),
        ([1], "invalid_call_item"),
        ([{"arguments": {}}], "missing_name"),
        ([{"name": "", "arguments": {}}], "missing_name"),
        ([{"name": "get_weather"}], "invalid_arguments"),
        ([{"name": "get_weather", "arguments": []}], "invalid_arguments"),
    ],
)
def test_schema_failures_are_typed(prediction: list[object], category: str) -> None:
    score = score_model_output(dump(prediction), GOLD)

    assert score.json_valid is True
    assert score.schema_valid is False
    assert score.error_category == category


def test_wrong_call_count_is_reported_before_argument_errors() -> None:
    prediction = GOLD + GOLD

    score = score_model_output(dump(prediction), GOLD)

    assert score.call_count_correct is False
    assert score.tool_correct is False
    assert score.error_category == "wrong_call_count"


def test_wrong_tool_is_reported() -> None:
    prediction = [{"name": "search_web", "arguments": {"query": "Sydney weather"}}]

    score = score_model_output(dump(prediction), GOLD)

    assert score.schema_valid is True
    assert score.tool_correct is False
    assert score.args_exact_given_tool is None
    assert score.error_category == "wrong_tool"


def test_missing_argument_is_reported_when_tool_is_correct() -> None:
    prediction = [{"name": "get_weather", "arguments": {"city": "Sydney"}}]

    score = score_model_output(dump(prediction), GOLD)

    assert score.tool_correct is True
    assert score.args_exact is False
    assert score.args_exact_given_tool is False
    assert score.error_category == "missing_argument"


def test_extra_argument_is_reported_when_tool_is_correct() -> None:
    prediction = [
        {
            "name": "get_weather",
            "arguments": {
                "city": "Sydney",
                "unit": "celsius",
                "date": "today",
            },
        }
    ]

    score = score_model_output(dump(prediction), GOLD)

    assert score.error_category == "extra_argument"


def test_wrong_argument_value_is_reported_when_keys_match() -> None:
    prediction = [{"name": "get_weather", "arguments": {"city": "Melbourne", "unit": "celsius"}}]

    score = score_model_output(dump(prediction), GOLD)

    assert score.error_category == "wrong_argument_value"


def test_tool_multiset_is_order_insensitive_for_multiple_calls() -> None:
    gold = [
        {"name": "first", "arguments": {"x": 1}},
        {"name": "second", "arguments": {"y": 2}},
    ]
    prediction = [
        {"name": "second", "arguments": {"y": 2}},
        {"name": "first", "arguments": {"x": 1}},
    ]

    score = score_model_output(dump(prediction), gold)

    assert score.tool_correct is True
    assert score.args_exact is True
    assert score.end_to_end_exact is True


def test_duplicate_tool_names_compare_as_multiset() -> None:
    gold = [
        {"name": "lookup", "arguments": {"id": 1}},
        {"name": "lookup", "arguments": {"id": 2}},
    ]
    prediction = [
        {"name": "lookup", "arguments": {"id": 1}},
        {"name": "lookup", "arguments": {"id": 3}},
    ]

    score = score_model_output(dump(prediction), gold)

    assert score.tool_correct is True
    assert score.error_category == "wrong_argument_value"


def test_score_to_dict_is_json_serializable() -> None:
    score = score_model_output(dump(GOLD), GOLD)
    as_dict = score_to_dict(score)

    assert json.loads(json.dumps(as_dict))["error_category"] == "exact"


def test_invalid_gold_calls_raise_project_error() -> None:
    with pytest.raises(ValueError, match="invalid gold calls"):
        score_model_output(dump(GOLD), [{"name": "bad", "arguments": []}])
