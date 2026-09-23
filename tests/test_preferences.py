from __future__ import annotations

import json

import pytest

from tool_calling_lora_dpo.preferences import (
    assert_dpo_ids_are_training_only,
    build_preference_pair,
    deduplicate_candidates,
    pair_to_json_dict,
    parse_pair_json,
    rejection_group,
)
from tool_calling_lora_dpo.prompting import canonical_json


GOLD = [{"name": "get_weather", "arguments": {"city": "Sydney", "unit": "celsius"}}]
EXACT = json.dumps(GOLD)
WRONG_TOOL = json.dumps([{"name": "search_web", "arguments": {"query": "weather"}}])
BAD_ARGS = json.dumps([{"name": "get_weather", "arguments": {"city": "Sydney"}}])
MALFORMED = "not json"


def test_deduplicate_candidates_preserves_order() -> None:
    assert deduplicate_candidates(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_sampled_exact_candidate_is_chosen() -> None:
    outcome = build_preference_pair(
        source_id="1",
        prompt="prompt",
        gold_calls=GOLD,
        candidates=[BAD_ARGS, EXACT],
    )

    assert outcome.skipped_reason is None
    assert outcome.pair is not None
    assert outcome.pair.chosen == EXACT
    assert outcome.pair.rejected == BAD_ARGS
    assert outcome.pair.chosen_source == "sampled_exact"


def test_gold_fallback_is_used_when_no_candidate_is_exact() -> None:
    outcome = build_preference_pair(
        source_id="1",
        prompt="prompt",
        gold_calls=GOLD,
        candidates=[BAD_ARGS, MALFORMED],
    )

    assert outcome.pair is not None
    assert outcome.pair.chosen == canonical_json(GOLD)
    assert outcome.pair.chosen_source == "gold_fallback"


def test_wrong_tool_rejection_has_priority_over_bad_arguments() -> None:
    outcome = build_preference_pair(
        source_id="1",
        prompt="prompt",
        gold_calls=GOLD,
        candidates=[BAD_ARGS, WRONG_TOOL, MALFORMED],
    )

    assert outcome.pair is not None
    assert outcome.pair.rejected == WRONG_TOOL
    assert outcome.pair.rejection_reason == "wrong_tool"


def test_bad_arguments_are_preferred_over_malformed_json_when_no_wrong_tool() -> None:
    outcome = build_preference_pair(
        source_id="1",
        prompt="prompt",
        gold_calls=GOLD,
        candidates=[MALFORMED, BAD_ARGS],
    )

    assert outcome.pair is not None
    assert outcome.pair.rejected == BAD_ARGS
    assert outcome.pair.rejection_reason == "missing_argument"


def test_all_correct_candidates_are_skipped() -> None:
    outcome = build_preference_pair(
        source_id="1",
        prompt="prompt",
        gold_calls=GOLD,
        candidates=[EXACT, EXACT],
    )

    assert outcome.pair is None
    assert outcome.skipped_reason == "all_candidates_correct"


def test_no_rejected_candidate_is_skipped_for_empty_candidates() -> None:
    outcome = build_preference_pair(
        source_id="1",
        prompt="prompt",
        gold_calls=GOLD,
        candidates=[],
    )

    assert outcome.pair is None
    assert outcome.skipped_reason == "no_rejected_candidate"


@pytest.mark.parametrize(
    ("category", "group"),
    [
        ("wrong_tool", "wrong_tool"),
        ("wrong_call_count", "wrong_tool"),
        ("missing_argument", "bad_arguments"),
        ("extra_argument", "bad_arguments"),
        ("wrong_argument_value", "bad_arguments"),
        ("malformed_json", "malformed_json"),
        ("invalid_arguments", "malformed_json"),
    ],
)
def test_rejection_group_maps_categories(category: str, group: str) -> None:
    assert rejection_group(category) == group


def test_pair_to_json_dict_and_parse_round_trip() -> None:
    outcome = build_preference_pair(
        source_id="1",
        prompt="prompt",
        gold_calls=GOLD,
        candidates=[BAD_ARGS],
    )
    assert outcome.pair is not None

    row = pair_to_json_dict(outcome.pair)
    parsed = parse_pair_json(json.dumps(row))

    assert parsed == outcome.pair


def test_assert_dpo_ids_are_training_only_accepts_training_subset() -> None:
    assert_dpo_ids_are_training_only(
        {"1", "2"},
        train_ids={"1", "2", "3"},
        validation_ids={"4"},
        test_ids={"5"},
    )


def test_assert_dpo_ids_are_training_only_rejects_validation_leakage() -> None:
    with pytest.raises(ValueError, match="subset of training"):
        assert_dpo_ids_are_training_only(
            {"1", "4"},
            train_ids={"1", "2", "3"},
            validation_ids={"4"},
            test_ids={"5"},
        )


def test_assert_dpo_ids_are_training_only_rejects_test_leakage() -> None:
    with pytest.raises(ValueError, match="test IDs"):
        assert_dpo_ids_are_training_only(
            {"1", "5"},
            train_ids={"1", "2", "3", "5"},
            validation_ids={"4"},
            test_ids={"5"},
        )
