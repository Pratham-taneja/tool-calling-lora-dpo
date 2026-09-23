from __future__ import annotations

import json

import pytest
import torch

from tool_calling_lora_dpo.dpo_training import (
    DpoPairCollator,
    dpo_loss,
    sequence_log_probability,
)
from tool_calling_lora_dpo.preferences import DpoPair


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 9
    pad_token = "<eos>"
    pad_token_id = 9
    padding_side = "right"

    def __call__(self, text, *, add_special_tokens=False):
        return {"input_ids": list(range(1, len(text.split()) + 1))}


def pair() -> DpoPair:
    return DpoPair(
        source_id="1",
        prompt="one two ",
        chosen="three",
        rejected="four five",
        chosen_source="gold_fallback",
        rejection_reason="wrong_argument_value",
        candidate_count=4,
        score_summary={},
    )


def test_collator_masks_prompt_and_keeps_completion_labels() -> None:
    batch = DpoPairCollator(FakeTokenizer(), max_seq_length=8)(pair())
    assert batch["chosen_labels"].tolist() == [[-100, -100, 3]]
    assert batch["rejected_labels"].tolist() == [[-100, -100, 3, 4]]


def test_collator_rejects_over_length_sequences() -> None:
    with pytest.raises(ValueError, match="above max_seq_length"):
        DpoPairCollator(FakeTokenizer(), max_seq_length=2)(pair())


def test_sequence_log_probability_only_scores_unmasked_labels() -> None:
    logits = torch.zeros((1, 3, 4))
    labels = torch.tensor([[-100, 2, 1]])
    result = sequence_log_probability(logits, labels)
    assert result.item() == pytest.approx(-2 * math_log(4))


def math_log(value: float) -> float:
    import math

    return math.log(value)


def test_dpo_loss_is_log_two_when_policy_equals_reference() -> None:
    value = dpo_loss(
        torch.tensor([2.0]),
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([1.0]),
        beta=0.1,
    )
    assert value.item() == pytest.approx(math_log(2))


def test_dpo_loss_decreases_when_policy_preference_margin_improves() -> None:
    neutral = dpo_loss(
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        beta=0.1,
    )
    improved = dpo_loss(
        torch.tensor([3.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        beta=0.1,
    )
    assert improved < neutral
