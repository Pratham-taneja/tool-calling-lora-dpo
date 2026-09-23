from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from tool_calling_lora_dpo.prompting import canonical_json
from tool_calling_lora_dpo.scoring import (
    ErrorCategory,
    ScoreResult,
    score_model_output,
    score_to_dict,
)


RejectionGroup = Literal["wrong_tool", "bad_arguments", "malformed_json"]
SkipReason = Literal["all_candidates_correct", "no_rejected_candidate", "chosen_equals_rejected"]


@dataclass(frozen=True)
class CandidateScore:
    """One sampled completion plus its strict score."""

    text: str
    score: ScoreResult


@dataclass(frozen=True)
class DpoPair:
    """A single preference example suitable for DPO training."""

    source_id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_source: str
    rejection_reason: ErrorCategory
    candidate_count: int
    score_summary: dict[str, Any]


@dataclass(frozen=True)
class PairBuildOutcome:
    """Result of attempting to build one DPO pair."""

    pair: DpoPair | None
    skipped_reason: SkipReason | None


def deduplicate_candidates(candidates: list[str]) -> list[str]:
    """Remove byte-identical sampled outputs while preserving first-seen order."""

    seen: set[str] = set()
    unique: list[str] = []

    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)

    return unique


def build_preference_pair(
    *,
    source_id: str,
    prompt: str,
    gold_calls: list[dict[str, Any]],
    candidates: list[str],
) -> PairBuildOutcome:
    """Choose one DPO pair from sampled candidates.

    Policy from the handoff:
    - prefer a sampled exact candidate as chosen
    - otherwise use the gold answer as the chosen fallback
    - reject wrong-tool outputs first, then argument errors, then malformed JSON
    - skip examples where every unique sampled candidate is already correct
    """

    unique_candidates = deduplicate_candidates(candidates)
    scored = [
        CandidateScore(text=candidate, score=score_model_output(candidate, gold_calls))
        for candidate in unique_candidates
    ]

    if scored and all(candidate.score.end_to_end_exact for candidate in scored):
        return PairBuildOutcome(pair=None, skipped_reason="all_candidates_correct")

    chosen_candidate = next(
        (candidate for candidate in scored if candidate.score.end_to_end_exact),
        None,
    )
    if chosen_candidate is None:
        chosen = canonical_json(gold_calls)
        chosen_source = "gold_fallback"
    else:
        chosen = chosen_candidate.text
        chosen_source = "sampled_exact"

    rejected_candidate = select_rejected_candidate(scored)
    if rejected_candidate is None:
        return PairBuildOutcome(pair=None, skipped_reason="no_rejected_candidate")

    if chosen == rejected_candidate.text:
        return PairBuildOutcome(pair=None, skipped_reason="chosen_equals_rejected")

    return PairBuildOutcome(
        pair=DpoPair(
            source_id=source_id,
            prompt=prompt,
            chosen=chosen,
            rejected=rejected_candidate.text,
            chosen_source=chosen_source,
            rejection_reason=rejected_candidate.score.error_category,
            candidate_count=len(unique_candidates),
            score_summary=summarize_candidate_scores(scored),
        ),
        skipped_reason=None,
    )


def select_rejected_candidate(candidates: list[CandidateScore]) -> CandidateScore | None:
    """Pick the highest-priority rejected candidate."""

    rejected = [candidate for candidate in candidates if not candidate.score.end_to_end_exact]
    if not rejected:
        return None

    for group in ("wrong_tool", "bad_arguments", "malformed_json"):
        for candidate in rejected:
            if rejection_group(candidate.score.error_category) == group:
                return candidate

    return rejected[0]


def rejection_group(category: ErrorCategory) -> RejectionGroup:
    """Map detailed scorer categories to DPO rejection priority groups."""

    if category in {"wrong_call_count", "wrong_tool"}:
        return "wrong_tool"

    if category in {"missing_argument", "extra_argument", "wrong_argument_value"}:
        return "bad_arguments"

    return "malformed_json"


def summarize_candidate_scores(candidates: list[CandidateScore]) -> dict[str, Any]:
    """Build compact provenance counts for DPO pair audits."""

    categories: dict[str, int] = {}
    exact_count = 0

    for candidate in candidates:
        category = candidate.score.error_category
        categories[category] = categories.get(category, 0) + 1
        if candidate.score.end_to_end_exact:
            exact_count += 1

    return {
        "candidate_count": len(candidates),
        "exact_count": exact_count,
        "error_categories": categories,
    }


def pair_to_json_dict(pair: DpoPair) -> dict[str, Any]:
    """Convert a DPO pair into a JSONL-friendly dictionary."""

    return {
        "source_id": pair.source_id,
        "prompt": pair.prompt,
        "chosen": pair.chosen,
        "rejected": pair.rejected,
        "chosen_source": pair.chosen_source,
        "rejection_reason": pair.rejection_reason,
        "candidate_count": pair.candidate_count,
        "score_summary": pair.score_summary,
    }


def scored_candidates_to_json(candidates: list[CandidateScore]) -> list[dict[str, Any]]:
    """Serialize scored candidates for optional audit files."""

    return [
        {"text": candidate.text, "score": score_to_dict(candidate.score)}
        for candidate in candidates
    ]


def assert_dpo_ids_are_training_only(
    dpo_ids: set[str],
    *,
    train_ids: set[str],
    validation_ids: set[str],
    test_ids: set[str],
) -> None:
    """Guard against split leakage before saving preference pairs."""

    outside_train = dpo_ids - train_ids
    if outside_train:
        raise ValueError(f"DPO IDs must be a subset of training IDs: {sorted(outside_train)}")

    validation_overlap = dpo_ids & validation_ids
    if validation_overlap:
        raise ValueError(f"DPO IDs overlap validation IDs: {sorted(validation_overlap)}")

    test_overlap = dpo_ids & test_ids
    if test_overlap:
        raise ValueError(f"DPO IDs overlap test IDs: {sorted(test_overlap)}")


def parse_pair_json(line: str) -> DpoPair:
    """Load one saved DPO pair from JSONL."""

    row = json.loads(line)
    return DpoPair(
        source_id=str(row["source_id"]),
        prompt=str(row["prompt"]),
        chosen=str(row["chosen"]),
        rejected=str(row["rejected"]),
        chosen_source=str(row["chosen_source"]),
        rejection_reason=row["rejection_reason"],
        candidate_count=int(row["candidate_count"]),
        score_summary=dict(row["score_summary"]),
    )
