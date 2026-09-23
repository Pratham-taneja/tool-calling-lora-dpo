from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from tool_calling_lora_dpo.data import PreparedExample, ids_sha256
from tool_calling_lora_dpo.preferences import (
    DpoPair,
    PairBuildOutcome,
    assert_dpo_ids_are_training_only,
    build_preference_pair,
    pair_to_json_dict,
)


class TextTokenizer(Protocol):
    eos_token: str | None

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> Any: ...


@dataclass(frozen=True)
class CandidateRecord:
    source_id: str
    prompt: str
    gold_calls: list[dict[str, Any]]
    candidates: list[str]


@dataclass(frozen=True)
class DpoDatasetResult:
    pairs: list[DpoPair]
    skipped_reasons: dict[str, int]
    over_length_candidates: int


def select_dpo_sources(
    examples: list[PreparedExample],
    *,
    eligible_ids: set[str],
    count: int,
    seed: int,
) -> list[PreparedExample]:
    """Select a reproducible training-only subset without relying on file order."""

    eligible = sorted(
        (example for example in examples if example.source_id in eligible_ids),
        key=lambda example: example.source_id,
    )
    if count <= 0:
        raise ValueError("DPO source count must be positive")
    if count > len(eligible):
        raise ValueError(f"requested {count} DPO sources but only {len(eligible)} are eligible")

    rng = random.Random(seed)
    selected = rng.sample(eligible, count)
    return sorted(selected, key=lambda example: example.source_id)


def build_candidate_identity(
    *,
    config_hash: str,
    adapter_sha256: str,
    source_ids: list[str],
    candidates_per_prompt: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "stage": "dpo_candidate_generation",
        "config_hash": config_hash,
        "adapter_sha256": adapter_sha256,
        "source_ids_hash": ids_sha256(source_ids),
        "source_count": len(source_ids),
        "candidates_per_prompt": candidates_per_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
    }


def identity_sha256(identity: dict[str, Any]) -> str:
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def candidate_record_to_dict(record: CandidateRecord) -> dict[str, Any]:
    return asdict(record)


def candidate_record_from_dict(row: dict[str, Any]) -> CandidateRecord:
    return CandidateRecord(
        source_id=str(row["source_id"]),
        prompt=str(row["prompt"]),
        gold_calls=list(row["gold_calls"]),
        candidates=[str(candidate) for candidate in row["candidates"]],
    )


def write_candidate_checkpoint(
    path: str | Path,
    *,
    identity: dict[str, Any],
    records: list[CandidateRecord],
) -> None:
    """Atomically save resumable sampled candidates."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": identity,
        "identity_hash": identity_sha256(identity),
        "records": [candidate_record_to_dict(record) for record in records],
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_path, output_path)


def read_candidate_checkpoint(
    path: str | Path,
    *,
    expected_identity: dict[str, Any],
) -> list[CandidateRecord]:
    input_path = Path(path)
    if not input_path.exists():
        return []

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("identity_hash") != identity_sha256(expected_identity):
        raise ValueError("candidate checkpoint identity does not match this run")
    if payload.get("identity") != expected_identity:
        raise ValueError("candidate checkpoint settings do not match this run")
    return [candidate_record_from_dict(row) for row in payload.get("records", [])]


def validate_candidate_prefix(
    records: list[CandidateRecord],
    source_examples: list[PreparedExample],
) -> None:
    expected_ids = [example.source_id for example in source_examples[: len(records)]]
    actual_ids = [record.source_id for record in records]
    if actual_ids != expected_ids:
        raise ValueError("candidate checkpoint records are not the expected source prefix")


def completion_with_eos(tokenizer: TextTokenizer, completion: str) -> str:
    eos = tokenizer.eos_token or ""
    if eos and not completion.endswith(eos):
        return completion + eos
    return completion


def sequence_token_length(tokenizer: TextTokenizer, prompt: str, completion: str) -> int:
    encoded = tokenizer(
        prompt + completion_with_eos(tokenizer, completion),
        add_special_tokens=False,
    )
    token_ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") else encoded
    return len(token_ids)


def build_dpo_dataset(
    records: list[CandidateRecord],
    *,
    tokenizer: TextTokenizer,
    max_seq_length: int,
    train_ids: set[str],
    validation_ids: set[str],
    test_ids: set[str],
) -> DpoDatasetResult:
    """Score candidates, build pairs, and reject sequences that would be truncated."""

    pairs: list[DpoPair] = []
    skipped: dict[str, int] = {}
    over_length_candidates = 0

    for record in records:
        fitting_candidates = []
        for candidate in record.candidates:
            if sequence_token_length(tokenizer, record.prompt, candidate) <= max_seq_length:
                fitting_candidates.append(candidate)
            else:
                over_length_candidates += 1

        outcome: PairBuildOutcome = build_preference_pair(
            source_id=record.source_id,
            prompt=record.prompt,
            gold_calls=record.gold_calls,
            candidates=fitting_candidates,
        )
        if outcome.pair is None:
            reason = outcome.skipped_reason or "unknown"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        chosen_length = sequence_token_length(
            tokenizer,
            outcome.pair.prompt,
            outcome.pair.chosen,
        )
        if chosen_length > max_seq_length:
            skipped["chosen_over_length"] = skipped.get("chosen_over_length", 0) + 1
            continue
        pairs.append(outcome.pair)

    assert_dpo_ids_are_training_only(
        {pair.source_id for pair in pairs},
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
    )
    return DpoDatasetResult(
        pairs=pairs,
        skipped_reasons=skipped,
        over_length_candidates=over_length_candidates,
    )


def build_dpo_manifest(
    *,
    identity: dict[str, Any],
    records: list[CandidateRecord],
    result: DpoDatasetResult,
    max_seq_length: int,
) -> dict[str, Any]:
    chosen_sources: dict[str, int] = {}
    rejection_reasons: dict[str, int] = {}
    for pair in result.pairs:
        chosen_sources[pair.chosen_source] = chosen_sources.get(pair.chosen_source, 0) + 1
        rejection_reasons[pair.rejection_reason] = (
            rejection_reasons.get(pair.rejection_reason, 0) + 1
        )

    return {
        "identity": identity,
        "identity_hash": identity_sha256(identity),
        "max_seq_length": max_seq_length,
        "counts": {
            "source_examples": len(records),
            "sampled_candidates": sum(len(record.candidates) for record in records),
            "preference_pairs": len(result.pairs),
            "over_length_candidates": result.over_length_candidates,
        },
        "pair_ids_hash": ids_sha256([pair.source_id for pair in result.pairs]),
        "chosen_sources": chosen_sources,
        "rejection_reasons": rejection_reasons,
        "skipped_reasons": result.skipped_reasons,
    }


def write_candidate_records(path: str | Path, records: list[CandidateRecord]) -> None:
    _write_jsonl(path, [candidate_record_to_dict(record) for record in records])


def write_dpo_pairs(path: str | Path, pairs: list[DpoPair]) -> None:
    _write_jsonl(path, [pair_to_json_dict(pair) for pair in pairs])


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
