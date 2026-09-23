from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tool_calling_lora_dpo.data import PreparedExample, ids_sha256
from tool_calling_lora_dpo.prompting import (
    ChatTemplateTokenizer,
    render_training_prompt,
    token_length,
)


@dataclass(frozen=True)
class SftRecord:
    """One training row for supervised fine-tuning."""

    source_id: str
    text: str
    token_length: int
    prompt_token_length: int


@dataclass(frozen=True)
class SftExclusion:
    """One training example excluded before SFT."""

    source_id: str
    token_length: int
    max_seq_length: int
    reason: str


@dataclass(frozen=True)
class SftDatasetBuildResult:
    """Kept records plus exclusion metadata for one SFT build."""

    records: list[SftRecord]
    exclusions: list[SftExclusion]


def build_sft_dataset(
    examples: list[PreparedExample],
    tokenizer: ChatTemplateTokenizer,
    *,
    max_seq_length: int,
) -> SftDatasetBuildResult:
    """Render and filter SFT examples without truncating gold answers.

    An example is kept only when the full training prompt, including the gold
    assistant JSON answer, fits within the configured sequence length.
    """

    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")

    records: list[SftRecord] = []
    exclusions: list[SftExclusion] = []

    for example in examples:
        length = token_length(tokenizer, example, include_assistant=True)
        prompt_length = token_length(tokenizer, example, include_assistant=False)

        if length > max_seq_length:
            exclusions.append(
                SftExclusion(
                    source_id=example.source_id,
                    token_length=length,
                    max_seq_length=max_seq_length,
                    reason="training_prompt_exceeds_max_seq_length",
                )
            )
            continue

        records.append(
            SftRecord(
                source_id=example.source_id,
                text=render_training_prompt(tokenizer, example),
                token_length=length,
                prompt_token_length=prompt_length,
            )
        )

    return SftDatasetBuildResult(records=records, exclusions=exclusions)


def sft_record_to_json_dict(record: SftRecord) -> dict[str, Any]:
    """Convert an SFT record into a JSONL row."""

    return asdict(record)


def sft_record_from_json_dict(row: dict[str, Any]) -> SftRecord:
    """Load one SFT JSONL row and validate fields used by the trainer."""

    source_id = _require_non_empty_string(row.get("source_id"), "source_id")
    text = _require_non_empty_string(row.get("text"), "text")
    token_length = _require_positive_int(row.get("token_length"), "token_length")
    prompt_token_length = _require_positive_int(
        row.get("prompt_token_length"),
        "prompt_token_length",
    )

    if prompt_token_length >= token_length:
        raise ValueError("prompt_token_length must be smaller than token_length")

    return SftRecord(
        source_id=source_id,
        text=text,
        token_length=token_length,
        prompt_token_length=prompt_token_length,
    )


def sft_exclusion_to_json_dict(exclusion: SftExclusion) -> dict[str, Any]:
    """Convert an exclusion record into a JSON-friendly row."""

    return asdict(exclusion)


def build_exclusion_manifest(
    *,
    result: SftDatasetBuildResult,
    input_example_count: int,
    max_seq_length: int,
    model_id: str,
    config_hash: str,
    source_split: str = "train",
) -> dict[str, Any]:
    """Create the manifest that documents exactly what SFT excluded."""

    kept_ids = [record.source_id for record in result.records]
    excluded_ids = [exclusion.source_id for exclusion in result.exclusions]

    return {
        "source_split": source_split,
        "model_id": model_id,
        "config_hash": config_hash,
        "max_seq_length": max_seq_length,
        "counts": {
            "input": input_example_count,
            "kept": len(result.records),
            "excluded": len(result.exclusions),
        },
        "hashes": {
            "kept_ids": ids_sha256(kept_ids),
            "excluded_ids": ids_sha256(excluded_ids),
        },
        "exclusions": [sft_exclusion_to_json_dict(exclusion) for exclusion in result.exclusions],
    }


def write_sft_records(path: str | Path, records: list[SftRecord]) -> None:
    """Write SFT records as stable JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(sft_record_to_json_dict(record), sort_keys=True) + "\n")


def write_exclusion_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write the SFT exclusion manifest as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")

    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")

    return value
