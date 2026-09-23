from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tool_calling_lora_dpo.data import PreparedExample, ids_sha256
from tool_calling_lora_dpo.evaluation import evaluate_predictions, evaluation_sha256
from tool_calling_lora_dpo.inference import (
    GenerationConfig,
    GenerationRecord,
    GenerationTokenizer,
    TextGenerationModel,
    cuda_memory_summary,
    generate_one,
    generation_record_to_json_dict,
)
from tool_calling_lora_dpo.reporting import add_metric_confidence_intervals


@dataclass(frozen=True)
class EvaluationIdentity:
    """Identity fields that must match before resuming a run."""

    stage: str
    model_id: str
    adapter_path: str | None
    split_name: str
    config_hash: str
    manifest_hash: str
    example_ids_hash: str
    max_new_tokens: int
    do_sample: bool
    max_seq_length: int


@dataclass(frozen=True)
class EvaluationRunResult:
    """Generated records and runtime metadata for a completed/resumed run."""

    records: list[GenerationRecord]
    metadata: dict[str, Any]


def build_evaluation_identity(
    *,
    stage: str,
    model_id: str,
    split_name: str,
    config_hash: str,
    manifest_hash: str,
    examples: list[PreparedExample],
    generation_config: GenerationConfig,
    adapter_path: str | None = None,
) -> EvaluationIdentity:
    """Build resume identity from immutable run inputs."""

    return EvaluationIdentity(
        stage=stage,
        model_id=model_id,
        adapter_path=adapter_path,
        split_name=split_name,
        config_hash=config_hash,
        manifest_hash=manifest_hash,
        example_ids_hash=ids_sha256([example.source_id for example in examples]),
        max_new_tokens=generation_config.max_new_tokens,
        do_sample=generation_config.do_sample,
        max_seq_length=generation_config.max_seq_length,
    )


def run_resumable_evaluation(
    *,
    model: TextGenerationModel,
    tokenizer: GenerationTokenizer,
    examples: list[PreparedExample],
    generation_config: GenerationConfig,
    identity: EvaluationIdentity,
    checkpoint_path: str | Path,
    checkpoint_every: int = 25,
) -> EvaluationRunResult:
    """Generate model outputs with periodic checkpoint/resume support."""

    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")

    checkpoint = load_checkpoint(checkpoint_path, identity)
    records = [
        _record_from_json(row)
        for row in checkpoint.get("records", [])
    ]
    completed_ids = {record.source_id for record in records}
    start_time = time.perf_counter()
    generated_since_checkpoint = 0

    for example in examples:
        if example.source_id in completed_ids:
            continue

        record = generate_one(
            model=model,
            tokenizer=tokenizer,
            example=example,
            generation_config=generation_config,
        )
        records.append(record)
        completed_ids.add(record.source_id)
        generated_since_checkpoint += 1

        if generated_since_checkpoint >= checkpoint_every:
            save_checkpoint(checkpoint_path, identity=identity, records=records)
            generated_since_checkpoint = 0

    save_checkpoint(checkpoint_path, identity=identity, records=records)
    elapsed = time.perf_counter() - start_time
    generated_now = len(records) - len(checkpoint.get("records", []))

    return EvaluationRunResult(
        records=records,
        metadata={
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_every": checkpoint_every,
            "resumed_records": len(checkpoint.get("records", [])),
            "generated_records_this_run": generated_now,
            "elapsed_seconds_this_run": elapsed,
            "seconds_per_generated_record": elapsed / generated_now if generated_now else 0.0,
            "cuda_memory": cuda_memory_summary(),
        },
    )


def build_final_evaluation_artifact(
    *,
    examples: list[PreparedExample],
    run_result: EvaluationRunResult,
    identity: EvaluationIdentity,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Score generated outputs and attach metadata for portfolio reporting."""

    prediction_rows = [generation_record_to_json_dict(record) for record in run_result.records]
    evaluation = evaluate_predictions(
        examples,
        prediction_rows,
        stage=identity.stage,
        metadata={
            **asdict(identity),
            **run_result.metadata,
        },
    )
    return add_confidence_intervals(evaluation, bootstrap_seed=bootstrap_seed)


def add_confidence_intervals(
    evaluation: dict[str, Any],
    *,
    bootstrap_seed: int,
    iterations: int = 1000,
) -> dict[str, Any]:
    """Attach bootstrap CIs to an existing evaluation artifact."""

    from tool_calling_lora_dpo.scoring import ScoreResult, ToolCall

    score_results = []
    for row in evaluation["per_example"]:
        score = row["score"]
        score_results.append(
            ScoreResult(
                json_valid=score["json_valid"],
                schema_valid=score["schema_valid"],
                call_count_correct=score["call_count_correct"],
                tool_correct=score["tool_correct"],
                args_exact=score["args_exact"],
                args_exact_given_tool=score["args_exact_given_tool"],
                end_to_end_exact=score["end_to_end_exact"],
                error_category=score["error_category"],
                predicted_calls=[
                    ToolCall(name=call["name"], arguments=call["arguments"])
                    for call in score["predicted_calls"]
                ],
            )
        )

    evaluation["aggregate"]["confidence_intervals"] = add_metric_confidence_intervals(
        score_results,
        seed=bootstrap_seed,
        iterations=iterations,
    )
    evaluation["metadata"]["evaluation_hash"] = evaluation_sha256(evaluation)
    return evaluation


def save_checkpoint(
    path: str | Path,
    *,
    identity: EvaluationIdentity,
    records: list[GenerationRecord],
) -> None:
    """Atomically write checkpoint JSON."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": asdict(identity),
        "records": [generation_record_to_json_dict(record) for record in records],
    }
    atomic_write_json(checkpoint_path, payload)


def load_checkpoint(path: str | Path, identity: EvaluationIdentity) -> dict[str, Any]:
    """Load a checkpoint and reject identity mismatches."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return {"identity": asdict(identity), "records": []}

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("identity") != asdict(identity):
        raise ValueError("checkpoint identity does not match current evaluation run")

    records = checkpoint.get("records")
    if not isinstance(records, list):
        raise ValueError("checkpoint records must be a list")

    return checkpoint


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via temp file plus replace to avoid partial checkpoints."""

    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp_path, path)


def _record_from_json(row: dict[str, Any]) -> GenerationRecord:
    return GenerationRecord(
        source_id=str(row["source_id"]),
        model_output=str(row["model_output"]),
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        latency_seconds=float(row["latency_seconds"]),
    )
