from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_calling_lora_dpo.config import config_hash, load_config
from tool_calling_lora_dpo.data import example_from_json_dict, read_jsonl, write_jsonl
from tool_calling_lora_dpo.eval_runner import (
    build_evaluation_identity,
    build_final_evaluation_artifact,
    run_resumable_evaluation,
)
from tool_calling_lora_dpo.evaluation import write_evaluation
from tool_calling_lora_dpo.inference import (
    GenerationConfig,
    generation_record_to_json_dict,
    load_hf_causal_lm,
    load_hf_tokenizer,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "split_manifest.json"
DEFAULT_EXAMPLES = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "artifacts" / "predictions" / "baseline_eval.jsonl"
DEFAULT_EVALUATION = PROJECT_ROOT / "artifacts" / "evaluations" / "baseline_eval.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "checkpoints" / "evaluation" / "baseline_eval_checkpoint.json"
)
DEFAULT_HF_HOME = PROJECT_ROOT / "data" / "hf_cache"


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    config = load_config(args.config)
    examples = [example_from_json_dict(row) for row in read_jsonl(args.examples)]
    if args.limit is not None:
        examples = examples[: args.limit]

    model_id = args.model_id or config.model.id
    predictions_path = args.predictions or default_predictions_path(args.stage)
    evaluation_path = args.evaluation or default_evaluation_path(args.stage)
    checkpoint_path = args.checkpoint or default_checkpoint_path(args.stage)
    generation_config = GenerationConfig(
        max_new_tokens=config.evaluation.max_new_tokens,
        do_sample=config.evaluation.do_sample,
        max_seq_length=config.prompt.max_seq_length,
    )
    identity = build_evaluation_identity(
        stage=args.stage,
        model_id=model_id,
        split_name=args.split_name,
        config_hash=config_hash(config),
        manifest_hash=file_sha256(args.manifest),
        examples=examples,
        generation_config=generation_config,
        adapter_path=str(args.adapter_path) if args.adapter_path is not None else None,
    )

    tokenizer_id = str(args.tokenizer_id or args.adapter_path or model_id)
    tokenizer = load_hf_tokenizer(tokenizer_id)
    model = load_hf_causal_lm(
        model_id,
        load_in_4bit=config.model.load_in_4bit,
        adapter_path=str(args.adapter_path) if args.adapter_path is not None else None,
    )
    run_result = run_resumable_evaluation(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        generation_config=generation_config,
        identity=identity,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
    )
    evaluation = build_final_evaluation_artifact(
        examples=examples,
        run_result=run_result,
        identity=identity,
        bootstrap_seed=config.seed,
    )

    prediction_rows = [generation_record_to_json_dict(record) for record in run_result.records]
    write_jsonl(predictions_path, prediction_rows)
    write_evaluation(evaluation_path, evaluation)

    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote evaluation to {evaluation_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(json.dumps(summary_for_console(evaluation), indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen/resumable model evaluation.")
    parser.add_argument("--stage", default="baseline", choices=["baseline", "sft", "dpo"])
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--evaluation", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_predictions_path(stage: str) -> Path:
    if stage == "baseline":
        return DEFAULT_PREDICTIONS
    return PROJECT_ROOT / "artifacts" / "predictions" / f"{stage}_eval.jsonl"


def default_evaluation_path(stage: str) -> Path:
    if stage == "baseline":
        return DEFAULT_EVALUATION
    return PROJECT_ROOT / "artifacts" / "evaluations" / f"{stage}_eval.json"


def default_checkpoint_path(stage: str) -> Path:
    if stage == "baseline":
        return DEFAULT_CHECKPOINT
    return (
        PROJECT_ROOT
        / "artifacts"
        / "checkpoints"
        / "evaluation"
        / f"{stage}_eval_checkpoint.json"
    )


def summary_for_console(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": evaluation["stage"],
        "example_count": evaluation["aggregate"]["example_count"],
        "metrics": evaluation["aggregate"]["metrics"],
        "error_counts": evaluation["aggregate"]["error_counts"],
        "cuda_memory": evaluation["metadata"].get("cuda_memory", {}),
    }


if __name__ == "__main__":
    raise SystemExit(main())
