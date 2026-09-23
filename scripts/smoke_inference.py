from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_calling_lora_dpo.config import config_hash, load_config
from tool_calling_lora_dpo.data import example_from_json_dict, read_jsonl, write_jsonl
from tool_calling_lora_dpo.evaluation import (
    evaluate_predictions,
    evaluation_sha256,
    write_evaluation,
)
from tool_calling_lora_dpo.inference import (
    GenerationConfig,
    cuda_memory_summary,
    generate_many,
    generation_record_to_json_dict,
    load_hf_causal_lm,
    load_hf_tokenizer,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"
DEFAULT_EXAMPLES = PROJECT_ROOT / "data" / "processed" / "validation.jsonl"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "artifacts" / "predictions" / "baseline_smoke.jsonl"
DEFAULT_EVALUATION = PROJECT_ROOT / "artifacts" / "evaluations" / "baseline_smoke_eval.json"
DEFAULT_HF_HOME = PROJECT_ROOT / "data" / "hf_cache"


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    config = load_config(args.config)
    examples = [example_from_json_dict(row) for row in read_jsonl(args.examples)[: args.limit]]
    generation_config = GenerationConfig(
        max_new_tokens=config.evaluation.max_new_tokens,
        do_sample=config.evaluation.do_sample,
        max_seq_length=config.prompt.max_seq_length,
    )

    tokenizer = load_hf_tokenizer(args.model_id or config.model.id)
    model = load_hf_causal_lm(
        args.model_id or config.model.id,
        load_in_4bit=config.model.load_in_4bit,
    )

    records = generate_many(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        generation_config=generation_config,
    )
    prediction_rows = [generation_record_to_json_dict(record) for record in records]
    write_jsonl(args.predictions, prediction_rows)

    evaluation = evaluate_predictions(
        examples,
        prediction_rows,
        stage=args.stage,
        metadata={
            "model_id": args.model_id or config.model.id,
            "example_source": str(args.examples),
            "example_count": len(examples),
            "config_hash": config_hash(config),
            "cuda_memory": cuda_memory_summary(),
        },
    )
    evaluation["metadata"]["evaluation_hash"] = evaluation_sha256(evaluation)
    write_evaluation(args.evaluation, evaluation)

    print(f"Wrote predictions to {args.predictions}")
    print(f"Wrote evaluation to {args.evaluation}")
    print(json.dumps(evaluation["aggregate"]["metrics"], indent=2, sort_keys=True))
    print(json.dumps(evaluation["metadata"]["cuda_memory"], indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small 4-bit model inference smoke test.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--stage", default="baseline_smoke")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
