from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_calling_lora_dpo.data import example_from_json_dict, read_jsonl
from tool_calling_lora_dpo.evaluation import (
    evaluate_predictions,
    evaluation_sha256,
    write_evaluation,
)


def main() -> int:
    args = parse_args()

    examples = [example_from_json_dict(row) for row in read_jsonl(args.examples)]
    predictions = read_jsonl(args.predictions)
    metadata = {
        "examples_path": str(args.examples),
        "predictions_path": str(args.predictions),
    }

    evaluation = evaluate_predictions(
        examples,
        predictions,
        stage=args.stage,
        metadata=metadata,
    )
    evaluation["metadata"]["evaluation_hash"] = evaluation_sha256(evaluation)
    write_evaluation(args.output, evaluation)

    print(f"Wrote {args.stage} evaluation to {args.output}")
    print(f"End-to-end exact: {evaluation['aggregate']['metrics']['end_to_end_exact']:.4f}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score saved tool-call model predictions.")
    parser.add_argument("--stage", required=True, choices=["baseline", "sft", "dpo"])
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
