from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_calling_lora_dpo.config import config_hash, load_config
from tool_calling_lora_dpo.data import (
    example_to_json_dict,
    make_deterministic_splits,
    parse_source_rows,
    split_manifest,
    write_jsonl,
)
from tool_calling_lora_dpo.prompting import token_length


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_HF_HOME = PROJECT_ROOT / "data" / "hf_cache"


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    config = load_config(args.config)

    rows = load_dataset_rows(config.data.id, split=args.dataset_split, limit=args.limit)
    examples = parse_source_rows(rows)
    splits = make_deterministic_splits(
        [example.source_id for example in examples],
        validation_size=config.data.validation_size,
        test_size=config.data.test_size,
        seed=config.seed,
    )

    examples_by_id = {example.source_id: example for example in examples}
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)

    write_jsonl(
        output_dir / "train.jsonl",
        [example_to_json_dict(examples_by_id[source_id]) for source_id in splits.train_ids],
    )
    write_jsonl(
        output_dir / "validation.jsonl",
        [example_to_json_dict(examples_by_id[source_id]) for source_id in splits.validation_ids],
    )
    write_jsonl(
        output_dir / "test.jsonl",
        [example_to_json_dict(examples_by_id[source_id]) for source_id in splits.test_ids],
    )

    manifest = split_manifest(splits, seed=config.seed)
    manifest["config_hash"] = config_hash(config)
    write_json(output_dir / "split_manifest.json", manifest)

    if args.audit_tokens:
        audit = build_token_audit(
            examples,
            model_id=config.model.id,
            max_seq_length=config.prompt.max_seq_length,
        )
        write_json(results_dir / "token_audit.json", audit)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic xLAM tool-call splits.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional local smoke limit. Do not use for final reported results.",
    )
    parser.add_argument(
        "--audit-tokens",
        action="store_true",
        help="Load the tokenizer and write results/token_audit.json.",
    )
    return parser.parse_args()


def load_dataset_rows(dataset_id: str, *, split: str, limit: int | None) -> list[dict[str, Any]]:
    """Load gated Hugging Face rows after local authentication."""

    try:
        from datasets import load_dataset
    except Exception as error:
        raise RuntimeError("datasets is required for data preparation") from error

    dataset = load_dataset(dataset_id, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return [dict(row) for row in dataset]


def build_token_audit(
    examples,
    *,
    model_id: str,
    max_seq_length: int,
) -> dict[str, Any]:
    """Measure token lengths without truncating gold answers."""

    try:
        from transformers import AutoTokenizer
    except Exception as error:
        raise RuntimeError("transformers is required for token audits") from error

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    lengths = [
        token_length(tokenizer, example, include_assistant=True)
        for example in examples
    ]

    return {
        "model_id": model_id,
        "example_count": len(lengths),
        "max_seq_length": max_seq_length,
        "percentiles": percentile_summary(lengths),
        "over_limit_counts": {
            "1024": sum(length > 1024 for length in lengths),
            "1536": sum(length > 1536 for length in lengths),
            "2048": sum(length > 2048 for length in lengths),
        },
    }


def percentile_summary(values: list[int]) -> dict[str, int]:
    if not values:
        raise ValueError("cannot audit token lengths for an empty dataset")

    ordered = sorted(values)
    return {
        "p50": percentile(ordered, 50),
        "p90": percentile(ordered, 90),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
        "max": ordered[-1],
    }


def percentile(sorted_values: list[int], percent: int) -> int:
    """Nearest-rank percentile for compact audit reporting."""

    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")

    index = round((percent / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
