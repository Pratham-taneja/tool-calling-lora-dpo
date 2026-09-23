from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from transformers import AutoTokenizer

from tool_calling_lora_dpo.config import config_hash, load_config
from tool_calling_lora_dpo.data import example_from_json_dict, read_jsonl
from tool_calling_lora_dpo.sft_dataset import (
    build_exclusion_manifest,
    build_sft_dataset,
    write_exclusion_manifest,
    write_sft_records,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "sft_train.jsonl"
DEFAULT_EXCLUSIONS = PROJECT_ROOT / "results" / "sft_exclusions.json"
DEFAULT_HF_HOME = PROJECT_ROOT / "data" / "hf_cache"


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    config = load_config(args.config)
    examples = [example_from_json_dict(row) for row in read_jsonl(args.input)]
    tokenizer = AutoTokenizer.from_pretrained(config.model.id)

    result = build_sft_dataset(
        examples,
        tokenizer,
        max_seq_length=config.prompt.max_seq_length,
    )
    manifest = build_exclusion_manifest(
        result=result,
        input_example_count=len(examples),
        max_seq_length=config.prompt.max_seq_length,
        model_id=config.model.id,
        config_hash=config_hash(config),
    )

    write_sft_records(args.output, result.records)
    write_exclusion_manifest(args.exclusions, manifest)

    print(f"Wrote SFT records to {args.output}")
    print(f"Wrote exclusion manifest to {args.exclusions}")
    print(f"Kept: {len(result.records)}")
    print(f"Excluded: {len(result.exclusions)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build token-length-filtered SFT data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
