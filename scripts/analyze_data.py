from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_calling_lora_dpo.analysis import summarize_processed_splits, write_json


DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "data_profile.json"
DEFAULT_TOKEN_AUDIT = PROJECT_ROOT / "results" / "token_audit.json"


def main() -> int:
    args = parse_args()

    profile = summarize_processed_splits(args.processed_dir)
    if args.token_audit.exists():
        profile["token_audit"] = json.loads(args.token_audit.read_text(encoding="utf-8"))

    write_json(args.output, profile)
    print(f"Wrote data profile to {args.output}")
    print(f"Total examples: {profile['total_examples']}")
    print(f"Split overlaps: {profile['split_overlaps']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile processed dataset splits.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--token-audit", type=Path, default=DEFAULT_TOKEN_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
