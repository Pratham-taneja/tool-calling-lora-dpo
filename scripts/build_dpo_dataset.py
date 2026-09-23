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
from tool_calling_lora_dpo.data import example_from_json_dict, read_jsonl
from tool_calling_lora_dpo.dpo_data import (
    CandidateRecord,
    build_candidate_identity,
    build_dpo_dataset,
    build_dpo_manifest,
    file_sha256,
    read_candidate_checkpoint,
    select_dpo_sources,
    validate_candidate_prefix,
    write_candidate_checkpoint,
    write_candidate_records,
    write_dpo_pairs,
    write_json,
)
from tool_calling_lora_dpo.inference import load_hf_causal_lm, load_hf_tokenizer
from tool_calling_lora_dpo.prompting import render_generation_prompt

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"
DEFAULT_EXAMPLES = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "split_manifest.json"
DEFAULT_EXCLUSIONS = PROJECT_ROOT / "results" / "sft_exclusions.json"
DEFAULT_ADAPTER = PROJECT_ROOT / "artifacts" / "models" / "sft_lora_full"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "dpo" / "candidates.json"
DEFAULT_CANDIDATES = PROJECT_ROOT / "artifacts" / "preferences" / "dpo_candidates.jsonl"
DEFAULT_PAIRS = PROJECT_ROOT / "data" / "processed" / "dpo_pairs.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "dpo_dataset_report.json"
DEFAULT_HF_HOME = PROJECT_ROOT / "data" / "hf_cache"


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = load_config(args.config)
    examples = [example_from_json_dict(row) for row in read_jsonl(args.examples)]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    exclusions = json.loads(args.exclusions.read_text(encoding="utf-8"))
    excluded_ids = {str(row["source_id"]) for row in exclusions["exclusions"]}
    train_ids = {str(source_id) for source_id in manifest["ids"]["train"]}
    validation_ids = {str(source_id) for source_id in manifest["ids"]["validation"]}
    test_ids = {str(source_id) for source_id in manifest["ids"]["test"]}
    eligible_ids = train_ids - excluded_ids

    source_count = args.limit or config.dpo.source_examples
    selected = select_dpo_sources(
        examples,
        eligible_ids=eligible_ids,
        count=source_count,
        seed=config.seed,
    )
    adapter_weights = args.adapter_path / "adapter_model.safetensors"
    identity = build_candidate_identity(
        config_hash=config_hash(config),
        adapter_sha256=file_sha256(adapter_weights),
        source_ids=[example.source_id for example in selected],
        candidates_per_prompt=config.dpo.candidates_per_prompt,
        temperature=config.dpo.sample_temperature,
        top_p=config.dpo.sample_top_p,
        max_new_tokens=args.max_new_tokens or config.evaluation.max_new_tokens,
        seed=config.seed,
    )
    records = read_candidate_checkpoint(args.checkpoint, expected_identity=identity)
    validate_candidate_prefix(records, selected)

    tokenizer = load_hf_tokenizer(str(args.adapter_path))
    if len(records) < len(selected):
        require_cuda()
        model = load_hf_causal_lm(
            config.model.id,
            load_in_4bit=config.model.load_in_4bit,
            adapter_path=str(args.adapter_path),
        )
        model.eval()
        for index, example in enumerate(selected[len(records) :], start=len(records)):
            prompt = render_generation_prompt(tokenizer, example)
            candidates = sample_candidates(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                candidate_count=config.dpo.candidates_per_prompt,
                temperature=config.dpo.sample_temperature,
                top_p=config.dpo.sample_top_p,
                max_new_tokens=args.max_new_tokens or config.evaluation.max_new_tokens,
                seed=config.seed + index,
            )
            records.append(
                CandidateRecord(
                    source_id=example.source_id,
                    prompt=prompt,
                    gold_calls=example.gold_answers,
                    candidates=candidates,
                )
            )
            if len(records) % args.checkpoint_every == 0:
                write_candidate_checkpoint(args.checkpoint, identity=identity, records=records)
                print(f"Saved candidate checkpoint: {len(records)}/{len(selected)}")
        write_candidate_checkpoint(args.checkpoint, identity=identity, records=records)

    result = build_dpo_dataset(
        records,
        tokenizer=tokenizer,
        max_seq_length=config.prompt.max_seq_length,
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
    )
    report = build_dpo_manifest(
        identity=identity,
        records=records,
        result=result,
        max_seq_length=config.prompt.max_seq_length,
    )
    write_candidate_records(args.candidates_output, records)
    write_dpo_pairs(args.pairs_output, result.pairs)
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample SFT candidates and build DPO pairs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--candidates-output", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--pairs-output", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DPO candidate generation")


def sample_candidates(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate_count: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> list[str]:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_length = int(encoded["input_ids"].shape[-1])
    if hasattr(encoded, "to"):
        encoded = encoded.to(model.device)
    else:
        encoded = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=candidate_count,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return [
        tokenizer.decode(row[input_length:], skip_special_tokens=True).strip()
        for row in generated
    ]


if __name__ == "__main__":
    raise SystemExit(main())
