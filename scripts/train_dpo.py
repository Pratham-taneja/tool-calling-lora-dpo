from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_calling_lora_dpo.config import config_hash, load_config
from tool_calling_lora_dpo.dpo_training import (
    DpoPairCollator,
    build_dpo_training_plan,
    build_training_result,
    configure_tokenizer_for_dpo,
    dpo_loss,
    limit_pairs,
    optimizer_step_count,
    read_dpo_pairs,
    sequence_log_probability,
    write_training_report,
)
from tool_calling_lora_dpo.inference import load_hf_causal_lm

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "dpo_pairs.jsonl"
DEFAULT_SFT_ADAPTER = PROJECT_ROOT / "artifacts" / "models" / "sft_lora_full"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "models" / "dpo_lora"
DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "training" / "dpo_train_report.json"
DEFAULT_HF_HOME = PROJECT_ROOT / "data" / "hf_cache"


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = load_config(args.config)
    pairs = limit_pairs(read_dpo_pairs(args.input), args.limit)
    plan = build_dpo_training_plan(
        config=config,
        sft_adapter_path=args.sft_adapter_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )
    if args.dry_run:
        print(json.dumps(dry_run_summary(plan, len(pairs)), indent=2))
        return 0

    require_cuda()
    resume_path = resolve_resume(args.resume, Path(plan.output_dir))
    metrics = run_training(config=config, plan=plan, pairs=pairs, resume_path=resume_path)
    result = build_training_result(config=config, plan=plan, pairs=pairs, metrics=metrics)
    write_training_report(args.report, result)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DPO adapter from scored pairs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sft-adapter-path", type=Path, default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DPO training")


def run_training(
    *,
    config: Any,
    plan: Any,
    pairs: list[Any],
    resume_path: Path | None,
) -> dict[str, Any]:
    import torch
    from bitsandbytes.optim import PagedAdamW8bit
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

    set_seed(config.seed)
    tokenizer = configure_tokenizer_for_dpo(AutoTokenizer.from_pretrained(plan.sft_adapter_path))
    base_model = load_hf_causal_lm(plan.model_id, load_in_4bit=config.model.load_in_4bit)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    policy_path = str(resume_path or plan.sft_adapter_path)
    model = PeftModel.from_pretrained(base_model, policy_path, is_trainable=True)
    model.load_adapter(plan.sft_adapter_path, adapter_name="reference", is_trainable=False)
    model.set_adapter("default")
    disable_dropout(model)
    model.train()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = PagedAdamW8bit(trainable, lr=plan.learning_rate)
    total_steps = optimizer_step_count(len(pairs), plan)
    warmup_steps = max(1, round(total_steps * 0.03))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda")
    collator = DpoPairCollator(tokenizer, max_seq_length=plan.max_seq_length)

    order = list(range(len(pairs))) * plan.epochs
    random.Random(config.seed).shuffle(order)
    position = 0
    optimizer_step = 0
    loss_sum = 0.0
    examples_seen = 0
    if resume_path is not None:
        state = load_checkpoint_state(resume_path, optimizer, scheduler, scaler)
        if state["config_hash"] != config_hash(config):
            raise ValueError("DPO checkpoint config hash does not match")
        position = int(state["position"])
        optimizer_step = int(state["optimizer_step"])
        loss_sum = float(state["loss_sum"])
        examples_seen = int(state["examples_seen"])

    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    accumulation = 0
    for position_index in range(position, len(order)):
        if optimizer_step >= total_steps:
            break
        pair = pairs[order[position_index]]
        batch = {key: value.to("cuda") for key, value in collator(pair).items()}

        model.set_adapter("reference")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ref_chosen = forward_logp(model, batch, "chosen")
            ref_rejected = forward_logp(model, batch, "rejected")

        model.set_adapter("default")
        with torch.autocast("cuda", dtype=torch.float16):
            policy_chosen = forward_logp(model, batch, "chosen")
            policy_rejected = forward_logp(model, batch, "rejected")
            loss = dpo_loss(
                policy_chosen,
                policy_rejected,
                ref_chosen,
                ref_rejected,
                beta=plan.beta,
            )
        scaler.scale(loss / plan.gradient_accumulation_steps).backward()
        loss_sum += float(loss.detach().cpu())
        examples_seen += 1
        accumulation += 1

        final_example = position_index + 1 == len(order)
        if accumulation == plan.gradient_accumulation_steps or final_example:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation = 0
            optimizer_step += 1

            if optimizer_step == 1 or optimizer_step % plan.logging_steps == 0:
                print(
                    json.dumps(
                        {
                            "step": optimizer_step,
                            "total_steps": total_steps,
                            "mean_loss": loss_sum / examples_seen,
                            "learning_rate": scheduler.get_last_lr()[0],
                        }
                    )
                )
            if optimizer_step % plan.save_steps == 0 or optimizer_step == total_steps:
                save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    output_dir=Path(plan.output_dir),
                    optimizer_step=optimizer_step,
                    state={
                        "config_hash": config_hash(config),
                        "position": position_index + 1,
                        "optimizer_step": optimizer_step,
                        "total_steps": total_steps,
                        "loss_sum": loss_sum,
                        "examples_seen": examples_seen,
                    },
                )

    output_dir = Path(plan.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.set_adapter("default")
    model.save_pretrained(output_dir, selected_adapters=["default"])
    tokenizer.save_pretrained(output_dir)
    runtime = time.perf_counter() - started
    return {
        "optimizer_steps": optimizer_step,
        "total_steps": total_steps,
        "examples_seen": examples_seen,
        "mean_loss": loss_sum / examples_seen,
        "runtime_seconds": runtime,
    }


def forward_logp(model: Any, batch: dict[str, Any], prefix: str) -> Any:
    output = model(
        input_ids=batch[f"{prefix}_input_ids"],
        attention_mask=batch[f"{prefix}_attention_mask"],
        use_cache=False,
    )
    return sequence_log_probability(output.logits, batch[f"{prefix}_labels"])


def disable_dropout(model: Any) -> None:
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def save_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    output_dir: Path,
    optimizer_step: int,
    state: dict[str, Any],
) -> None:
    import torch

    checkpoint = output_dir / f"checkpoint-{optimizer_step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.set_adapter("default")
    model.save_pretrained(checkpoint, selected_adapters=["default"])
    tokenizer.save_pretrained(checkpoint)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
    torch.save(scaler.state_dict(), checkpoint / "scaler.pt")
    torch.save(
        {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
        checkpoint / "rng_state.pth",
    )
    (checkpoint / "dpo_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checkpoints = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    for old_checkpoint in checkpoints[:-2]:
        remove_old_checkpoint(old_checkpoint)


def remove_old_checkpoint(checkpoint: Path, *, attempts: int = 5) -> None:
    """Remove a superseded checkpoint without letting OneDrive locks stop training."""
    for attempt in range(1, attempts + 1):
        try:
            shutil.rmtree(checkpoint)
            return
        except FileNotFoundError:
            return
        except OSError as error:
            if attempt == attempts:
                print(
                    json.dumps(
                        {
                            "warning": "old_checkpoint_cleanup_failed",
                            "checkpoint": str(checkpoint),
                            "error": str(error),
                        }
                    ),
                    file=sys.stderr,
                )
                return
            time.sleep(0.5 * attempt)


def load_checkpoint_state(
    checkpoint: Path,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
) -> dict[str, Any]:
    import torch

    state = json.loads((checkpoint / "dpo_state.json").read_text(encoding="utf-8"))
    optimizer.load_state_dict(
        torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    )
    scheduler.load_state_dict(
        torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    )
    scaler.load_state_dict(
        torch.load(checkpoint / "scaler.pt", map_location="cpu", weights_only=True)
    )
    rng = torch.load(checkpoint / "rng_state.pth", map_location="cpu", weights_only=True)
    torch.set_rng_state(rng["cpu"])
    torch.cuda.set_rng_state_all(rng["cuda"])
    return state


def resolve_resume(value: str | None, output_dir: Path) -> Path | None:
    if value is None:
        return None
    if value.lower() in {"latest", "last", "auto"}:
        checkpoints = [
            path
            for path in output_dir.glob("checkpoint-*")
            if (path / "dpo_state.json").is_file()
            and (path / "adapter_model.safetensors").is_file()
        ]
        if not checkpoints:
            raise ValueError(f"no valid DPO checkpoints found under {output_dir}")
        return max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))
    checkpoint = Path(value)
    if not checkpoint.is_dir() or not (checkpoint / "dpo_state.json").is_file():
        raise ValueError(f"invalid DPO checkpoint: {checkpoint}")
    return checkpoint


def dry_run_summary(plan: Any, pair_count: int) -> dict[str, Any]:
    return {
        "stage": "dpo",
        "pair_count": pair_count,
        "optimizer_steps": optimizer_step_count(pair_count, plan),
        "plan": plan.__dict__,
    }


if __name__ == "__main__":
    raise SystemExit(main())
