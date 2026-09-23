from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from tool_calling_lora_dpo.config import ExperimentConfig, config_hash
from tool_calling_lora_dpo.data import ids_sha256
from tool_calling_lora_dpo.dpo_data import completion_with_eos
from tool_calling_lora_dpo.inference import cuda_memory_summary
from tool_calling_lora_dpo.preferences import DpoPair, parse_pair_json


class TrainingTokenizer(Protocol):
    eos_token: str | None
    eos_token_id: int | None
    pad_token: str | None
    pad_token_id: int | None
    padding_side: str

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> Any: ...


@dataclass(frozen=True)
class DpoTrainingPlan:
    model_id: str
    sft_adapter_path: str
    output_dir: str
    max_seq_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    beta: float
    max_steps: int
    logging_steps: int
    save_steps: int


@dataclass(frozen=True)
class DpoTrainingResult:
    stage: str
    model_id: str
    sft_adapter_path: str
    output_dir: str
    pair_count: int
    pair_ids_hash: str
    config_hash: str
    metrics: dict[str, Any]
    cuda_memory: dict[str, Any]


class DpoPairCollator:
    """Tokenize preference pairs and mask prompt/padding labels."""

    def __init__(self, tokenizer: TrainingTokenizer, *, max_seq_length: int) -> None:
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, pair: DpoPair) -> dict[str, Any]:
        import torch

        rows = {}
        for key, completion in (("chosen", pair.chosen), ("rejected", pair.rejected)):
            prompt_ids = _token_ids(self.tokenizer, pair.prompt)
            full_ids = _token_ids(
                self.tokenizer,
                pair.prompt + completion_with_eos(self.tokenizer, completion),
            )
            prompt_length = _adjusted_prompt_length(prompt_ids, full_ids)
            if len(full_ids) > self.max_seq_length:
                raise ValueError(
                    f"source_id={pair.source_id} {key} sequence has {len(full_ids)} tokens, "
                    f"above max_seq_length={self.max_seq_length}"
                )
            labels = full_ids.copy()
            labels[:prompt_length] = [-100] * prompt_length
            rows[f"{key}_input_ids"] = torch.tensor([full_ids], dtype=torch.long)
            rows[f"{key}_attention_mask"] = torch.ones((1, len(full_ids)), dtype=torch.long)
            rows[f"{key}_labels"] = torch.tensor([labels], dtype=torch.long)
        return rows


def configure_tokenizer_for_dpo(tokenizer: TrainingTokenizer) -> TrainingTokenizer:
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def read_dpo_pairs(path: str | Path) -> list[DpoPair]:
    pairs = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                pairs.append(parse_pair_json(line))
            except Exception as error:
                raise ValueError(f"invalid DPO pair at line {line_number}") from error
    if not pairs:
        raise ValueError("DPO pair dataset is empty")
    return pairs


def limit_pairs(pairs: list[DpoPair], limit: int | None) -> list[DpoPair]:
    if limit is None:
        return pairs
    if limit <= 0:
        raise ValueError("limit must be positive when provided")
    return pairs[:limit]


def build_dpo_training_plan(
    *,
    config: ExperimentConfig,
    sft_adapter_path: str | Path,
    output_dir: str | Path,
    max_steps: int,
    logging_steps: int,
    save_steps: int,
) -> DpoTrainingPlan:
    if max_steps < -1 or max_steps == 0:
        raise ValueError("max_steps must be -1 or positive")
    if logging_steps <= 0 or save_steps <= 0:
        raise ValueError("logging_steps and save_steps must be positive")
    return DpoTrainingPlan(
        model_id=config.model.id,
        sft_adapter_path=str(sft_adapter_path),
        output_dir=str(output_dir),
        max_seq_length=config.prompt.max_seq_length,
        micro_batch_size=config.dpo.micro_batch_size,
        gradient_accumulation_steps=config.dpo.gradient_accumulation_steps,
        epochs=config.dpo.epochs,
        learning_rate=config.dpo.learning_rate,
        beta=config.dpo.beta,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
    )


def optimizer_step_count(pair_count: int, plan: DpoTrainingPlan) -> int:
    full_steps = math.ceil(pair_count / plan.gradient_accumulation_steps) * plan.epochs
    return min(full_steps, plan.max_steps) if plan.max_steps > 0 else full_steps


def sequence_log_probability(logits: Any, labels: Any) -> Any:
    """Sum completion-token log probabilities for each sequence."""

    import torch

    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = torch.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    reference_chosen_logps: Any,
    reference_rejected_logps: Any,
    *,
    beta: float,
) -> Any:
    """Standard sigmoid DPO loss from policy/reference log-probability margins."""

    import torch.nn.functional as functional

    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    return -functional.logsigmoid(beta * (policy_margin - reference_margin)).mean()


def build_training_result(
    *,
    config: ExperimentConfig,
    plan: DpoTrainingPlan,
    pairs: list[DpoPair],
    metrics: dict[str, Any],
) -> DpoTrainingResult:
    return DpoTrainingResult(
        stage="dpo",
        model_id=plan.model_id,
        sft_adapter_path=plan.sft_adapter_path,
        output_dir=plan.output_dir,
        pair_count=len(pairs),
        pair_ids_hash=ids_sha256([pair.source_id for pair in pairs]),
        config_hash=config_hash(config),
        metrics=metrics,
        cuda_memory=cuda_memory_summary(),
    )


def write_training_report(path: str | Path, result: DpoTrainingResult) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")


def _token_ids(tokenizer: TrainingTokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if hasattr(encoded, "__getitem__") else encoded
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _adjusted_prompt_length(prompt_ids: list[int], full_ids: list[int]) -> int:
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids)
    # A tokenizer may merge the final prompt token with the first completion token.
    if prompt_ids and full_ids[: len(prompt_ids) - 1] == prompt_ids[:-1]:
        return len(prompt_ids) - 1
    raise ValueError("prompt tokenization is not a prefix of the full DPO sequence")
