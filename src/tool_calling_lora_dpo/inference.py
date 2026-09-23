from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from tool_calling_lora_dpo.data import PreparedExample
from tool_calling_lora_dpo.prompting import render_generation_prompt


@dataclass(frozen=True)
class GenerationConfig:
    """Greedy generation settings for frozen evaluation and smoke runs."""

    max_new_tokens: int
    do_sample: bool
    max_seq_length: int


@dataclass(frozen=True)
class GenerationRecord:
    """One generated model output plus lightweight runtime metadata."""

    source_id: str
    model_output: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


class TextGenerationModel(Protocol):
    """Small protocol shared by real HF models and CPU test fakes."""

    def generate(self, **kwargs: Any) -> Any:
        """Generate token IDs from tokenized prompt inputs."""


class GenerationTokenizer(Protocol):
    """Tokenizer interface needed for prompt tokenization and decoding."""

    eos_token_id: int | None
    pad_token_id: int | None
    eos_token: str | None
    pad_token: str | None
    padding_side: str

    def __call__(self, text: str, *, return_tensors: str, truncation: bool) -> Any:
        """Tokenize rendered prompt text."""

    def decode(self, token_ids: Any, *, skip_special_tokens: bool = True) -> str:
        """Decode only newly generated token IDs."""


def configure_tokenizer_for_generation(tokenizer: GenerationTokenizer) -> GenerationTokenizer:
    """Apply generation-safe tokenizer defaults.

    Decoder-only models should use left padding for batched generation. Qwen also
    may not define a pad token, so we reuse EOS for padding when needed.
    """

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def generate_one(
    *,
    model: TextGenerationModel,
    tokenizer: GenerationTokenizer,
    example: PreparedExample,
    generation_config: GenerationConfig,
) -> GenerationRecord:
    """Generate one strict-output candidate and decode only new tokens."""

    if generation_config.do_sample:
        raise ValueError("smoke and frozen evaluation generation must be greedy")

    prompt = render_generation_prompt(tokenizer, example)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = encoded["input_ids"]
    input_length = _sequence_length(input_ids)

    if input_length > generation_config.max_seq_length:
        raise ValueError(
            f"source_id={example.source_id} prompt has {input_length} tokens, "
            f"above max_seq_length={generation_config.max_seq_length}"
        )

    encoded = _move_batch_to_model_device(encoded, model)

    start = time.perf_counter()
    generated = model.generate(
        **encoded,
        max_new_tokens=generation_config.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    latency = time.perf_counter() - start

    new_tokens = _slice_new_tokens(generated, input_length)
    model_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return GenerationRecord(
        source_id=example.source_id,
        model_output=model_output,
        input_tokens=input_length,
        output_tokens=_sequence_length(new_tokens),
        latency_seconds=latency,
    )


def generate_many(
    *,
    model: TextGenerationModel,
    tokenizer: GenerationTokenizer,
    examples: list[PreparedExample],
    generation_config: GenerationConfig,
) -> list[GenerationRecord]:
    """Generate records sequentially to keep VRAM pressure low on 6 GB GPUs."""

    return [
        generate_one(
            model=model,
            tokenizer=tokenizer,
            example=example,
            generation_config=generation_config,
        )
        for example in examples
    ]


def generation_record_to_json_dict(record: GenerationRecord) -> dict[str, Any]:
    """Convert a generation record to JSONL-friendly data."""

    return asdict(record)


def load_hf_tokenizer(model_id: str) -> GenerationTokenizer:
    """Load a Hugging Face tokenizer lazily so tests do not import transformers."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return configure_tokenizer_for_generation(tokenizer)


def load_hf_causal_lm(
    model_id: str,
    *,
    load_in_4bit: bool,
    adapter_path: str | None = None,
) -> TextGenerationModel:
    """Load the causal LM for local CUDA inference.

    The 4-bit path uses bitsandbytes NF4 quantization with float16 compute,
    which is the memory-safe starting point for the RTX 3050 6 GB plan.
    When an adapter path is provided, the base model is loaded first and the
    LoRA adapter is attached with PEFT.
    """

    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = {
        "quantization_config": quantization_config,
        "device_map": {"": 0} if torch.cuda.is_available() else None,
        "low_cpu_mem_usage": True,
    }
    dtype = torch.float16 if torch.cuda.is_available() else None

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **model_kwargs)
    except TypeError:
        # Older Transformers versions still use torch_dtype.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                **model_kwargs,
            )

    if adapter_path is None:
        return model

    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path)


def cuda_memory_summary() -> dict[str, Any]:
    """Return compact CUDA memory numbers for smoke logs."""

    import torch

    if not torch.cuda.is_available():
        return {"cuda_available": False}

    device = torch.cuda.current_device()
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(device),
        "allocated_mb": round(torch.cuda.memory_allocated(device) / 1024 / 1024, 2),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / 1024 / 1024, 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 2),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024 / 1024, 2),
    }


def _sequence_length(token_ids: Any) -> int:
    shape = getattr(token_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])

    if isinstance(token_ids, list):
        return len(token_ids)

    return len(token_ids)


def _slice_new_tokens(generated: Any, input_length: int) -> Any:
    shape = getattr(generated, "shape", None)
    if shape is not None and len(shape) == 2:
        return generated[0, input_length:]

    if isinstance(generated, list) and generated and isinstance(generated[0], list):
        return generated[0][input_length:]

    return generated[input_length:]


def _move_batch_to_model_device(encoded: Any, model: TextGenerationModel) -> Any:
    device = getattr(model, "device", None)
    if device is None:
        return encoded

    if hasattr(encoded, "to"):
        return encoded.to(device)

    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
