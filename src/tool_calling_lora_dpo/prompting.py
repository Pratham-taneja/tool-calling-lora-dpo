from __future__ import annotations

import json
from typing import Any, Protocol

from tool_calling_lora_dpo.data import PreparedExample

SYSTEM_PROMPT = """You are a tool-calling assistant.

Given the user's request and the available tools, respond only with a JSON array of tool calls.

Rules:
- Return only valid JSON.
- Do not use Markdown.
- Do not explain your answer.
- Each tool call must be an object with "name" and "arguments".
- "name" must match one of the provided tools.
- "arguments" must be a JSON object.
- If multiple calls are needed, preserve the correct call order.

Available tools:
{tools_json}
"""


class ChatTemplateTokenizer(Protocol):
    """Minimal tokenizer interface used by this project.

    Hugging Face tokenizers expose `apply_chat_template`. Tests use a small fake
    tokenizer with the same method so prompt behavior stays CPU-only.
    """

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> Any:
        """Render chat messages using the model's native chat template."""


def canonical_json(value: Any) -> str:
    """Serialize JSON predictably for prompts, gold answers, and tests."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_system_prompt(tools: list[dict[str, Any]]) -> str:
    """Build the system message that carries the available tool schemas."""

    return SYSTEM_PROMPT.format(tools_json=canonical_json(tools))


def build_gold_answer(answers: list[dict[str, Any]]) -> str:
    """Render the target assistant answer as strict JSON."""

    return canonical_json(answers)


def build_messages(example: PreparedExample, *, include_assistant: bool) -> list[dict[str, str]]:
    """Build canonical messages for training or generation.

    Training includes the assistant gold answer. Generation omits it and relies
    on the tokenizer's generation marker.
    """

    messages = [
        {"role": "system", "content": build_system_prompt(example.tools)},
        {"role": "user", "content": example.query},
    ]

    if include_assistant:
        messages.append({"role": "assistant", "content": build_gold_answer(example.gold_answers)})

    return messages


def build_training_messages(example: PreparedExample) -> list[dict[str, str]]:
    """Messages used for supervised fine-tuning."""

    return build_messages(example, include_assistant=True)


def build_generation_messages(example: PreparedExample) -> list[dict[str, str]]:
    """Messages used for baseline/SFT/DPO generation and evaluation."""

    return build_messages(example, include_assistant=False)


def render_training_prompt(tokenizer: ChatTemplateTokenizer, example: PreparedExample) -> str:
    """Render one full training prompt with the assistant answer included."""

    rendered = tokenizer.apply_chat_template(
        build_training_messages(example),
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer returned token IDs when text was requested")
    return rendered


def render_generation_prompt(tokenizer: ChatTemplateTokenizer, example: PreparedExample) -> str:
    """Render one inference prompt with the assistant generation marker."""

    rendered = tokenizer.apply_chat_template(
        build_generation_messages(example),
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer returned token IDs when text was requested")
    return rendered


def tokenize_training_prompt(
    tokenizer: ChatTemplateTokenizer,
    example: PreparedExample,
) -> list[int]:
    """Tokenize a full training prompt using the model's native chat template."""

    tokens = tokenizer.apply_chat_template(
        build_training_messages(example),
        tokenize=True,
        add_generation_prompt=False,
    )
    return _normalize_token_ids(tokens)


def token_length(
    tokenizer: ChatTemplateTokenizer,
    example: PreparedExample,
    *,
    include_assistant: bool,
) -> int:
    """Count chat-template tokens for token-length audits."""

    tokens = tokenizer.apply_chat_template(
        build_messages(example, include_assistant=include_assistant),
        tokenize=True,
        add_generation_prompt=not include_assistant,
    )
    return len(_normalize_token_ids(tokens))


def _normalize_token_ids(tokenizer_output: Any) -> list[int]:
    """Return a flat token ID list from common chat-template return shapes."""

    tokens = tokenizer_output
    if not isinstance(tokens, list) and hasattr(tokens, "get"):
        tokens = tokens.get("input_ids")

    if not isinstance(tokens, list) or not all(isinstance(token_id, int) for token_id in tokens):
        raise TypeError("tokenizer returned text when token IDs were requested")

    return tokens
