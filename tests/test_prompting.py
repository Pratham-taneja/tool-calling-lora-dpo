from __future__ import annotations

import json
from typing import Any

import pytest

from tool_calling_lora_dpo.data import PreparedExample
from tool_calling_lora_dpo.prompting import (
    build_generation_messages,
    build_gold_answer,
    build_system_prompt,
    build_training_messages,
    canonical_json,
    render_generation_prompt,
    render_training_prompt,
    token_length,
    tokenize_training_prompt,
)


class FakeTokenizer:
    """Tiny stand-in for a Hugging Face tokenizer chat template."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        rendered = "".join(
            f"<|{message['role']}|>\n{message['content']}\n" for message in conversation
        )
        if add_generation_prompt:
            rendered += "<|assistant|>\n"

        if tokenize:
            return [ord(character) for character in rendered]
        return rendered


class BadTextTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        if tokenize:
            return "wrong"
        return super().apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


class BadTokenTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        if not tokenize:
            return [1, 2, 3]
        return super().apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


def example() -> PreparedExample:
    return PreparedExample(
        source_id="1",
        query="Book a flight from Sydney to Melbourne.",
        tools=[
            {
                "name": "search_flights",
                "description": "Search for flights.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination": {"type": "string"},
                        "origin": {"type": "string"},
                    },
                },
            }
        ],
        gold_answers=[
            {
                "name": "search_flights",
                "arguments": {
                    "origin": "Sydney",
                    "destination": "Melbourne",
                },
            }
        ],
    )


def test_canonical_json_is_stable_and_compact() -> None:
    first: dict[str, Any] = {"b": 2, "a": 1}
    second: dict[str, Any] = {"a": 1, "b": 2}

    assert canonical_json(first) == canonical_json(second)
    assert canonical_json(first) == '{"a":1,"b":2}'


def test_system_prompt_contains_stable_tools_json() -> None:
    prompt = build_system_prompt(example().tools)

    tools_json = prompt.split("Available tools:\n", maxsplit=1)[1].strip()
    decoded = json.loads(tools_json)

    assert decoded[0]["name"] == "search_flights"
    assert "Return only valid JSON" in prompt


def test_gold_answer_is_strict_json() -> None:
    answer = build_gold_answer(example().gold_answers)

    assert json.loads(answer) == example().gold_answers
    assert "\n" not in answer


def test_training_messages_include_assistant_answer() -> None:
    messages = build_training_messages(example())

    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert json.loads(messages[-1]["content"])[0]["arguments"]["origin"] == "Sydney"


def test_generation_messages_omit_assistant_answer() -> None:
    messages = build_generation_messages(example())

    assert [message["role"] for message in messages] == ["system", "user"]


def test_render_training_prompt_does_not_add_generation_marker() -> None:
    rendered = render_training_prompt(FakeTokenizer(), example())

    assert rendered.count("<|assistant|>") == 1
    assert rendered.rstrip().endswith('}]')


def test_render_generation_prompt_adds_generation_marker() -> None:
    rendered = render_generation_prompt(FakeTokenizer(), example())

    assert rendered.count("<|assistant|>") == 1
    assert rendered.endswith("<|assistant|>\n")
    assert '"destination":"Melbourne"' not in rendered


def test_token_length_and_tokenize_use_chat_template_tokens() -> None:
    tokenizer = FakeTokenizer()

    tokens = tokenize_training_prompt(tokenizer, example())
    length = token_length(tokenizer, example(), include_assistant=True)

    assert len(tokens) == length
    assert length > 0


def test_rendering_rejects_token_ids_when_text_was_requested() -> None:
    with pytest.raises(TypeError, match="token IDs"):
        render_generation_prompt(BadTokenTokenizer(), example())


def test_token_length_rejects_text_when_tokens_were_requested() -> None:
    with pytest.raises(TypeError, match="text"):
        token_length(BadTextTokenizer(), example(), include_assistant=False)
