from __future__ import annotations

from tool_calling_lora_dpo.data import PreparedExample
from tool_calling_lora_dpo.inference import (
    GenerationConfig,
    configure_tokenizer_for_generation,
    generate_many,
    generate_one,
    generation_record_to_json_dict,
)


class FakeTensor:
    def __init__(self, values):
        self.values = values
        if values and isinstance(values[0], int):
            self.shape = (1, len(values))
        else:
            self.shape = (1, len(values[0]))

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row, column_slice = item
            return self.values[row][column_slice]
        return self.values[item]

    def to(self, device):
        return self


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = None
    eos_token = "<eos>"
    pad_token = None
    padding_side = "right"

    def apply_chat_template(self, conversation, *, tokenize=False, add_generation_prompt=False):
        rendered = "\n".join(f"{message['role']}:{message['content']}" for message in conversation)
        if add_generation_prompt:
            rendered += "\nassistant:"
        return rendered

    def __call__(self, text, *, return_tensors, truncation):
        return FakeBatch({"input_ids": FakeTensor([1, 2, 3])})

    def decode(self, token_ids, *, skip_special_tokens=True):
        return '[{"name":"lookup","arguments":{"id":"1"}}]'


class FakeModel:
    device = "cuda"

    def generate(self, **kwargs):
        return FakeTensor([[1, 2, 3, 10, 11, 12]])


def example(source_id: str = "1") -> PreparedExample:
    return PreparedExample(
        source_id=source_id,
        query="Find item 1",
        tools=[{"name": "lookup", "parameters": {}}],
        gold_answers=[{"name": "lookup", "arguments": {"id": "1"}}],
    )


def test_configure_tokenizer_sets_left_padding_and_pad_token() -> None:
    tokenizer = FakeTokenizer()

    configured = configure_tokenizer_for_generation(tokenizer)

    assert configured.padding_side == "left"
    assert configured.pad_token == "<eos>"


def test_generate_one_decodes_only_new_tokens() -> None:
    record = generate_one(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        example=example(),
        generation_config=GenerationConfig(
            max_new_tokens=8,
            do_sample=False,
            max_seq_length=10,
        ),
    )

    assert record.source_id == "1"
    assert record.input_tokens == 3
    assert record.output_tokens == 3
    assert record.model_output.startswith("[")


def test_generate_one_rejects_sampling_for_smoke_eval() -> None:
    try:
        generate_one(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            example=example(),
            generation_config=GenerationConfig(
                max_new_tokens=8,
                do_sample=True,
                max_seq_length=10,
            ),
        )
    except ValueError as error:
        assert "greedy" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_generate_one_rejects_prompt_over_limit() -> None:
    try:
        generate_one(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            example=example(),
            generation_config=GenerationConfig(
                max_new_tokens=8,
                do_sample=False,
                max_seq_length=2,
            ),
        )
    except ValueError as error:
        assert "above max_seq_length" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_generate_many_preserves_order() -> None:
    records = generate_many(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        examples=[example("1"), example("2")],
        generation_config=GenerationConfig(
            max_new_tokens=8,
            do_sample=False,
            max_seq_length=10,
        ),
    )

    assert [record.source_id for record in records] == ["1", "2"]


def test_generation_record_to_json_dict_is_json_ready() -> None:
    record = generate_one(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        example=example(),
        generation_config=GenerationConfig(
            max_new_tokens=8,
            do_sample=False,
            max_seq_length=10,
        ),
    )

    assert generation_record_to_json_dict(record)["source_id"] == "1"
