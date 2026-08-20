# Fine-Tuning a Small LLM for Tool-Calling (LoRA + DPO)

> LoRA + DPO fine-tuning of a small open-weight LLM for tool-calling — trained on Salesforce's
> xlam-function-calling-60k, entirely on Kaggle's free GPU tier, with before/after eval on
> tool-selection and argument accuracy.

Fine-tuning an open-weight instruction model to reliably select the right tool and format
correct arguments, using parameter-efficient LoRA supervised fine-tuning followed by DPO
preference tuning — trained entirely on Kaggle's free GPU tier.

## Objective

Given a user query and a list of available tools (name, description, parameter schema), the
model should output the correct tool call(s) as JSON — right tool, right arguments, valid
format — or correctly recognize when no tool applies.

This is fundamentally a JSON-schema-conformance problem with a tool-selection decision layered
on top, so the evaluation is designed to separate the two: did the model pick the right tool,
and separately, did it fill in the arguments correctly.

## Dataset

**[Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)**
— 60,000 examples generated via Salesforce's APIGen pipeline, covering 3,673 executable APIs
across 21 domains. Each example is verified through three stages (format check, actual
execution, semantic verification); human evaluation on a 600-example sample found a >95%
correct rate.

- Format: `query` (user request), `tools` (available function schemas), `answers` (ground-truth
  tool call(s))
- License: CC-BY-4.0. Gated — requires accepting terms on the HF dataset page.
- Note: intended by Salesforce for research use in support of their paper; noted here for
  transparency, not a blocker for this project.

Citation:
```bibtex
@article{liu2024apigen,
  title={APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets},
  author={Liu, Zuxin and Hoang, Thai and Zhang, Jianguo and Zhu, Ming and Lan, Tian and Kokane, Shirley and Tan, Juntao and Yao, Weiran and Liu, Zhiwei and Feng, Yihao and others},
  journal={arXiv preprint arXiv:2406.18518},
  year={2024}
}
```

### Example

A single row from the dataset — one query, the tools available to answer it, and the
ground-truth tool call(s):

```json
{
  "query": "Find the sum of all the multiples of 3 and 5 between 1 and 1000. Also find the product of the first five prime numbers.",
  "tools": [
    {
      "name": "math_toolkit.sum_of_multiples",
      "description": "Find the sum of all multiples of specified numbers within a specified range.",
      "parameters": {
        "lower_limit": {
          "type": "int",
          "description": "The start of the range (inclusive).",
          "required": true
        },
        "upper_limit": {
          "type": "int",
          "description": "The end of the range (inclusive).",
          "required": true
        },
        "multiples": {
          "type": "list",
          "description": "The numbers to find multiples of.",
          "required": true
        }
      }
    },
    {
      "name": "math_toolkit.product_of_primes",
      "description": "Find the product of the first n prime numbers.",
      "parameters": {
        "count": {
          "type": "int",
          "description": "The number of prime numbers to multiply together.",
          "required": true
        }
      }
    }
  ],
  "answers": [
    {
      "name": "math_toolkit.sum_of_multiples",
      "arguments": {
        "lower_limit": 1,
        "upper_limit": 1000,
        "multiples": [3, 5]
      }
    },
    {
      "name": "math_toolkit.product_of_primes",
      "arguments": {
        "count": 5
      }
    }
  ]
}
```

This is exactly the shape the model needs to learn: read `query` and `tools`, produce the
`answers` array — correct tool names, correct argument keys, correct argument values.

## Model & Method

- **Base model:** Qwen2.5-3B-Instruct — comfortably fits 4-bit LoRA on a Kaggle T4/P100 via
  Unsloth (pre-quantized weights available), with headroom left in the free-tier GPU quota
- **Fine-tuning:** LoRA / QLoRA via [Unsloth](https://github.com/unslothai/unsloth)
- **Preference tuning:** DPO, using self-generated preference pairs — multiple completions
  sampled per held-out prompt from the SFT model, each checked against the dataset's own
  ground-truth answer to label chosen/rejected (no external preference dataset needed)
- **Compute:** Kaggle Notebooks, free T4/P100 GPU tier

## Pipeline

1. **Data prep** — load, parse, format into a chat template (system prompt with tool schemas +
   user query + assistant tool-call JSON), split ~59.7k train / 300 held-out eval
2. **Baseline eval** — base model, zero-shot, on held-out eval set
3. **LoRA SFT** — fine-tune on training split, re-evaluate
4. **DPO pair generation** — sample completions from the SFT model on held-out prompts,
   validate against ground truth, build chosen/rejected pairs
5. **DPO** — preference-tune on top of the SFT adapter, re-evaluate
6. **Final comparison** — baseline vs. SFT vs. DPO, side by side

## Metrics

- **JSON validity rate** — does the output parse at all
- **Tool-selection accuracy** — correct tool(s) chosen
- **Argument exact-match rate** — correct argument names and values
- **Adapter size on disk** — efficiency reference point (LoRA params vs. full model)

## Results

| Stage | JSON validity | Tool-selection accuracy | Argument exact-match |
|---|---|---|---|
| Baseline (zero-shot) | TBD | TBD | TBD |
| + LoRA SFT | TBD | TBD | TBD |
| + DPO | TBD | TBD | TBD |

*To be filled in after training runs complete.*

## Repro

Run the numbered Kaggle notebooks in order. Each expects Internet enabled and an `HF_TOKEN`
Kaggle Secret with access to the gated dataset.

1. `01_data_prep.ipynb` — produces `train.jsonl` / `eval.jsonl`
2. *(next)* baseline eval + LoRA SFT
3. *(next)* DPO pair generation + DPO training + final eval
