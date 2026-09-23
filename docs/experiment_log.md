# Experiment Log

This file records completed experiment milestones and the exact numbers produced by local runs.
Only include results that came from real commands and saved artifacts.

## Baseline evaluation

Date recorded: 2026-09-14

Stage: `baseline`

Model: `Qwen/Qwen2.5-3B-Instruct`

Split: `test`

Examples: `300`

Decoding:

- `do_sample`: `false`
- `max_new_tokens`: `384`
- `max_seq_length`: `1024`

Runtime:

- Device: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`
- CUDA available: `true`
- Allocated memory: `1977.45 MB`
- Reserved memory: `2510.0 MB`
- Max allocated memory: `2199.67 MB`
- Max reserved memory: `2510.0 MB`
- Elapsed time: `1647.58 seconds`
- Seconds per generated record: `5.49`

Metrics:

| Metric | Value |
| --- | ---: |
| `json_valid` | `0.9300` |
| `schema_valid` | `0.8867` |
| `call_count_correct` | `0.8400` |
| `tool_correct` | `0.8333` |
| `args_exact` | `0.5033` |
| `args_exact_given_tool` | `0.6040` |
| `end_to_end_exact` | `0.5033` |

Error counts:

| Error category | Count |
| --- | ---: |
| `exact` | `151` |
| `wrong_argument_value` | `63` |
| `extra_argument` | `26` |
| `malformed_json` | `21` |
| `wrong_call_count` | `14` |
| `invalid_arguments` | `11` |
| `missing_argument` | `10` |
| `wrong_tool` | `2` |
| `invalid_call_item` | `2` |

Run identity:

- Config hash: `214ec2c3b912d3056f659478b1108cffe31d931628284073469a921549615ba4`
- Manifest hash: `902340453022cca5e44b8b2beb4e458d6378149a00d365e2cfe4519ae021d011`
- Example IDs hash: `f25b889ef7215a7b94d24f4c3d89e7c313bc6d3917e9d1a11290cad8ec922ea6`
- Evaluation hash: `0b70ed4480ffaaf8c13dc5ca72abe9e7747e11e2d572b5969598f7b99ac80472`

Artifacts:

- `artifacts/predictions/baseline_eval.jsonl`
- `artifacts/evaluations/baseline_eval.json`
- `artifacts/checkpoints/evaluation/baseline_eval_checkpoint.json`

## SFT smoke training

Date recorded: 2026-09-14

Stage: `sft`

Purpose: verify that the SFT trainer can load the base model in 4-bit mode, attach a LoRA adapter,
run optimizer steps, save the adapter, and reload the adapter through the evaluator.

This is a smoke test only. Do not report it as a trained-model result.

Training setup:

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Input records: `16`
- Max train steps: `2`
- LoRA rank: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.0`
- Assistant-only loss: `true`
- Optimizer: `paged_adamw_8bit`
- Gradient checkpointing: `true`

Training runtime:

- Device: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`
- Max allocated memory: `4601.37 MB`
- Max reserved memory: `5674.0 MB`
- Train runtime: `51.69 seconds`
- Train loss: `0.8990`

Adapter smoke evaluation:

| Metric | Value |
| --- | ---: |
| `json_valid` | `1.0000` |
| `schema_valid` | `1.0000` |
| `call_count_correct` | `1.0000` |
| `tool_correct` | `1.0000` |
| `args_exact` | `0.6000` |
| `args_exact_given_tool` | `0.6000` |
| `end_to_end_exact` | `0.6000` |

Artifacts:

- `artifacts/models/sft_lora_smoke`
- `artifacts/training/sft_train_smoke_report.json`
- `artifacts/predictions/sft_smoke_eval.jsonl`
- `artifacts/evaluations/sft_smoke_eval.json`
- `artifacts/checkpoints/evaluation/sft_smoke_eval_checkpoint.json`

## SFT pilot training

Date recorded: 2026-09-14

Stage: `sft`

Purpose: run a medium SFT pilot before committing to full one-epoch training.

Training setup:

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Input records: `1024`
- Max train steps: `64`
- LoRA rank: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.0`
- Assistant-only loss: `true`
- Optimizer: `paged_adamw_8bit`
- Gradient accumulation steps: `16`

Training runtime:

- Device: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`
- Max allocated memory: `5392.2 MB`
- Max reserved memory: `9882.0 MB`
- Train runtime: `1734.45 seconds`
- Train samples per second: `0.59`
- Train steps per second: `0.037`
- Train loss: `0.1005`
- Epoch: `1.0`

Run identity:

- Config hash: `214ec2c3b912d3056f659478b1108cffe31d931628284073469a921549615ba4`
- Train IDs hash: `0746bffe00d95d782acb9d913c951c92b6a2e4cc63c1a6bc4e70b5bedabb2bb4`

Artifacts:

- `artifacts/models/sft_lora_pilot`
- `artifacts/training/sft_train_pilot_report.json`

Pilot adapter evaluation:

This is a 50-example frozen test subset evaluation. Treat it as a directional pilot signal,
not the final SFT result.

| Metric | Value |
| --- | ---: |
| `json_valid` | `1.0000` |
| `schema_valid` | `1.0000` |
| `call_count_correct` | `0.9600` |
| `tool_correct` | `0.9600` |
| `args_exact` | `0.7000` |
| `args_exact_given_tool` | `0.7292` |
| `end_to_end_exact` | `0.7000` |

Evaluation error counts:

| Error category | Count |
| --- | ---: |
| `exact` | `35` |
| `extra_argument` | `5` |
| `missing_argument` | `4` |
| `wrong_argument_value` | `4` |
| `wrong_call_count` | `2` |

Evaluation runtime:

- Device: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`
- Max allocated memory: `2217.72 MB`
- Max reserved memory: `2416.0 MB`

Evaluation artifacts:

- `artifacts/predictions/sft_pilot_eval_50.jsonl`
- `artifacts/evaluations/sft_pilot_eval_50.json`
- `artifacts/checkpoints/evaluation/sft_pilot_eval_50_checkpoint.json`

Full pilot adapter evaluation:

This is the full 300-example frozen test evaluation for the 1024-record SFT pilot adapter.

| Metric | Baseline | SFT pilot |
| --- | ---: | ---: |
| `json_valid` | `0.9300` | `0.9967` |
| `schema_valid` | `0.8867` | `0.9967` |
| `call_count_correct` | `0.8400` | `0.9767` |
| `tool_correct` | `0.8333` | `0.9700` |
| `args_exact` | `0.5033` | `0.7067` |
| `args_exact_given_tool` | `0.6040` | `0.7285` |
| `end_to_end_exact` | `0.5033` | `0.7067` |

Evaluation error counts:

| Error category | Count |
| --- | ---: |
| `exact` | `212` |
| `wrong_argument_value` | `34` |
| `extra_argument` | `28` |
| `missing_argument` | `17` |
| `wrong_call_count` | `6` |
| `wrong_tool` | `2` |
| `malformed_json` | `1` |

Evaluation runtime:

- Device: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`
- Max allocated memory: `2257.52 MB`
- Max reserved memory: `2644.0 MB`
- Elapsed time: `2204.79 seconds`
- Seconds per generated record: `7.35`

Run identity:

- Manifest hash: `902340453022cca5e44b8b2beb4e458d6378149a00d365e2cfe4519ae021d011`
- Example IDs hash: `f25b889ef7215a7b94d24f4c3d89e7c313bc6d3917e9d1a11290cad8ec922ea6`
- Evaluation hash: `ce86ab0af562ca9ceda54f0ef9ea842e7ab9e8253de044af52ea372cf465d2bd`

Evaluation artifacts:

- `artifacts/predictions/sft_pilot_eval.jsonl`
- `artifacts/evaluations/sft_pilot_eval.json`
- `artifacts/checkpoints/evaluation/sft_pilot_eval_checkpoint.json`
