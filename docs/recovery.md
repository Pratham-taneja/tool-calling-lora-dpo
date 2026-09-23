# Recovery Notes

Date: 2026-09-17

## Provenance

The earlier project directory contained three readable source files:
`scripts/train_sft.py`, `src/tool_calling_lora_dpo/sft_training.py`, and
`tests/test_sft_training.py`. These were copied unchanged, including the latest
checkpoint-resume helpers.

Other project text was recovered from the original development conversation's
complete code blocks, full-file reads, and successful file patches. Recorded
shell commands were treated as data and were not replayed. A formatting change
prevented automatic recovery of one patch; its original `sys.path` bootstrap
changes were reapplied to `scripts/prepare_data.py` and `scripts/evaluate.py`.

The original prompt text, serialization rules, split logic, and experiment
configuration were preserved. The README and this recovery note are new.
The original experiment log and notebook retain historical results; embedded
notebook outputs and listed artifact paths do not imply those files survived.

## Reference hashes

These values were recorded during the earlier real runs:

| Identity | SHA-256 |
| --- | --- |
| Experiment configuration | `214ec2c3b912d3056f659478b1108cffe31d931628284073469a921549615ba4` |
| Train IDs | `f66e7e3c8e9e6dad0d29d537ea3936d9f4655c8d9d9882803f5e1504b67241f3` |
| Validation IDs | `7f8fc449367012f64e27d606f2a002298ee126524bd040ef28c741451cb18e9c` |
| Test IDs | `f25b889ef7215a7b94d24f4c3d89e7c313bc6d3917e9d1a11290cad8ec922ea6` |

Verify these against regenerated data before comparing new results with the
historical baseline and pilot. Matching ID hashes verifies split membership,
not the contents of the remote dataset or model weights.

## Missing artifacts

The previous trained adapters, full-SFT checkpoints, generated predictions,
and processed dataset files were not recovered. The last user-reported full
training progress was approximately 2,895 of 3,668 steps. That progress cannot
be resumed without its corresponding checkpoint.

The new environment is created independently; the damaged old environment
is not copied. Core package versions are pinned to the earlier working stack.
Transitive dependency versions may differ; the new environment's lock file
records the versions actually installed.

No full training run is automatically started as part of recovery.

## Recovery verification

- All 39 recovered project text files were verified against their source.
- The instruction prompt was compared directly with its original text.
- The configuration hash matches the earlier recorded value.
- The complete restored test suite passed: 144 tests.
- Ruff passed after restoring five import-formatting fixes.
- `pip check` reported no broken requirements.
- CUDA preflight passed with PyTorch `2.12.1+cu132` and CUDA `13.2`.
- The Trainer/PEFT imports and a small bitsandbytes NF4 CUDA forward pass passed.

Two original base-model weight shards were copied and verified by SHA-256:

| File | SHA-256 |
| --- | --- |
| `model-00001-of-00002.safetensors` | `67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2` |
| `model-00002-of-00002.safetensors` | `a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d` |

These are pretrained base weights, not the missing fine-tuned adapters.
The cache snapshot is `aa8e72537993ba99e69dfaafa59ed015b17504d1`.

The initial data preparation attempt was blocked by missing Hugging Face
authentication. After signing in with `HF_HOME` set to this project's
`data/hf_cache` directory, data preparation completed successfully.

## Data rebuild verification

Verified on 2026-09-17. Machine-readable results are saved in
`results/recovery_data_verification.json`.

- Downloaded and parsed all 60,000 source examples.
- Regenerated 59,200 train, 500 validation, and 300 test examples.
- All three split ID hashes and the configuration hash match the original run.
- Verified unique IDs within each split and no IDs shared across splits.
- Token audit matches the historical results: p50 402, p90 678, p95 784,
  p99 1,005, maximum 2,271 tokens.
- Rebuilt 58,682 SFT records and recorded 518 over-length exclusions.
- Verified that kept and excluded IDs partition only the training split.
- Every kept record fits 1,024 tokens and has a nonempty assistant target.
- The training dry run succeeded with the original full-SFT settings.

The dataset revision used was `26d14ebfe18b1f7b524bd39b404b50af5dc97866`.
The model revision remains `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
New processed-file hashes were recorded for future checks. Old row-content
hashes are unavailable, so historical parity is verified for ID membership
and audit statistics, not byte-for-byte source rows.

The SFT builder initially exited with a console encoding error after writing
both output files. The Japanese folder name could not be printed through
the redirected Windows cp1252 stream. Setting `PYTHONUTF8=1` resolved console
output for subsequent validation. No prompt, training, or filtering code was
changed, and the generated files passed the complete data verification.

Full training has not restarted. Missing old adapters/checkpoints still
require a fresh training run after the remaining training smoke gate.
