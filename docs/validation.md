# Execution validation

The implementation was checked locally in the base Conda environment on an
RTX 3050 Ti laptop. Released Transformers 5.18.0 was also installed into an
isolated import directory for compatibility testing without changing base.

## Offline tests

Eight tests cover sparse bottleneck gradients/interventions, held-out synthetic
examples, a tiny real Qwen3.5 hybrid text architecture, the complete training and
evaluation path, teacher invariance when LoRA is disabled, exact checkpoint
resume, corrupted data rejection, and validation gate/token-budget failures.
These tests require no network or pretrained model downloads.

## Pretrained model smoke test

Cached pinned Qwen3.5-0.8B weights were exercised with 128 short real Atlas chunks,
two selected concepts, two training steps per stage, and a two-document test
sample. Both frozen and LoRA stages completed and wrote native-bottleneck
generation examples. A separate LoRA smoke test covered the top six layers,
including linear-attention and full-attention projections.

The first smoke report had 39 test target tokens, base NLL 3.5269 and retrofit
NLL 3.5486 at residual scale 0.9. AUC was correctly null because the test sample
had no concepts with both positive and unassigned examples. These are execution
checks and provide no evidence about interpretability or semantic steering
quality.

## Public data and CLI

The public `prepare-atlas` command fetched the pinned Guide Labs metadata and
a bounded prefix of shard 0, wrote split files and a manifest, and exited cleanly.
The tested path disables Arrow pre-buffering and closes suspended parquet
generators. This fixed a Python 3.13 shutdown failure found in the first attempt.

The public `run-all` command completed on the offline tiny-Qwen fixture. A
repeat with `--resume` skips completed trainers and regenerates evaluation
artifacts. The 100M-token profile is configured and budget-checked, but has not
been executed at that scale on this laptop.
