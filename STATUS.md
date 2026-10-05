# Status

## Active repository state

- No training process is active.
- Active backbone default: `Qwen/Qwen3.5-0.8B`.
- The package provides bounded Atlas preparation, layer probes, sequence-level
  frozen and top-layer-LoRA trainers, checkpoint resume, held-out evaluation,
  and native-bottleneck steering examples.
- `scripts/run_experiment.sh` launches the debug or 100M-token profiles.
- An offline tiny-Qwen integration suite and a two-step-per-stage pretrained
  Qwen3.5-0.8B/real-Atlas smoke test passed. The 100M-token study is not yet run.

## Completed diagnostic result

The retired frozen-reader prototype showed that a frozen bottleneck with
chunk-level labels does not establish named causal routing. On a matched
96-concept validation screen, its original baseline outperformed every stronger
concept-loss variant. Details and code are in
[`archives/laptop-prototype/`](archives/laptop-prototype/ARCHIVE.md).
