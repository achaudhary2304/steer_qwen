# Status

## Active repository state

- No training process is active.
- Active backbone default: `Qwen/Qwen3.5-0.8B`.
- The package currently provides the validated configuration schema, sparse
  final-layer bottleneck component, and core loss definitions.
- The next implementation milestone is a streaming Atlas reader plus a
  sequence-level top-layer-LoRA trainer.

## Completed diagnostic result

The retired frozen-reader prototype showed that a frozen bottleneck with
chunk-level labels does not establish named causal routing. On a matched
96-concept validation screen, its original baseline outperformed every stronger
concept-loss variant. Details and code are in
[`archives/laptop-prototype/`](archives/laptop-prototype/ARCHIVE.md).
