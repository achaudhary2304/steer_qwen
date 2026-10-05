# Laptop prototype archive

This folder preserves the experiments run before the repository was reorganized
on 2026-10-05. None of its scripts are part of the active training interface.

| Folder | Contents | Why archived |
|---|---|---|
| `data-preparation/` | Atlas staging and 300k-sample preparation | Dataset code coupled to RAM-resident activation caches. |
| `frozen-reader/` | Frozen Qwen reader, scaled bottleneck, configs, and tests | Demonstrated that weak chunk supervision plus residual/unknown bypass does not create a faithful bottleneck. |
| `native-output-bottleneck/` | Early LM-head hook prototype | Useful implementation reference, not a full post-training system. |
| `steering-sidecar/` | Steering and activation-vector pilots | Sidecar steering is not evidence that a named bottleneck routes generation. |
| `evaluation-and-audits/` | Prototype audits and benchmark scripts | Inputs, paths, and metrics are specific to retired runs. |
| `retired-1.5b/` | Older 1.5B settings | Retained only for traceability. |
| `support/` | Prompt and I/O helpers | Coupled to prototype module names. |

## Key conclusion

The strongest frozen-reader run used 640 named concepts and 1,920 unknown
features. On a matched validation screen, increasing the concept-loss weight
made all tracked outcomes worse. Residual and unknown channels still allowed
near-complete recovery of the concept labels. The active repository therefore
starts from a different hypothesis: train a causal final-layer bottleneck while
adapting Qwen's upper layers with LoRA and measuring leakage explicitly.
