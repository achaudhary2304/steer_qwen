# Active experiment plan

## Question

Can a pretrained causal language model be post-trained into a model whose final
token representation is causally routed through named concepts, without a large
capability loss?

## Architecture

The active module sits immediately before Qwen's existing LM head:

```text
Qwen hidden state h_t
        |
        v
known concepts + unknown features + residual
        |
        v
reconstructed hidden state h_bar_t
        |
        v
existing LM head -> logits
```

The position immediately before the linear head gives an exact additive logit
decomposition. The residual begins near full strength, then its penalty and
dropout are scheduled upward. Full-residual identity is a safety initialization,
not a preservation result.

## Stages

1. **Layer probe:** measure concept accessibility at several Qwen layers.
2. **Frozen viability check:** train only the final bottleneck on 20M then 100M
   streamed Atlas tokens. Reject designs that degrade teacher agreement or leak
   labels through residual/unknown features.
3. **Top-layer adaptation:** apply LoRA to the top 20–30% of layers and train
   sequence-level next-token CE, teacher KL, positive-unlabeled concept loss,
   reconstruction, scheduled residual pressure, and leakage minimization.
4. **Scale only the winner:** 1,024–2,048 diverse concepts and 500M+ tokens.
5. **Steering:** create token-level labels for a small evaluation concept set;
   compare amplification and suppression with a matched activation-vector
   baseline.

## Required claims and measurements

Capability is measured with held-out next-token loss and external benchmarks.
Interpretability requires direct concept detection, sparse activations, logit
contribution, and lower concept recoverability from unknown/residual channels.
Controllability requires matched semantic steering tests, fluency checks, and
unrelated-concept side-effect measurements.

No experiment should claim capability preservation from an algebraic identity at
residual scale one, or claim token-level attribution from chunk-level labels.
