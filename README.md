# Concept Retrofit

Can a pretrained causal language model be post-trained so its final prediction
path uses sparse, named concepts while preserving useful capability?

This repository studies that question with Qwen3.5-0.8B first. It is **not** a
Steerling reproduction: Steerling is an 8B causal-diffusion model trained at a
much larger scale. This project tests whether a concept bottleneck can be
retrofitted into a pretrained autoregressive model.

## Repository layout

```text
concept_retrofit/     active package
  models/             final-layer bottleneck components
  training/           post-training losses and loops
  data/               Atlas manifests and streaming readers
  evaluation/         capability, leakage, and steering evaluations
  cli/                public commands
configs/              phase-specific experiment configurations
docs/                 method and experiment plan
archives/             organized, retired laptop prototypes
references/           Guide Labs paper source and repository copies
tests/                tests for active code only
```

`runs/`, downloaded data, checkpoints, and logs are deliberately ignored by
Git. They are experiment artifacts, not source code.

## Active method

```text
Qwen final hidden state h_t
             |
             v
known concepts + unknown features + residual
             |
             v
reconstructed state h_bar_t
             |
             v
existing Qwen LM head -> next-token logits
```

The module is positioned directly before Qwen's linear LM head, so its known,
unknown, and residual components have an exact additive logit decomposition.
The active plan then adapts Qwen's upper layers with LoRA; it does not claim
that a frozen classifier constitutes causal interpretability.

Read [the experiment plan](docs/experiment-plan.md) before starting a run.

## Setup

Use the base Conda environment:

```bash
conda activate base
python -m pip install -e .
python -m unittest discover -s tests -v
```

Validate a configuration:

```bash
python -m concept_retrofit.cli validate-config \
  --config configs/top_lora/qwen35-08b_1024concept_100m.json
```

The checked-in configurations are plans, not a claim that their data manifests
already exist. First create an Atlas manifest under `data/manifests/`; then run
the layer-probe stage before selecting a bottleneck insertion layer.

## Experiment order

1. `configs/probe/`: identify which Qwen layer linearly exposes the selected
   concepts.
2. `configs/frozen_bottleneck/`: establish whether a final bottleneck can work
   without backbone updates.
3. `configs/top_lora/`: adapt the top layers with LoRA, teacher distillation,
   scheduled residual pressure, and leakage measurements.
4. Scale only after the held-out capability, concept-routing, and steering
   checks agree.

## Historical work

[`archives/laptop-prototype/`](archives/laptop-prototype/ARCHIVE.md) contains
the prior prototypes and their exact configs. They remain available for audit,
but are not importable as the active package.
