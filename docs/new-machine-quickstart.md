# New-machine quickstart

These commands assume an NVIDIA driver is already installed and that Conda is
available. The repository does not include Atlas data, Qwen weights, Hugging
Face cache files, checkpoints, or experiment runs.

## 1. Clone and create the environment

```bash
git clone git@github.com:achaudhary2304/steer_qwen.git
cd steer_qwen
conda env create -f environment.yml
conda run -n concept-retrofit bash scripts/bootstrap.sh
```

If the machine already has a suitable CUDA-enabled PyTorch installation, use
that environment instead and run `python -m pip install -e .`.

## 2. Verify the repository before downloading model or data

```bash
conda run -n concept-retrofit bash scripts/bootstrap.sh
```

This performs four checks:

1. installs the local package in editable mode;
2. reports PyTorch/CUDA/GPU availability;
3. runs the active unit tests;
4. trains the synthetic bottleneck smoke test and writes
   `runs/smoke/synthetic-report.json`.

The smoke test must pass before running any real experiment.

## 3. Authenticate for model and dataset access

```bash
huggingface-cli login
```

Use a Hugging Face token with access to the selected Qwen model and Atlas/FineWeb
dataset release. Model weights will download into Hugging Face's normal cache;
they are not stored in Git.

## 4. Real-experiment status

The repository currently has validated configs for layer probing, frozen
bottleneck checks, and top-layer-LoRA runs. The streaming Atlas preparation,
Qwen layer-probe runner, and sequence-level LoRA trainer are the next code
milestone and are not yet exposed as commands. Do not create empty manifests or
claim a real run from the synthetic command.

When those commands land, the intended order is:

```text
prepare Atlas manifest -> probe Qwen layers -> frozen bottleneck check
-> top-layer LoRA run -> held-out capability/leakage/steering evaluation
```

See [the experiment plan](experiment-plan.md) for the required measurements.
