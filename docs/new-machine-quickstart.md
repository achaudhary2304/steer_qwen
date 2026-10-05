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

## 3. Optional Hugging Face authentication

```bash
hf auth login
```

The selected repositories are public. A token is optional but can help with
download limits. Model weights download into Hugging Face's normal cache;
they are not stored in Git. If useful, set `HF_HOME` to a disk with enough space.

## 4. Launch the real experiment

First verify the actual Qwen/data path on a small sample:

```bash
conda run --no-capture-output -n concept-retrofit bash scripts/run_experiment.sh debug
```

Then run the 100M-token-per-stage experiment on the larger GPU:

```bash
conda run --no-capture-output -n concept-retrofit bash scripts/run_experiment.sh 100m
```

The launcher creates its data and log directories, streams a bounded Atlas
sample, measures its unique training-token count, selects 1,024 concepts from
training support, downloads Qwen3.5-0.8B, runs layer probes, trains both stages,
and writes test reports plus generated steering examples. No laptop files are
required. Neither the large data preparation nor 100M-token training has been
completed on the laptop; small execution tests validate the code path.

The profiles stop at language-preservation validation gates and leave checkpoints
intact. The debug profile also stops on concept AUC below 0.5 or unavailable AUC.
The 100M profile logs concept AUC without using it as a stop condition, so weak
early concept detection does not prevent measuring the full-budget outcome.
Removing that stop condition does not mean concept detection passed.
Read `runs/100m/experiment/<stage>/status.json` to distinguish completion from a
gate failure. `PIPELINE_COMPLETE` is printed only when both stages and reports
finish. Run the same command to resume an interrupted trainer. Preparation itself
restarts if its manifest was never completed. Use `tmux` to keep the process alive
when disconnecting from a remote machine.

The command order is:

```text
prepare Atlas manifest -> probe Qwen layers -> frozen bottleneck check
-> top-layer LoRA run -> held-out capability/leakage/steering evaluation
```

See [the experiment plan](experiment-plan.md) for the required measurements.

Individual commands are available via:

```bash
python -m concept_retrofit.cli --help
python -m concept_retrofit.cli prepare-atlas --help
python -m concept_retrofit.cli train --help
```
