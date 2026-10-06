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

The debug profile stops at language-preservation and concept AUC gates, leaving
checkpoints intact. The 100M profile reports KL, NLL and concept AUC without
stopping on finite degradation or weak/unavailable AUC. Nonfinite losses and
validation values still stop training. Stop thresholds and diagnostic frequency may change on resume
without changing the dataset, optimizer, or training settings. Completion does
not imply capability preservation or successful concept detection.
Read `runs/100m/experiment/<stage>/status.json` to distinguish completion from a
gate failure. `PIPELINE_COMPLETE` is printed only when both stages and reports
finish. Run the same command to resume an interrupted trainer. Preparation itself
restarts if its manifest was never completed. Use `tmux` to keep the process alive
when disconnecting from a remote machine.

Every 10M processed tokens per stage, the 100M profile saves validation-only
concept, language, attribution, leakage and intervention diagnostics plus raw
steering examples in `experiment/<stage>/diagnostics/`. The log marks these with
`DIAGNOSTICS_START` and `DIAGNOSTICS_COMPLETE`. See the README for file names and
the limits of these monitoring tests. An existing process needs a checkpointed
restart to pick up newly pulled code or diagnostic settings.

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
