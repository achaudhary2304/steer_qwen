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

For a fresh computer, follow the [new-machine quickstart](docs/new-machine-quickstart.md).
Execution checks and their limits are recorded in [validation.md](docs/validation.md).

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

For a fresh machine with an NVIDIA GPU:

```bash
conda env create -f environment.yml
conda activate concept-retrofit
```

To use the existing base Conda environment on this laptop:

```bash
conda activate base
python -m pip install -e .
python -m unittest discover -s tests -v
```

`environment.yml` creates a Python 3.11 Conda environment and installs PyTorch
2.7.1 and Transformers 5.18.0 through pip. The Linux PyTorch wheel supplies its
CUDA runtime dependencies; an NVIDIA driver is still required. LoRA is
implemented explicitly in `models/qwen.py`. Optional Qwen
performance kernels are not required for correctness and are intentionally not
part of the environment definition.

Validate a configuration:

```bash
python -m concept_retrofit.cli validate-config \
  --config configs/runnable/qwen35-08b-atlas-100m.json
```

## Start here: synthetic smoke test

Run this before downloading data or loading Qwen:

```bash
python -m concept_retrofit.cli synthetic-smoke \
  --steps 500 \
  --out runs/smoke/synthetic-report.json
```

It generates hidden states with six known ground-truth concepts, trains the
actual sparse bottleneck, then checks concept detection, reconstruction,
teacher-logit agreement, and the effect of a named intervention on LM-head
logits. A pass verifies the module and loss plumbing only; it does not show
that the method works on Qwen or Atlas.

## Run the actual pipeline

For a small real-data execution check:

```bash
bash scripts/run_experiment.sh debug
```

For the agreed first substantive study on the larger GPU:

```bash
bash scripts/run_experiment.sh 100m
```

These commands download the pinned FineWeb Atlas data subset and Qwen weights
as needed, prepare document-disjoint splits, probe several Qwen layers, train
the frozen bottleneck, initialize top-layer LoRA from its selected checkpoint,
and write held-out evaluation reports and steering examples. The final
bottleneck always remains immediately before the LM head; layer probes diagnose
concept accessibility rather than changing the insertion point.

The `100m` profile uses 1,024 concepts and requires at least 100M **unique training
target tokens** in the prepared corpus. Each training stage processes 100M
tokens, so the two stages together process about 200M tokens. Sampling is bounded
over shard prefixes, not uniform across the full Atlas corpus. Preparation fails
clearly if its sample does not meet the requested token or concept support
budget. Increase sampling limits in the launcher if that happens.

Logs are `runs/<profile>/prepare.log` and `runs/<profile>/pipeline.log`. Rerun the
same launcher to resume saved training checkpoints; completed stages are skipped.
Checkpoints include optimizer and RNG/sampler state. Dataset preparation is
restarted if interrupted before its manifest is written. In the debug profile,
validation KL above 0.5, NLL increase above 0.3 nats/token, or annotation AUC below
0.5/unavailable stops training. The 100M profile reports these metrics without
stopping on finite capability degradation or weak concept detection. Nonfinite
loss/validation values still stop training. Resume permits changes to stop
thresholds and diagnostic/judging settings; training settings and dataset identity must match.

During residual warm-up, validation uses the current training residual scale,
which is printed in each validation line. Final held-out reports use the target
scale (0.75 in the supplied profiles). `last.pt` is saved throughout warm-up;
`best.pt` selection starts only when the target scale is reached, so near-identity
warm-up checkpoints cannot win against fully compressed checkpoints. Resuming
older checkpoints retains training state but resets their old best-selection score.

The 100M profile also runs validation-only diagnostics every 10M processed
tokens in each stage. Files live under
`runs/100m/experiment/<stage>/diagnostics/tokens-XXXXXXXXX/`:

- `validation-current.json`: current-scale NLL/perplexity, KL, AUC/AP and logit shares;
- `validation-target.json`: the same audit at target residual scale 0.75;
- `leakage.json`: linear readers fit on up to 256 training chunks and scored on
  up to 256 validation chunks;
- `interventions.json`: amplification/suppression prediction changes on up to
  16 annotated and 16 unassigned chunks per concept, using the last next-token
  target in each chunk;
- `steering-concept-*.json`: matched base/retrofit/amplification/suppression
  outputs for up to three concepts, one neutral and one concept-eliciting prompt,
  and a 32-token output cap;
- `judge-summary.json`: GPT-OSS-20B semantic scores, direction success rates,
  opportunity counts, quality changes and API failure/skip counts;
- `judge-records/*.json`: exact judging prompts, blinded candidate mappings,
  raw provider responses, validated scores and failed attempts;
- `status.json`: completion marker, step, token count and residual scales.

These diagnostics reuse the resident model, preserve training RNG state, and
never use the held-out test split. Concepts are selected by validation support;
they are a small monitoring sample, not a representative semantic steering
benchmark. No metric threshold from this audit stops training. Target-scale
results during warm-up are stress tests, not evaluations of the current training
path. Periodic diagnostics add runtime and cannot activate in an already-running
Python process; pull updates and resume from the saved checkpoint to enable them.

### Semantic judging through Groq

The 100M profile enables `diagnostics_judge`. It uses Groq's
`openai/gpt-oss-20b` with JSON responses and a validated 0-4 rubric for concept
presence, fluency, prompt following, and unrelated-meaning preservation.
Candidate order is shuffled deterministically and intervention labels are
hidden from the judge. Target-presence changes are measured relative to the
unsteered retrofit. Suppression success excludes cases where the concept was
already absent; amplification excludes already-maximal scores. API errors and
invalid/truncated judge responses are never counted as successes. Missing keys
are reported as skipped, not silently scored. Provider failures do not halt
training, and completed response groups are reused when rerunning the judge.

Requests start at least 10 seconds apart across judging processes on the same
machine. A milestone needs up to six requests, plus retries, so judging adds at
least about a minute of waiting/response time. No local judge model is loaded.
The API receives generated text and concept definitions, not weights or keys
embedded in prompts. Credentials come from `GROQ_API_KEY` or the private file
`~/.config/concept-retrofit/groq.key`; never commit credentials.

To judge or retry one already-saved diagnostic folder without restarting GPU
training:

```bash
python -m concept_retrofit.cli judge --diagnostics \
  runs/100m/experiment/frozen/diagnostics/tokens-060000000
```

These are small, potentially truncated monitoring samples, not a full steering
benchmark. A different judge/rubric does not reproduce Steerling's reported
numbers. External capability benchmarks and activation-steering comparisons
remain separate work.

For a separate content-concept audit with longer outputs:

```bash
python -u -m concept_retrofit.cli content-steer --data data/atlas-100m \
  --checkpoint runs/100m/experiment/frozen/last.pt \
  --out runs/100m/content-audit --max-new-tokens 96 \
  --concepts 'Music,Home cooking,Astronomy,Computer Science'
```

This copies the checkpoint to a fixed snapshot, generates all four conditions
for a neutral and a topic-specific prompt per concept, records factual concept
scores/top-k activation rates, and judges outputs through Groq. It does not stop
the trainer, but running concurrently uses additional RAM/VRAM and shares GPU
time. Use a fresh output folder for each audit. Judge errors can be retried with
the `judge` command above, without regenerating text.

`content-steer --strengths 2,4` sweeps stronger signed coefficients on one fixed
checkpoint. Strength 1 clamps amplification to 1 and suppression to 0; strength
2 uses 2/-1, and strength 4 uses 4/-3. These are coefficient overrides, not
multipliers of the factual score. Values outside [0,1] extrapolate beyond the
training activation range. The exact hidden-state change is
`(override - factual_sparse_activation) * learned_concept_vector`, holding
unknown and residual contributions fixed for that forward pass. Multi-strength
audits save each setting in `strength-*/` with a shared checkpoint snapshot and
write `strength-summary.json` for comparison.

The test reports contain teacher/base and retrofit perplexity, KL, annotation
AUC/AP, named/unknown/residual logit shares, and post-hoc linear leakage probes.
Saved steering outputs include base, retrofit, amplification, and suppression.
They are inspection examples, not semantic judge scores or comparisons with
ordinary activation steering. External maths/code/reasoning benchmarks and a
matched activation-steering baseline are later evaluations.

## Experiment order

`configs/runnable/` contains executable pipeline configs. The other config
folders retain the original design proposals and are not trainer inputs.

1. Probe concept accessibility across depth on train/validation only.
2. Train the final bottleneck with frozen Qwen.
3. Adapt the top six layers with LoRA, teacher KL, next-token CE, chunk-label
   supervision, reconstruction, residual pressure, and a linear leakage adversary.
4. Evaluate each selected checkpoint on the same held-out test split.

Chunk supervision predicts released machine-annotation membership using weighted
BCE. Unassigned labels are proxy zeros, not certified semantic negatives. This
implementation does not claim a statistically calibrated positive-unlabeled
estimator or token-level supervision. A linear adversary also does not guarantee
removal of all concept information from the bypass channels.

## Historical work

[`archives/laptop-prototype/`](archives/laptop-prototype/ARCHIVE.md) contains
the prior prototypes and their exact configs. They remain available for audit,
but are not importable as the active package.


### Intervention-aware steering pilot

The frozen/LoRA stages above train chunk detection and the bottleneck. They do
**not** train response to interventions. The optional `steering` stage now does:

- inject a normalized known-concept vector before every selected upper block;
- calibrate its scale with `gamma = tau / max_v(W_v · e_c)`;
- supervise `respond = -log(sigmoid(concept_logit))` at injected positions;
- supervise `express = -log(sum of next-token probabilities over the concept's
  high-lift vocabulary tokens)`;
- retain next-token CE and teacher KL; disable the chunk, reconstruction,
  residual and adversarial losses in this dedicated stage;
- suppress with negative layer injections plus
  `logits -= suppression_strength * relu(W · e_c)`.

This adapts [Steerling §§6.2 and 10.2.4](https://arxiv.org/html/2608.07594v1)
to **causal autoregressive** Qwen. The prediction state at position `t` is
supervised for a concept-expressing target at `t+1`. Injection at inference is
at the last causal prediction position; no diffusion masks are used.

**Token annotation limitation:** Atlas is chunk-labelled. The first pilot
uses chunk membership plus training-only token lift to infer weak token
positions. This is not Guide Labs' token-attributed dataset. Lexical resources
are fitted on the training split only; no validation/test text is used.
`tau` is a calibration reference, not a guaranteed total logit shift after
nonlinear Transformer propagation. These changes do not establish superiority
over ordinary activation steering, and suppression is not separately trained
with a negative-intervention loss.

Run the bounded pilot without changing a running trainer:

```bash
python -u scripts/steering_pilot.py \
  --data data/atlas-100m \
  --checkpoint runs/100m/experiment/frozen/last.pt \
  --out runs/steering-pilot \
  --strengths 1 --max-new-tokens 64 \
  > runs/steering-pilot.log 2>&1
```

The pilot copies the checkpoint, builds a train-only lexicon for Music, Home
cooking, Astronomy and Computer Science, evaluates **the same calibrated
interface before and after training**, and runs 500 top-layer LoRA/bottleneck
updates. Groq judging stays paced at ten seconds per request. Raw prompts,
answers, judge responses, validation metrics, injection counts and resumable
training checkpoints are saved under the output directory. Look for
`STEERING_TRAIN`, `VALIDATION steering` and `STEERING_PILOT_COMPLETE` in the log.
The token-limited examples are monitoring evidence, not a full benchmark.

For separate preparation/training:

```bash
python -m concept_retrofit.cli prepare-steering --data data/atlas-100m \
  --concept-indices 11,61,433,968 --out runs/steering-pilot/lexicon.json
python -u -m concept_retrofit.cli train --data data/atlas-100m \
  --config configs/runnable/qwen35-08b-steering-pilot.json --stage steering \
  --initialize runs/100m/experiment/frozen/last.pt --out runs/steering-pilot/training
```

Indices above belong to the current local vocabulary; use the names in
`scripts/steering_pilot.py` when preparing a new Atlas sample. A future fresh
`run-all` can append this stage with `--steering-config <config.json>` after
LoRA; the lexicon must already exist. An already-running Python process does
not acquire these changes automatically.

Judge rubric v2 explicitly separates concept presence from fluency, word
limits, and resemblance to the unsteered answer. Rejudge both sides with the
same rubric before comparing; older rubric v1 scores remain historical results.


For the larger follow-up, use `configs/runnable/qwen35-08b-steering-scaled.json`:
5,000 steps with every step receiving steering supervision. Concepts rotate
equally; half each batch contains training chunks with verified eligible weak
positions for that concept, and half comes from random training chunks. Both
halves retain CE/KL. Pool sizes and actual injected positions are logged; this
balances repeated exposures, not necessarily unique data. `--lexicon-documents
100000 --expanded-prompts` fits the lexical resources on 100,000 training chunks
and evaluates four prompts per concept before/after (16 judge groups each).
This remains four concepts with approximate token labels, not 1,024 independently
validated steering controls.


To append steering to an **already-running** main experiment, use
`scripts/steering_after_main.py --main runs/100m/experiment --parent-pid <PID>
--data data/atlas-100m --out runs/100m/steering-after-lora`. It polls every five
minutes and starts only after the frozen and LoRA stages complete and the parent
trainer exits. This stage initializes both the bottleneck and existing LoRA
adapters from `lora/best.pt`; independent pilot adapter weights are not added
together. Queue state is written to `<out>-queue.json`.

This sequential schedule is a retrofit experiment, not Steerling's exact
schedule: Steerling interleaves steering phases early in mid-training, then
continues ordinary capability training. The standalone pilots remain diagnostic
controls; the queued stage produces the combined main-plus-steering model.


### One combined model with interleaved steering

`scripts/merged_retrofit.py` continues the completed main LoRA checkpoint and
retains the same bottleneck, LoRA parameters, and Adam state across all phases.
It fits weak lexical resources for the entire known vocabulary and reports the
eligible subset in `coverage.json`; it does not claim unsupported labels are
trained steering controls. The schedule alternates four 5,000-step steering
phases with three 2,000-step capability phases and a final 5,000-step capability
phase. Capability phases keep the trained residual scale instead of restarting
an identity warmup. Teacher KL is retained as a retrofit-specific addition.

Matched content audits run before the schedule and after every phase to detect
steering improvements or forgetting. Final held-out interpretability/language
metrics go to `final-test.json`; the deliverable is **one `final.pt` checkpoint**
containing both the bottleneck and trained LoRA adapters. Generated examples
now include signed per-token named/unknown/residual contributions, suppression
logit corrections, and the floating-point rounding gap. These are output-layer
attributions, not full explanations of the Transformer computation.

Append this schedule without interrupting a running main trainer:

```bash
python -u scripts/steering_after_main.py --merged \
  --main runs/100m/experiment --parent-pid <PID> \
  --data data/atlas-100m --out runs/100m/merged \
  > runs/100m/merged-queue.log 2>&1
```

Run directly from a completed main model:

```bash
python -u scripts/merged_retrofit.py --data data/atlas-100m \
  --checkpoint runs/100m/experiment/lora/best.pt --out runs/100m/merged
```

Remaining differences from Steerling are material: Qwen is autoregressive,
Atlas token positions here are lexical proxies, chunk scores use mean pooling,
residual pressure uses scaling rather than their dropout, and independence is
an adversarial objective. Original Qwen pretraining provenance is unavailable.
The script records these differences in its final status. Architecture and
training components being present do not establish comparable capability;
benchmarks and semantic intervention evaluations must demonstrate that.


The combined run evaluates up to **32 eligible content concepts** before and
after all training, with eight monitored after each phase. Selection spreads
across the available taxonomy groups; actual names/counts are written to
`evaluation-concepts.json`. Training eligibility spans the full known bank,
not just the evaluation concepts. `bash scripts/run_experiment.sh 100m` now
appends the combined schedule by default. The already-running process uses its
original loaded code, so a waiting queue connects its final LoRA checkpoint.

### Amplification calibration

The 5,000-step pilot generated repeated concept words at injection target
`tau=0.1`, despite fluent unsteered output. Training and inference injection
targets can now be recorded separately: `steering_tau` is the training target;
`steering_inference_tau` overrides it only for generation. The scaled recipe
initially tested an inference target of `0.02`. This is a calibration correction, not proof
that all concepts steer successfully. Before/after audits use the same target.

Run a matched strength diagnostic on a saved model, keeping greedy decoding:

```bash
python scripts/steering_strength_sweep.py --data data/atlas-100m \
  --checkpoint runs/steering-pilot-v2/training/last.pt \
  --out runs/amplification-strength-test --strengths 0.05,0.1,0.2,1 \
  --judge --save-strength 0.2
```

Strength multiplies the checkpoint's training target (`0.1` in this pilot), so
`0.2` corresponds to an inference target of `0.02`. The separate `calibrated.pt`
contains unchanged weights with the tested inference setting. Raw text,
repetition diagnostics, and paced GPT-OSS judge results remain in the output.
The small failure-focused diagnostic is not a held-out steering benchmark.

`qwen35-08b-steering-repair.json` is an experimental 1,000-step continuation
of the failed pilot. It retains its LoRA/bottleneck and optimizer state, reduces
respond/express weights to `0.25`, and adds `injected_ce_weight=1`: next-token
cross-entropy averaged over the injected causal positions. Global LM loss is
still active. This addresses normalization imbalance between whole-batch LM
loss and losses averaged over a small number of injected states. The extra
local LM objective is our AR retrofit adaptation, not Steerling's published
loss. It must pass matched semantic and fluency tests before replacing the
larger recipe.

The diagnostic also supports `--amplification-token-budget 8`: amplify only
during the first eight generated tokens, then continue through the same model
without injection. Zero preserves continuous amplification. Suppression is
unchanged. This is an experimental autoregressive control schedule, not
Steerling's masked-diffusion inference procedure; compare it against continuous
injection using the same checkpoint, prompts, strength, and decoding.

The subsequent four-concept diagnostic at `tau=0.05` found phrase loops under
continuous injection in Music and Home cooking. An eight-token amplification
window removed these loops in the same four neutral-prompt examples; semantic
direction success remained 2/4. GPT-OSS mean fluency was 2.5/4 in both audits,
so this does not establish an overall judged fluency improvement. The broader
recipe now uses inference `tau=0.05` with an eight-token window; its training
target stays `0.1`. This setting was selected on a small diagnostic, not the
final benchmark. The added local CE repair remains experimental because it
did not reliably recover all four controls.
