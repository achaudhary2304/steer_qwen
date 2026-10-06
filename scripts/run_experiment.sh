#!/usr/bin/env bash
# Run from the repository root inside the activated Conda environment.
set -euo pipefail
profile="${1:-debug}"
case "$profile" in
  debug)
    config=configs/runnable/qwen35-08b-debug.json
    dataset=data/atlas-debug
    known=64
    sample=20000
    scan=100000
    support=10
    token_budget=0
    max_length=128
    ;;
  100m)
    config=configs/runnable/qwen35-08b-atlas-100m.json
    dataset=data/atlas-100m
    known=1024
    sample=1500000
    scan=3000000
    support=50
    token_budget=100000000
    max_length=256
    ;;
  *) echo 'Usage: bash scripts/run_experiment.sh [debug|100m]' >&2; exit 2 ;;
esac
extra_args=()
if [ "$profile" = "100m" ]; then
  extra_args=(--merged-steering)
fi
mkdir -p "runs/$profile"
if [ ! -f "$dataset/manifest.json" ]; then
  python -u -m concept_retrofit.cli prepare-atlas --out "$dataset" \
    --known "$known" --sample-rows "$sample" --scan-rows "$scan" \
    --min-support "$support" --max-length "$max_length" \
    --training-token-budget "$token_budget" \
    2>&1 | tee -a "runs/$profile/prepare.log"
fi
python -u -m concept_retrofit.cli run-all --data "$dataset" --config "$config" \
  --out "runs/$profile/experiment" --resume "${extra_args[@]}" \
  2>&1 | tee -a "runs/$profile/pipeline.log"
