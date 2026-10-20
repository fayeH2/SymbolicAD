#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/hjh/Desktop/NeuroSymbolic_FSAD
cd "$PROJECT"

conda run -n zeta --no-capture-output \
  python scripts/check_cuda_runtime.py \
  --gpus 0,1 \
  --llm-model /home/hjh/AgentModels/Qwen3-VL-8B-Thinking

exec conda run -n zeta --no-capture-output \
  python scripts/run_multidataset_all_gpus.py \
  --stage all \
  --scope all \
  --include-experimental \
  --gpus 0,1 \
  --shot 4 \
  --support-seed 1 \
  --baseline-workers-per-gpu 1 \
  --proposal-workers-per-gpu 1 \
  --eval-workers-per-gpu 2 \
  --class-specific-workers-per-gpu 2 \
  --cpu-threads-per-job 4 \
  --data-workers 4 \
  --batch-size 16 \
  --baseline-epochs 200 \
  --output-root /home/hjh/Desktop/NeuroSymbolic_FSAD/results_v6_multidataset \
  "$@"
