#!/usr/bin/env bash
set -euo pipefail

# End-to-end, resumable controlled experiment.  INP-Former stays frozen; both
# GPUs are used for independent local-Qwen support-image rule proposals.
PROJECT_ROOT="${PROJECT_ROOT:-/home/hjh/Desktop/NeuroSymbolic_FSAD}"
DATA_ROOT="${DATA_ROOT:-/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTec}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results_v7_logic}"
BASELINE_CACHE="${BASELINE_CACHE:-${PROJECT_ROOT}/results_v6_multidataset/cache/mvtec/transistor/k4}"
OLD_WEIGHTED_METRICS="${OLD_WEIGHTED_METRICS:-${PROJECT_ROOT}/results_v6_multidataset/class_specific/mvtec/transistor/k4/metrics_v5_class_specific.json}"
LLM_MODEL="${LLM_MODEL:-/home/hjh/AgentModels/Qwen3-VL-8B-Thinking}"
H2VLR_DESTINATION="${H2VLR_DESTINATION:-/home/hjh/Desktop/H2VLR/1号_分类_MVTEC数据集/mvtec/k_4/NeuroSymbolicFSAD_v7_transistor_k30}"
CONDA_ENV="${CONDA_ENV:-zeta}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RULE_SUPPORT_COUNT="${RULE_SUPPORT_COUNT:-30}"
NESTED_COUNTS="${NESTED_COUNTS:-4 8 16 30}"
SUPPORT_SEED="${SUPPORT_SEED:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT_ROOT}/logs"
if [[ "${GPU0}" == "${GPU1}" ]]; then
  echo "GPU0 and GPU1 must name different physical GPUs." >&2
  exit 2
fi
exec 9>"${RESULT_ROOT}/logs/transistor_v7_launcher.lock"
if ! flock -n 9; then
  echo "Another transistor v7 launcher is already running." >&2
  exit 2
fi

COMMON=(
  python scripts/generate_logic_rule_candidates.py
  --dataset mvtec
  --data-root "${DATA_ROOT}"
  --class-name transistor
  --output-root "${RESULT_ROOT}"
  --shot "${RULE_SUPPORT_COUNT}"
  --support-seed "${SUPPORT_SEED}"
  --llm-model "${LLM_MODEL}"
  --llm-device cuda:0
  --num-shards 2
)

CUDA_VISIBLE_DEVICES="${GPU0}" PYTHONDONTWRITEBYTECODE=1 \
  conda run -n "${CONDA_ENV}" --no-capture-output \
  "${COMMON[@]}" --shard-index 0 \
  >"${RESULT_ROOT}/logs/transistor_k${RULE_SUPPORT_COUNT}_qwen_gpu${GPU0}.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES="${GPU1}" PYTHONDONTWRITEBYTECODE=1 \
  conda run -n "${CONDA_ENV}" --no-capture-output \
  "${COMMON[@]}" --shard-index 1 \
  >"${RESULT_ROOT}/logs/transistor_k${RULE_SUPPORT_COUNT}_qwen_gpu${GPU1}.log" 2>&1 &
PID1=$!

cleanup() {
  kill "${PID0}" "${PID1}" 2>/dev/null || true
}
trap cleanup INT TERM
set +e
wait "${PID0}"
STATUS0=$?
wait "${PID1}"
STATUS1=$?
set -e
trap - INT TERM
if [[ "${STATUS0}" -ne 0 || "${STATUS1}" -ne 0 ]]; then
  echo "Qwen shard failed: gpu${GPU0}=${STATUS0}, gpu${GPU1}=${STATUS1}. See ${RESULT_ROOT}/logs/." >&2
  exit 1
fi

# Rebuild one complete aggregate after both processes finish.  This does not
# load Qwen and catches a missing/invalid per-support result before evaluation.
PYTHONDONTWRITEBYTECODE=1 conda run -n "${CONDA_ENV}" --no-capture-output \
  python scripts/generate_logic_rule_candidates.py \
  --dataset mvtec --data-root "${DATA_ROOT}" --class-name transistor \
  --output-root "${RESULT_ROOT}" --shot "${RULE_SUPPORT_COUNT}" \
  --support-seed "${SUPPORT_SEED}" --num-shards 1 --shard-index 0 \
  --aggregate-only

SOURCE_RUN_DIR="${RESULT_ROOT}/mvtec/transistor/k${RULE_SUPPORT_COUNT}/qwen_logic_candidates_v7"
PYTHONDONTWRITEBYTECODE=1 conda run -n "${CONDA_ENV}" --no-capture-output \
  python scripts/build_nested_candidate_banks_v7.py \
  --source-run-dir "${SOURCE_RUN_DIR}" --output-root "${RESULT_ROOT}" \
  --counts ${NESTED_COUNTS}

SUMMARY_ARGS=()
for COUNT in ${NESTED_COUNTS}; do
  PREFIX_DIR="${RESULT_ROOT}/mvtec/transistor/nested_k${COUNT}/qwen_logic_candidates_v7"
  RUN_DIR="${RESULT_ROOT}/mvtec/transistor/nested_k${COUNT}_executable_qwen"
  MAX_VISUALIZATIONS=4
  if [[ "${COUNT}" -eq "${RULE_SUPPORT_COUNT}" ]]; then
    MAX_VISUALIZATIONS=100
  fi
  PYTHONDONTWRITEBYTECODE=1 conda run -n "${CONDA_ENV}" --no-capture-output \
    python run_executable_logic_fsad.py \
    --dataset mvtec --data-root "${DATA_ROOT}" --class-name transistor \
    --baseline-shot 4 --rule-support-count "${COUNT}" \
    --support-seed "${SUPPORT_SEED}" \
    --candidate-bank "${PREFIX_DIR}/candidate_rule_bank.json" \
    --rule-support-manifest "${PREFIX_DIR}/support_manifest.json" \
    --prediction-cache-dir "${BASELINE_CACHE}" \
    --old-weighted-metrics "${OLD_WEIGHTED_METRICS}" \
    --output-dir "${RUN_DIR}" --max-visualizations "${MAX_VISUALIZATIONS}"
  SUMMARY_ARGS+=(--run-dir "${RUN_DIR}")
done

PYTHONDONTWRITEBYTECODE=1 conda run -n "${CONDA_ENV}" --no-capture-output \
  python scripts/summarize_executable_logic_v7.py \
  "${SUMMARY_ARGS[@]}" \
  --output-json "${RESULT_ROOT}/mvtec/transistor/nested_support_summary_v7.json" \
  --output-markdown "${RESULT_ROOT}/mvtec/transistor/nested_support_summary_v7.md"

FINAL_RUN_DIR="${RESULT_ROOT}/mvtec/transistor/nested_k${RULE_SUPPORT_COUNT}_executable_qwen"
PYTHONDONTWRITEBYTECODE=1 conda run -n "${CONDA_ENV}" --no-capture-output \
  python scripts/validate_executable_logic_v7.py \
  --run-dir "${FINAL_RUN_DIR}" \
  --output "${FINAL_RUN_DIR}/validation_v7.json" \
  --repeats 2000 --pixel-histogram-bins 4096 --seed 668

PYTHONDONTWRITEBYTECODE=1 conda run -n "${CONDA_ENV}" --no-capture-output \
  python scripts/export_executable_logic_h2vlr.py \
  --run-dir "${FINAL_RUN_DIR}" --destination "${H2VLR_DESTINATION}" \
  --seed "${SUPPORT_SEED}"

echo "Experiment complete: ${FINAL_RUN_DIR}"
echo "H2VLR export: ${H2VLR_DESTINATION}"
