#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/hjh/Desktop/NeuroSymbolic_FSAD}"
DATA_ROOT="${DATA_ROOT:-/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTec}"
BASE_LIBRARY="${BASE_LIBRARY:-${PROJECT_ROOT}/results_v7_logic/mvtec/transistor/nested_k30_executable_qwen/normality_rules_v7_executable_logic.json}"
BASE_CACHE="${BASE_CACHE:-${PROJECT_ROOT}/results_v6_multidataset/cache/mvtec/transistor/k4}"
FIXED_METRICS="${FIXED_METRICS:-${PROJECT_ROOT}/results_v7_logic/mvtec/transistor/nested_k30_executable_qwen/metrics_v7_executable_logic.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results_v8_symbolicdet/mvtec/transistor}"
LLM_MODEL="${LLM_MODEL:-/home/hjh/AgentModels/Qwen3-VL-8B-Thinking}"
H2VLR_DESTINATION="${H2VLR_DESTINATION:-/home/hjh/Desktop/H2VLR/1号_分类_MVTEC数据集/mvtec/k_4/NeuroSymbolicDetEvolution_v8_transistor_k30_all4}"
CONDA_ENV="${CONDA_ENV:-zeta}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

cd "${PROJECT_ROOT}"
test -f "${BASE_LIBRARY}" || {
  echo "Missing v7 Qwen candidate/grounder library: ${BASE_LIBRARY}" >&2
  echo "First run: bash scripts/run_transistor_executable_logic_2gpu.sh" >&2
  exit 2
}
test -f "${BASE_CACHE}/cache/test_inp_v4.npz" || {
  echo "Missing frozen INP cache: ${BASE_CACHE}/cache/test_inp_v4.npz" >&2
  exit 2
}
if [[ "${GPU0}" == "${GPU1}" ]]; then
  echo "GPU0 and GPU1 must be different." >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}/logs"
exec 9>"${OUTPUT_ROOT}/logs/v8_evolution.lock"
if ! flock -n 9; then
  echo "Another v8 evolution launcher is running." >&2
  exit 2
fi

COMMON=(
  python run_symbolicdet_evolution_fsad.py
  --data-root "${DATA_ROOT}"
  --base-rule-library "${BASE_LIBRARY}"
  --prediction-cache-dir "${BASE_CACHE}"
  --fixed-v7-metrics "${FIXED_METRICS}"
  --llm-model "${LLM_MODEL}"
  --llm-device cuda:0
  --population-size 64
  --generations 40
  --crossover-rate 0.5
  --mutation-rate 0.3
  --llm-feedback-interval 5
  --early-stop-patience 12
  --max-visualizations 100
)

CUDA_VISIBLE_DEVICES="${GPU0}" PYTHONDONTWRITEBYTECODE=1 \
  conda run -n "${CONDA_ENV}" --no-capture-output \
  "${COMMON[@]}" --output-dir "${OUTPUT_ROOT}/k30_evolution_qwen_v2" \
  >"${OUTPUT_ROOT}/logs/stable2_gpu${GPU0}.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES="${GPU1}" PYTHONDONTWRITEBYTECODE=1 \
  conda run -n "${CONDA_ENV}" --no-capture-output \
  "${COMMON[@]}" --evolve-all-groundable \
  --output-dir "${OUTPUT_ROOT}/k30_evolution_all_groundable_qwen" \
  >"${OUTPUT_ROOT}/logs/all4_gpu${GPU1}.log" 2>&1 &
PID1=$!

cleanup() {
  kill "${PID0}" "${PID1}" 2>/dev/null || true
}
trap cleanup INT TERM
set +e
wait "${PID0}"; STATUS0=$?
wait "${PID1}"; STATUS1=$?
set -e
trap - INT TERM
if [[ "${STATUS0}" -ne 0 || "${STATUS1}" -ne 0 ]]; then
  echo "v8 search failed: stable2=${STATUS0}, all4=${STATUS1}" >&2
  exit 1
fi

for RUN in k30_evolution_qwen_v2 k30_evolution_all_groundable_qwen; do
  PYTHONDONTWRITEBYTECODE=1 conda run -n "${CONDA_ENV}" --no-capture-output \
    python scripts/validate_symbolicdet_evolution_v8.py \
    --run-dir "${OUTPUT_ROOT}/${RUN}" \
    --output "${OUTPUT_ROOT}/${RUN}/validation_v8.json" \
    --repeats 2000 --pixel-histogram-bins 4096 --seed 668
done

PYTHONDONTWRITEBYTECODE=1 conda run -n "${CONDA_ENV}" --no-capture-output \
  python scripts/export_symbolicdet_evolution_h2vlr.py \
  --run-dir "${OUTPUT_ROOT}/k30_evolution_all_groundable_qwen" \
  --destination "${H2VLR_DESTINATION}" --seed 1

echo "v8 stable-pool result: ${OUTPUT_ROOT}/k30_evolution_qwen_v2"
echo "v8 all-groundable result: ${OUTPUT_ROOT}/k30_evolution_all_groundable_qwen"
echo "H2VLR export: ${H2VLR_DESTINATION}"
