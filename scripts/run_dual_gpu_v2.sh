#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   MODE=qwen    bash scripts/run_dual_gpu_v2.sh
#   MODE=no_llm  bash scripts/run_dual_gpu_v2.sh
#
# INP-Former inference and the learned fusion head use GPU 0. Qwen3-VL uses
# GPU 1 only in qwen mode. Fingerprinted INP predictions are reused from the
# v2 smoke runs, so changing the rule-proposal mode does not repeat inference.

PROJECT_ROOT="/home/hjh/Desktop/NeuroSymbolic_FSAD"
INP_ROOT="/home/hjh/Desktop/H2VLR_compare/INP-Former"
MVTEC_ROOT="/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTec"
BELTAD_ROOT="/home/hjh/Desktop/0Datasets/anomaly/industrial/BeltAD"
LLM_MODEL="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking"

MVTEC_CHECKPOINT="${INP_ROOT}/saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth"
BELTAD_CHECKPOINT="${INP_ROOT}/saved_results/INP-Former-Industrial-beltad-4/model.pth"

MODE="${MODE:-qwen}"
CONDA_ENV="${CONDA_ENV:-zeta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results_v2}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/results_v2_smoke}"
BATCH_SIZE="${BATCH_SIZE:-16}"
HEAD_EPOCHS="${HEAD_EPOCHS:-350}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

case "${MODE}" in
  no_llm|qwen) ;;
  *)
    printf 'ERROR: MODE must be no_llm or qwen; received %s\n' "${MODE}" >&2
    exit 2
    ;;
esac

for required_path in \
  "${PROJECT_ROOT}/run_reliable_neurosymbolic_fsad.py" \
  "${INP_ROOT}" \
  "${MVTEC_ROOT}" \
  "${BELTAD_ROOT}" \
  "${MVTEC_CHECKPOINT}" \
  "${BELTAD_CHECKPOINT}"; do
  if [[ ! -e "${required_path}" ]]; then
    printf 'ERROR: required path does not exist: %s\n' "${required_path}" >&2
    exit 1
  fi
done

if [[ "${MODE}" == "qwen" && ! -d "${LLM_MODEL}" ]]; then
  printf 'ERROR: local Qwen3-VL model does not exist: %s\n' "${LLM_MODEL}" >&2
  exit 1
fi

run_class() {
  local dataset="$1"
  local data_root="$2"
  local class_name="$3"
  local checkpoint="$4"
  local cache_dir="${CACHE_ROOT}/${dataset}/${class_name}/k4"
  local output_dir="${OUTPUT_ROOT}/${dataset}/${class_name}/k4_${MODE}"
  local -a mode_args=()

  if [[ "${MODE}" == "no_llm" ]]; then
    mode_args+=(--no-llm)
  else
    mode_args+=(
      --llm-model "${LLM_MODEL}"
      --llm-device cuda:1
      --max-new-tokens "${MAX_NEW_TOKENS}"
    )
  fi

  printf '\n[v2] dataset=%s class=%s mode=%s\n' \
    "${dataset}" "${class_name}" "${MODE}"
  printf '[v2] cache=%s\n[v2] output=%s\n' "${cache_dir}" "${output_dir}"

  PYTHONDONTWRITEBYTECODE=1 conda run -n "${CONDA_ENV}" \
    python run_reliable_neurosymbolic_fsad.py \
    --dataset "${dataset}" \
    --data-root "${data_root}" \
    --class-name "${class_name}" \
    --shot 4 \
    --support-strategy inp_official \
    --support-seed 1 \
    --seed 668 \
    --calibration-seed 20261013 \
    --checkpoint "${checkpoint}" \
    --inpformer-root "${INP_ROOT}" \
    --device cuda:0 \
    --head-device cuda:0 \
    --batch-size "${BATCH_SIZE}" \
    --head-epochs "${HEAD_EPOCHS}" \
    --minimum-rule-auc 0.58 \
    --maximum-rule-fpr 0.25 \
    --minimum-fusion-gain 0.005 \
    --prediction-cache-dir "${cache_dir}" \
    --output-dir "${output_dir}" \
    "${mode_args[@]}"
}

cd "${PROJECT_ROOT}"

run_class mvtec "${MVTEC_ROOT}" transistor "${MVTEC_CHECKPOINT}"
run_class mvtec "${MVTEC_ROOT}" screw "${MVTEC_CHECKPOINT}"
run_class mvtec "${MVTEC_ROOT}" cable "${MVTEC_CHECKPOINT}"
run_class beltad "${BELTAD_ROOT}" belt "${BELTAD_CHECKPOINT}"

printf '\n[v2] completed all four classes; results are under %s\n' "${OUTPUT_ROOT}"
