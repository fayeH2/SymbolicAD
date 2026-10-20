#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hjh/Desktop/NeuroSymbolic_FSAD"
DATA="/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTec"
INP_ROOT="/home/hjh/Desktop/H2VLR_compare/INP-Former"
CHECKPOINT="${INP_ROOT}/saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth"
OUTPUT="${ROOT}/results_v12_ovd_symbolic_residual/mvtec"

cd "${ROOT}"
mkdir -p "${OUTPUT}/logs"

run_class() {
  local class_name="$1"
  local gpu="$2"
  local proposal="$3"
  local cache_root="$4"
  PYTHONDONTWRITEBYTECODE=1 conda run -n zeta python run_ovd_symbolic_fsad.py \
    --dataset mvtec \
    --data-root "${DATA}" \
    --class-name "${class_name}" \
    --shot 4 \
    --support-seed 1 \
    --proposal-json "${proposal}" \
    --checkpoint "${CHECKPOINT}" \
    --inpformer-root "${INP_ROOT}" \
    --inp-device "cuda:${gpu}" \
    --ovd-device "cuda:${gpu}" \
    --batch-size 16 \
    --prediction-cache-dir "${cache_root}" \
    --max-visualizations 20 \
    --output-dir "${OUTPUT}/${class_name}/k4"
}

run_class \
  transistor 0 \
  "${ROOT}/results_v11_fresh_proposals/mvtec/transistor/k4/qwen_generic_relational_proposal.json" \
  "${ROOT}/results_v11_generic_orthogonal/mvtec/transistor/k4" \
  >"${OUTPUT}/logs/transistor.log" 2>&1 &
pid_transistor=$!

run_class \
  cable 1 \
  "${ROOT}/results_v12_fresh_proposals/mvtec/cable/k4/qwen_generic_relational_proposal.json" \
  "${OUTPUT}/cable/k4" \
  >"${OUTPUT}/logs/cable.log" 2>&1 &
pid_cable=$!

status=0
wait "${pid_transistor}" || status=1
wait "${pid_cable}" || status=1
exit "${status}"
