#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/hjh/Desktop/NeuroSymbolic_FSAD"
INP_ROOT="/home/hjh/Desktop/H2VLR_compare/INP-Former"
DATA_ROOT="/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTec"
CHECKPOINT="${INP_ROOT}/saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth"
LLM="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking"

cd "${PROJECT_ROOT}"
for CLASS_NAME in transistor screw cable; do
  case "${CLASS_NAME}" in
    transistor|screw) RUN_NAME="k4_official_support" ;;
    cable) RUN_NAME="k4_official_support_full" ;;
  esac
  PYTHONDONTWRITEBYTECODE=1 conda run -n zeta python run_neurosymbolic_fsad.py \
    --dataset mvtec \
    --data-root "${DATA_ROOT}" \
    --class-name "${CLASS_NAME}" \
    --shot 4 \
    --support-strategy inp_official \
    --support-seed 1 \
    --checkpoint "${CHECKPOINT}" \
    --inpformer-root "${INP_ROOT}" \
    --device cuda:0 \
    --batch-size 16 \
    --llm-model "${LLM}" \
    --llm-device cuda:1 \
    --max-new-tokens 1024 \
    --output-dir "${PROJECT_ROOT}/results/mvtec/${CLASS_NAME}/${RUN_NAME}"
done

PYTHONDONTWRITEBYTECODE=1 conda run -n zeta python run_neurosymbolic_fsad.py \
  --dataset beltad \
  --data-root /home/hjh/Desktop/0Datasets/anomaly/industrial/BeltAD \
  --class-name belt \
  --shot 4 \
  --support-strategy inp_official \
  --support-seed 1 \
  --checkpoint "${INP_ROOT}/saved_results/INP-Former-Industrial-beltad-4/model.pth" \
  --inpformer-root "${INP_ROOT}" \
  --device cuda:0 \
  --batch-size 16 \
  --llm-model "${LLM}" \
  --llm-device cuda:1 \
  --max-new-tokens 1024 \
  --output-dir "${PROJECT_ROOT}/results/beltad/belt/k4_official_support_full"
