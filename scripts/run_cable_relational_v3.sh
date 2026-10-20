#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/hjh/Desktop/NeuroSymbolic_FSAD}"
DATA_ROOT="${DATA_ROOT:-/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTec}"
INP_ROOT="${INP_ROOT:-/home/hjh/Desktop/H2VLR_compare/INP-Former}"
CHECKPOINT="${CHECKPOINT:-${INP_ROOT}/saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth}"
LLM_MODEL="${LLM_MODEL:-/home/hjh/AgentModels/Qwen3-VL-8B-Thinking}"
LLM_MODEL_HASH="${LLM_MODEL_HASH:-c41fe4d3d3532dab697894040481b39a83ab7e53edc294d7c8c3561a91054b37}"
OUTPUT_DIR="${OUTPUT_DIR:-results_v3/mvtec/cable/k4_qwen_relational_strict}"
CACHE_DIR="${CACHE_DIR:-results_v2_smoke/mvtec/cable/k4}"
PROPOSAL_JSON="${PROPOSAL_JSON:-}"

proposal_args=()
if [[ -n "$PROPOSAL_JSON" ]]; then
  proposal_args=(--proposal-json "$PROPOSAL_JSON")
fi

cd "$PROJECT_ROOT"

PYTHONDONTWRITEBYTECODE=1 conda run -n zeta \
  python run_relational_neurosymbolic_fsad.py \
  --dataset mvtec \
  --data-root "$DATA_ROOT" \
  --class-name cable \
  --shot 4 \
  --support-strategy inp_official \
  --support-seed 1 \
  --checkpoint "$CHECKPOINT" \
  --inpformer-root "$INP_ROOT" \
  --device cuda:0 \
  --llm-device cuda:1 \
  --llm-model "$LLM_MODEL" \
  --llm-model-hash "$LLM_MODEL_HASH" \
  --prediction-cache-dir "$CACHE_DIR" \
  --allowed-rule-families identity,count,position,geometry,containment,core \
  "${proposal_args[@]}" \
  --output-dir "$OUTPUT_DIR"
