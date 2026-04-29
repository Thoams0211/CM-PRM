#!/bin/bash

# Greedy Search Discriminator Runner

set -euo pipefail

# Default parameters (can be overridden by command line)
MODEL_PATH="# PRM_MODEL_PATH"
DISCRIMINATOR_PATH="buffer/discriminator_math.json"
GENERATOR_PATH="buffer/generator_math.json"
JSONL_PATH="dataset/MATH/test.jsonl"
TENSOR_PARALLEL_SIZE=2
BATCH_SIZE=32
PRM_N=8
TEMPERATURE=0.95
TOP_P=0.95
MAX_TOKENS=2048
DTYPE="bfloat16"
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN="8192"  # Empty means no limit

show_help() {
  cat <<'EOF'
Usage: bash discriminator_cmcl.sh [options]

Options:
  --model_path PATH
  --discriminator_path PATH
  --generator_path PATH
  --jsonl_path PATH
  --tensor_parallel_size N
  --batch_size N
  --prm_n N
  --temperature FLOAT
  --top_p FLOAT
  --max_tokens N
  --dtype TYPE
  --gpu_memory_utilization FLOAT
  --max_model_len N|""
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --discriminator_path) DISCRIMINATOR_PATH="$2"; shift 2 ;;
    --generator_path) GENERATOR_PATH="$2"; shift 2 ;;
    --jsonl_path) JSONL_PATH="$2"; shift 2 ;;
    --tensor_parallel_size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --prm_n) PRM_N="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top_p) TOP_P="$2"; shift 2 ;;
    --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
    -h|--help) show_help; exit 0 ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Run command
echo "============================================================"
echo "Starting Greedy Search Discriminator (CMCL PRM)..."
echo "Model Path: ${MODEL_PATH}"
echo "Discriminator Path: ${DISCRIMINATOR_PATH}"
echo "Generator Path: ${GENERATOR_PATH}"
echo "JSONL Path: ${JSONL_PATH}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "PRM N: ${PRM_N}"
echo "Temperature: ${TEMPERATURE}"
echo "Top P: ${TOP_P}"
echo "Max Tokens: ${MAX_TOKENS}"
echo "DType: ${DTYPE}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Max Model Len: ${MAX_MODEL_LEN:-None}"
echo "============================================================"
echo ""

CMD=(
  python discriminator_cmcl.py
  --model_path "${MODEL_PATH}"
  --discriminator_path "${DISCRIMINATOR_PATH}"
  --generator_path "${GENERATOR_PATH}"
  --jsonl_path "${JSONL_PATH}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --prm_n "${PRM_N}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --max_tokens "${MAX_TOKENS}"
  --dtype "${DTYPE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
)

# Only pass MAX_MODEL_LEN if set
if [ -n "${MAX_MODEL_LEN}" ]; then
  CMD+=(--max_model_len "${MAX_MODEL_LEN}")
fi

"${CMD[@]}"
