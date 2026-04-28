#!/bin/bash

# Greedy Search Generator

set -euo pipefail

# Default parameters (can be overridden by command line)
MODEL_PATH="# POLICY_MODEL_PATH"
JSONL_PATH="dataset/MATH/test.jsonl"
BUFFER_PATH="buffer/generator_math.json"
DISCRIMINATOR_PATH="buffer/discriminator_math.json"
TENSOR_PARALLEL_SIZE=2
BATCH_SIZE=32
NUM_CANDIDATES=8

show_help() {
  cat <<'EOF'
Usage: bash generator.sh [options]

Options:
  --model_path PATH
  --jsonl_path PATH
  --buffer_path PATH
  --discriminator_path PATH
  --tensor_parallel_size N
  --batch_size N
  --num_candidates N
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --jsonl_path) JSONL_PATH="$2"; shift 2 ;;
    --buffer_path) BUFFER_PATH="$2"; shift 2 ;;
    --discriminator_path) DISCRIMINATOR_PATH="$2"; shift 2 ;;
    --tensor_parallel_size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --num_candidates) NUM_CANDIDATES="$2"; shift 2 ;;
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
echo "Starting BoN Generator..."
echo "Model Path: ${MODEL_PATH}"
echo "JSONL Path: ${JSONL_PATH}"
echo "Buffer Path: ${BUFFER_PATH}"
echo "Discriminator Path: ${DISCRIMINATOR_PATH}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Num Candidates: ${NUM_CANDIDATES}"
echo "============================================================"
echo ""

CMD=(
  python generator.py
  --model_path "${MODEL_PATH}"
  --buffer_path "${BUFFER_PATH}"
  --discriminator_path "${DISCRIMINATOR_PATH}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --num_candidates "${NUM_CANDIDATES}"
  --jsonl_path "${JSONL_PATH}"
)

"${CMD[@]}"
