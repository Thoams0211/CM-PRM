#!/bin/bash

# Greedy Search BoN Runner

set -euo pipefail

# Get the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Default configuration parameters (can be overridden by command line)
# Datasets: AMC23, Math, AIME24, College math, Minerva-Math, Olympiadbench
NUM_ITERATIONS=30
CHECK_PATH="buffer/generator_math.json"

# Generator parameters
GEN_MODEL_PATH="# POLICY_MODEL_PATH"
JSONL_PATH="dataset/MATH/test.jsonl"
BUFFER_PATH="buffer/generator_math.json"
DISCRIMINATOR_PATH="buffer/discriminator_math.json"
GEN_TENSOR_PARALLEL_SIZE=2
GEN_BATCH_SIZE=32
NUM_CANDIDATES=8

# Discriminator parameters
DISC_MODEL_PATH="# PRM_MODEL_PATH"
DISC_TENSOR_PARALLEL_SIZE=2
DISC_BATCH_SIZE=32
PRM_N=8
TEMPERATURE=0.95
TOP_P=0.95
MAX_TOKENS=2048
DTYPE="bfloat16"
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN="8192"  # Empty means no limit

show_help() {
  cat <<'EOF'
Usage: bash run.sh [options]

Common options:
  --num_iterations N
  --check_path PATH
  --jsonl_path PATH
  --buffer_path PATH
  --discriminator_path PATH

Generator options:
  --gen_model_path PATH
  --gen_tensor_parallel_size N
  --gen_batch_size N
  --num_candidates N
  --init_sample_count N|""

Discriminator options:
  --disc_model_path PATH
  --disc_tensor_parallel_size N
  --disc_batch_size N
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
    --num_iterations) NUM_ITERATIONS="$2"; shift 2 ;;
    --check_path) CHECK_PATH="$2"; shift 2 ;;
    --jsonl_path) JSONL_PATH="$2"; shift 2 ;;
    --buffer_path) BUFFER_PATH="$2"; shift 2 ;;
    --discriminator_path) DISCRIMINATOR_PATH="$2"; shift 2 ;;

    --gen_model_path) GEN_MODEL_PATH="$2"; shift 2 ;;
    --gen_tensor_parallel_size) GEN_TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --gen_batch_size) GEN_BATCH_SIZE="$2"; shift 2 ;;
    --num_candidates) NUM_CANDIDATES="$2"; shift 2 ;;
    --init_sample_count) INIT_SAMPLE_COUNT="$2"; shift 2 ;;

    --disc_model_path) DISC_MODEL_PATH="$2"; shift 2 ;;
    --disc_tensor_parallel_size) DISC_TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --disc_batch_size) DISC_BATCH_SIZE="$2"; shift 2 ;;
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

echo "============================================================"
echo "Greedy Search Runner"
echo "Will run ${NUM_ITERATIONS} iterations"
echo "Each iteration: Generator -> Discriminator"
echo "Answer check file: ${CHECK_PATH}"
echo "============================================================"
echo ""

# Start loop
for i in $(seq 1 ${NUM_ITERATIONS}); do
  echo ""
  echo "============================================================"
  echo "Start ${i}/${NUM_ITERATIONS} iteration"
  echo "============================================================"
  echo ""

  # Step 1: Run Generator
  echo ">>> Step 1: Run Generator..."
  GEN_CMD=(
    bash generator.sh
    --model_path "${GEN_MODEL_PATH}"
    --jsonl_path "${JSONL_PATH}"
    --buffer_path "${BUFFER_PATH}"
    --discriminator_path "${DISCRIMINATOR_PATH}"
    --tensor_parallel_size "${GEN_TENSOR_PARALLEL_SIZE}"
    --batch_size "${GEN_BATCH_SIZE}"
    --num_candidates "${NUM_CANDIDATES}"
  )
  if [ -n "${INIT_SAMPLE_COUNT}" ]; then
    GEN_CMD+=(--init_sample_count "${INIT_SAMPLE_COUNT}")
  fi
  "${GEN_CMD[@]}"

  echo ""
  echo ">>> Generator completed"
  echo ""

  # Step 2: Run Discriminator
  echo ">>> Step 2: Run Discriminator..."
  DISC_CMD=(
    bash discriminator_cmcl.sh
    --model_path "${DISC_MODEL_PATH}"
    --discriminator_path "${DISCRIMINATOR_PATH}"
    --generator_path "${BUFFER_PATH}"
    --jsonl_path "${JSONL_PATH}"
    --tensor_parallel_size "${DISC_TENSOR_PARALLEL_SIZE}"
    --batch_size "${DISC_BATCH_SIZE}"
    --prm_n "${PRM_N}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --max_tokens "${MAX_TOKENS}"
    --dtype "${DTYPE}"
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  )
  if [ -n "${MAX_MODEL_LEN}" ]; then
    DISC_CMD+=(--max_model_len "${MAX_MODEL_LEN}")
  fi
  "${DISC_CMD[@]}"

  echo ""
  echo ">>> Discriminator completed"
  echo ""

  # Step 3: Check if all samples have answer
  echo ">>> Step 3: Run Answer Check..."
  CHECK_EXIT=0
  if python answer_check.py --path "${CHECK_PATH}"; then
    CHECK_EXIT=0
  else
    CHECK_EXIT=$?
  fi

  if [ ${CHECK_EXIT} -eq 0 ]; then
    echo ""
    echo "All samples have answer, terminate loop early."
    echo "============================================================"
    echo "Terminate loop early after ${i}/${NUM_ITERATIONS} iteration"
    echo "============================================================"
    exit 0
  elif [ ${CHECK_EXIT} -eq 2 ]; then
    echo "Error: Answer Check execution exception, terminate loop"
    exit 1
  fi

  echo ""
  echo "============================================================"
  echo "${i}/${NUM_ITERATIONS} iteration completed"
  echo "============================================================"
done

echo ""
echo "============================================================"
echo "All ${NUM_ITERATIONS} iterations completed!"
echo "============================================================"
