#!/usr/bin/env bash
#
# Script to run vLLM process benchmarking with LoRA support.
# Simplified version allowing direct customization of output paths.

set -e


# Environment & Device Configuration
export CUDA_VISIBLE_DEVICES="# CUDA_VISIBLE_DEVICES"
TENSOR_PARALLEL_SIZE="# TENSOR_PARALLEL_SIZE"


# Evaluation Hyperparameters
NUM_SAMPLES=8
MAX_TOKENS=2048
TEMPERATURE=0.95


# Paths & Files
SCRIPT_PATH="inference.py"
DATA_FILE="# INPUT_FILE"    # e.g. "olympiadbench_part.json"
MODEL_PATH="# MODEL_DIRECTORY or BACKBONE_MODEL_DIRECTORY"


# User-defined output file
OUTPUT_FILE="output/results.json"


# LoRA Configuration
USE_LORA="true"
LORA_ADAPTER_PATH="# LORA_ADAPTER_PATH"
LORA_NAME="# LORA_NAME"
LORA_INT_ID="# LORA_INT_ID"


# Execution Logic
# Construct LoRA arguments conditionally
LORA_ARGS=""
if [ "${USE_LORA}" == "true" ]; then
    LORA_ARGS="--lora-adapter-path ${LORA_ADAPTER_PATH} --lora-name ${LORA_NAME} --lora-int-id ${LORA_INT_ID}"
fi

# Automatically create the output directory if it does not exist
OUTPUT_DIR=$(dirname "${OUTPUT_FILE}")
mkdir -p "${OUTPUT_DIR}"

echo "Starting evaluation. Results will be saved to: ${OUTPUT_FILE}"

python3 "${SCRIPT_PATH}" \
  --data-file "${DATA_FILE}" \
  --model "${MODEL_PATH}" \
  --output "${OUTPUT_FILE}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
  ${LORA_ARGS}