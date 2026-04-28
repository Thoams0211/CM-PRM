#!/bin/bash
set -euo pipefail

# GPU settings (modify as needed).
export CUDA_VISIBLE_DEVICES="GPU_IDX"

# Base configuration (modify as needed).
MODEL_PATH="# MODEL_PATH or BACKBONE_MODEL_PATH"

TENSOR_PARALLEL_SIZE="# TENSOR_PARALLEL_SIZE"
BATCH_SIZE="# BATCH_SIZE"
MAX_TOKENS="# MAX_TOKENS"
TEMPERATURE="# TEMPERATURE"

# Sampling configuration.
N_SAMPLES="# N_SAMPLES"

# LoRA configuration (optional, leave empty if not needed).
LORA_ADAPTER_PATH="# LORA_ADAPTER_PATH"
LORA_NAME="# LORA_NAME"
LORA_INT_ID="# LORA_INT_ID"

# Define the list of classifications to iterate over.
declare -a CLASSIFICATIONS=(
  "redundency"
  "circular"
  "counterfactual"
  "step_contradiction"
  "domain_inconsistency"
  "confidence"
  "missing_condition"
  "deception"
  "multi_solutions"
)

# Iterate through each classification and run the evaluation.
for CLASSIFICATION in "${CLASSIFICATIONS[@]}"; do
  echo "================================================================================"
  echo "Starting evaluation for: ${CLASSIFICATION}"
  echo "================================================================================"

  # Construct dynamic paths based on the current classification.
  DATA_PATH="dataset/${CLASSIFICATION}.jsonl"
  OUTPUT_PATH="output/${CLASSIFICATION}_results.json"

  # Define the command array for the current iteration.
  CMD=(
    python prmbench.py
    --model_path "${MODEL_PATH}"
    --data_path "${DATA_PATH}"
    --output_path "${OUTPUT_PATH}"
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
    --batch_size "${BATCH_SIZE}"
    --max_tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --n "${N_SAMPLES}"
    --save_raw_output "true"
  )

  # Append LoRA arguments if the path is provided.
  if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
    CMD+=(
      --lora_adapter_path "${LORA_ADAPTER_PATH}"
      --lora_name "${LORA_NAME}"
      --lora_int_id "${LORA_INT_ID}"
    )
  fi

  echo "Running inference command:"
  printf ' %q' "${CMD[@]}"
  echo
  echo "--------------------------------------------------------------------------------"

  # Execute the inference command.
  "${CMD[@]}"
  
  echo "Finished evaluation for: ${CLASSIFICATION}"
  echo
done

echo "================================================================================"
echo "All inferences complete! Don't forget to run 'python analyse.py' to generate your evaluation report."