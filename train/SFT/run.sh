#!/usr/bin/env bash
#
# Script to run SFT training with DeepSpeed.
# Simplified version allowing direct customization of paths and parameters.

set -e

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1


# Environment & Device Configuration
CUDA_DEVICES="0,1"

# Paths & Directories
SCRIPT_PATH="train.py"
MODEL_PATH="# PATH_OF_BACKBONE_MODEL"
DATA_DIR="dataset"
DS_CONFIG="ds_config.json"

# User-defined output and logging directories
OUTPUT_DIR="output"
LOG_DIR="${OUTPUT_DIR}/logs"


# Training Hyperparameters
LEARNING_RATE=1e-4
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=32
NUM_TRAIN_EPOCHS=1
WARMUP_RATIO=0.03
MAX_LENGTH=2048
TOKENIZE_NUM_PROC=10
BF16="true"


# LoRA Configuration
USE_LORA="true"
LORA_RANK=32
LORA_ALPHA=128
LORA_DROPOUT=0.05
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


# Logging, Saving & Resuming
SAVE_STEPS=200
LOGGING_STEPS=5
SAVE_TOTAL_LIMIT=5
LOG_WITH="tensorboard"

# Set to "true" to resume, and specify the full checkpoint path below
RESUME_TRAINING="true"
RESUME_CHECKPOINT="# PATH_OF_CHECKPOINT"


# Execution Logic
# Handle resume logic conditionally
RESUME_ARG=""
if [ "${RESUME_TRAINING}" == "true" ]; then
    RESUME_ARG="--resume_from_checkpoint ${RESUME_CHECKPOINT}"
fi

# Automatically create the output and log directories if they do not exist
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "Starting training. Output will be saved to: ${OUTPUT_DIR}"

deepspeed --include "localhost:${CUDA_DEVICES}" "${SCRIPT_PATH}" \
  --model_name_or_path "${MODEL_PATH}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --logdir "${LOG_DIR}" \
  ${RESUME_ARG} \
  --tokenize_num_proc "${TOKENIZE_NUM_PROC}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --bf16 "${BF16}" \
  --use_lora "${USE_LORA}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules "${TARGET_MODULES}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --max_length "${MAX_LENGTH}" \
  --deepspeed "${DS_CONFIG}" \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --log_with "${LOG_WITH}"