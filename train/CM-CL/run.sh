#!/bin/bash
#
# Script to run CMCL training with DeepSpeed.
# Configured with categorized variables for easier parameter tuning.

set -e

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1


# Environment & Device Configuration
CUDA_DEVICES="# AVAILABLE_CUDA_DEVICES"
NUM_DEVICES="# NUMBER_OF_DEVICES"

# Paths & Files
SCRIPT_PATH="scripts/run_cmcl.py"
TRAIN_FILE="# DATASET_DIRECTORY"
MODEL_PATH="# SFT_MODEL_DIRECTORY"
REF_MODEL_PATH="# SFT_MODEL_DIRECTORY"
OUTPUT_BASE_DIR="output"
DS_CONFIG="ds_config_zero2.json"


# Training Hyperparameters
LEARNING_RATE=1e-6
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
NUM_TRAIN_EPOCHS=1
SEED=42
BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_DEVICES * GRADIENT_ACCUMULATION_STEPS))


# CMCL Hyperparameters
BETA=0.1
CMCL_ALPHA=1.0
CMCL_BETA=30.0
CMCL_GAMMA=60.0
CMCL_TAU=1.0
LOSS_TYPE="cmcl" # 'cmcl', 'dpo' or 'ipo'
FORCE_REF_MODEL="true"


# Sequence Length & Tokenization
MAX_LENGTH=2048
MAX_PROMPT_LENGTH=800
TOKENIZE_NUM_PROC=20


# LoRA Hyperparameters
USE_LORA="true"
LORA_R=32
LORA_ALPHA=128
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"


# Logging, Saving & Resuming
LOG_STEPS=5
SAVE_STEPS=50
SAVE_TOTAL_LIMIT=5
LOG_WITH="tensorboard"

# Set to "true" to resume, and specify the checkpoint step below
RESUME_TRAINING="false"
RESUME_STEP=""


# Hardware & Performance Flags

BF16="true"
USE_FLASH_ATTENTION_2="true"
GRADIENT_CHECKPOINTING="true"


# Directory Setup & Argument Construction
# Construct the experiment directory name based on key hyperparameters
EXPERIMENT_NAME="lr${LEARNING_RATE}_bs${BATCH_SIZE}_wd${WEIGHT_DECAY}_cmcl-alpha${CMCL_ALPHA}_cmcl-beta${CMCL_BETA}_cmcl-gamma${CMCL_GAMMA}_cmcl-tau${CMCL_TAU}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"
LOGGING_DIR="${OUTPUT_DIR}/logs"

# Handle resume logic conditionally
RESUME_ARG=""
if [ "${RESUME_TRAINING}" == "true" ]; then
    RESUME_ARG="--resume_from_checkpoint ${OUTPUT_DIR}/checkpoint-${RESUME_STEP}"
fi

# Handle gradient checkpointing flag conditionally
GC_ARG=""
if [ "${GRADIENT_CHECKPOINTING}" == "true" ]; then
    GC_ARG="--gradient_checkpointing"
fi


# Execution

deepspeed --include "localhost:${CUDA_DEVICES}" "${SCRIPT_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --model_name_or_path "${MODEL_PATH}" \
  --ref_model_name_or_path "${REF_MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  ${RESUME_ARG} \
  --max_length "${MAX_LENGTH}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --cmcl_alpha "${CMCL_ALPHA}" \
  --cmcl_beta "${CMCL_BETA}" \
  --cmcl_gamma "${CMCL_GAMMA}" \
  --cmcl_tau "${CMCL_TAU}" \
  --force_ref_model "${FORCE_REF_MODEL}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --bf16 "${BF16}" \
  --learning_rate "${LEARNING_RATE}" \
  --loss_type "${LOSS_TYPE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --logging_steps "${LOG_STEPS}" \
  --logging_dir "${LOGGING_DIR}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --seed "${SEED}" \
  --log_with "${LOG_WITH}" \
  --use_lora "${USE_LORA}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --deepspeed_config "${DS_CONFIG}" \
  --tokenize_num_proc "${TOKENIZE_NUM_PROC}" \
  --use_flash_attention_2 "${USE_FLASH_ATTENTION_2}" \
  ${GC_ARG}