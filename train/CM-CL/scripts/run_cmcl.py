# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, TrainerCallback
from peft import LoraConfig, PeftModel, TaskType
from trl import DPOConfig

from cmcl_config import CMCLConfig
from cmcl_trainer import CMCLTrainer


def create_peft_config(
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
) -> Optional[LoraConfig]:

    """
    Configuration of LoRA for CMCL training.

    Args:
        use_lora: Whether to use LoRA for CMCL training.
        lora_r: The rank of LoRA.
        lora_alpha: The alpha of LoRA.
        lora_dropout: The dropout of LoRA.
        lora_target_modules: The target modules of LoRA.

    Returns:
        Optional[LoraConfig]: The LoRA configuration.
    """

    if not use_lora:
        return None
    target_modules = [m.strip() for m in lora_target_modules.split(",") if m.strip()]
    if not target_modules:
        raise ValueError("LoRA target modules cannot be empty.")
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def load_filtered_parquet_dir(
    folder_path: str,
    max_samples: Optional[int] = None,
) -> Dataset:
    """Read parquet shards from the directory output by program 1.

    Expected columns: prompt, chosen, rejected

    Args:
        folder_path: The path to the directory containing the parquet shards.
        max_samples: The maximum number of samples to load.

    Returns:
        Dataset: The dataset containing the filtered parquet shards.

    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Data directory does not exist or is not a folder: {folder_path}")

    data_files = str(folder / "*.parquet")
    ds = load_dataset("parquet", data_files=data_files, split="train")

    required = {"prompt", "chosen", "rejected"}
    if not required.issubset(set(ds.column_names)):
        raise ValueError(f"Filtered parquet missing required columns: {required}, got={ds.column_names}")

    # Only keep columns needed for trainer
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("prompt", "chosen", "rejected")])
    
    return ds



def main() -> None:
    parser = argparse.ArgumentParser(description="CMCL Finetuning with TRL (Trainer tokenizes; dataset already filtered)")

    # Dataset configuration
    parser.add_argument("--train_file", type=str, required=True, help="Filtered training dataset directory containing parquet files")
    parser.add_argument("--eval_file", type=str, default=None, help="Filtered evaluation dataset directory containing parquet files (optional)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum training samples")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Maximum evaluation samples")

    # Length parameters for truncation and tokenization
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length of prompt + response (tokens)")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Maximum length of prompt (tokens)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to tokenize")
    parser.add_argument("--tokenize_num_proc", type=int, default=None, help="[unused] kept for compatibility")

    # Model configuration
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Policy model (SFT model) path")
    parser.add_argument("--ref_model_name_or_path", type=str, default=None, help="Reference model path, default same as policy model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--force_ref_model", type=bool, default=False, help="Force use reference model")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to custom DeepSpeed config JSON")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")

    # CMCL hyperparameters
    parser.add_argument("--cmcl_alpha", type=float, default=1.0, help="CMCL alpha hyperparameter")
    parser.add_argument("--cmcl_beta", type=float, default=0.1, help="CMCL beta hyperparameter")
    parser.add_argument("--cmcl_gamma", type=float, default=0.1, help="CMCL gamma hyperparameter")
    parser.add_argument("--cmcl_tau", type=float, default=1.0, help="CMCL tau hyperparameter")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta hyperparameter, only used when loss_type=dpo")

    # Training parameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--loss_type", type=str, default="cmcl", help="Loss type, default ipo")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Number of warmup steps")
    parser.add_argument("--logging_dir", type=str, default=None, help="Log directory")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", type=bool, default=False, help="Enable BF16 training")
    parser.add_argument("--fp16", type=bool, default=False, help="Enable FP16 training")
    parser.add_argument("--log_with", type=str, default="tensorboard", help="Logging backend, comma separated")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--use_flash_attention_2", type=bool, default=False, help="Enable Flash Attention 2")

    # LoRA configuration
    parser.add_argument("--use_lora", type=bool, default=False, help="Enable LoRA finetuning")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj",
        help="Modules to inject LoRA, comma separated",
    )

    args = parser.parse_args()

    # Directory preparation
    output_dir = Path(args.output_dir)
    log_dir = Path(args.logging_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    set_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"

    # dtype selection
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    # Flash Attention 2 configuration
    attn_implementation = None
    if args.use_flash_attention_2:
        import flash_attn
        attn_implementation = "flash_attention_2"
        print("Flash Attention 2 enabled")

    # Load policy model
    print("Loading policy model...")
    model_kwargs = {"torch_dtype": torch_dtype}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # Loading ref model
    ref_model = None
    if args.use_lora or args.force_ref_model:
        print("Using LoRA training or force_ref_model, set ref_model=None to save memory.")
        ref_model = None
    else:
        ref_model_path = args.ref_model_name_or_path or args.model_name_or_path
        print(f"Loading ref model: {ref_model_path}")
        ref_model_kwargs = {"torch_dtype": torch_dtype}
        if attn_implementation:
            ref_model_kwargs["attn_implementation"] = attn_implementation
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, **ref_model_kwargs)
        if getattr(ref_model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            ref_model.config.pad_token_id = tokenizer.pad_token_id
        ref_model.config.use_cache = False

    # Process LoRA
    peft_config: Optional[LoraConfig] = None
    if args.use_lora:
        peft_config = create_peft_config(
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
        )
        if peft_config is not None:
            print(f"Using LoRA, r={peft_config.r}, alpha={peft_config.lora_alpha}, dropout={peft_config.lora_dropout}")

    # Logging configuration
    report_to = None
    if args.log_with:
        report_to = [backend.strip() for backend in args.log_with.split(",") if backend.strip()]

    # DeepSpeed configuration
    deepspeed_config = args.deepspeed_config if args.deepspeed_config else None
    if deepspeed_config:
        print(f"Using user provided DeepSpeed config: {deepspeed_config}")

    # Load data
    print("Loading filtered datasets (prompt/chosen/rejected)...")
    train_dataset = load_filtered_parquet_dir(args.train_file, max_samples=args.max_train_samples)

    print(f"Loss type: {args.loss_type}")

    # Training configuration
    if args.loss_type == "cmcl":
        training_args = CMCLConfig(
            output_dir=str(output_dir),
            cmcl_alpha=args.cmcl_alpha,
            cmcl_beta=args.cmcl_beta,
            cmcl_gamma=args.cmcl_gamma,
            cmcl_tau=args.cmcl_tau,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            logging_dir=args.logging_dir,
            loss_type=args.loss_type,                  # Loss type, default cmcl
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=args.save_total_limit,
            report_to=report_to,
            bf16=args.bf16,
            fp16=args.fp16,
            seed=args.seed,
            dataset_num_proc=args.tokenize_num_proc,
            gradient_checkpointing=args.gradient_checkpointing,
            remove_unused_columns=False,
            deepspeed=deepspeed_config,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,  # Maximum length of prompt (tokens)
            max_grad_norm=0.14,
        )
    elif args.loss_type == "dpo":
        training_args = DPOConfig(
            output_dir=str(output_dir),
            beta=args.dpo_beta,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            logging_dir=args.logging_dir,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=args.save_total_limit,
            report_to=report_to,
            bf16=args.bf16,
            fp16=args.fp16,
            seed=args.seed,
            dataset_num_proc=args.tokenize_num_proc,
            gradient_checkpointing=args.gradient_checkpointing,
            remove_unused_columns=False,
            deepspeed=deepspeed_config,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,  # Maximum length of prompt (tokens)
            max_grad_norm=0.14,
        )
    elif args.loss_type == "ipo":
        training_args = DPOConfig(
            output_dir=str(output_dir),
            beta=args.dpo_beta,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            logging_dir=args.logging_dir,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=args.save_total_limit,
            report_to=report_to,
            bf16=args.bf16,
            fp16=args.fp16,
            seed=args.seed,
            dataset_num_proc=args.tokenize_num_proc,
            gradient_checkpointing=args.gradient_checkpointing,
            remove_unused_columns=False,
            deepspeed=deepspeed_config,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,  # Maximum length of prompt (tokens)
            max_grad_norm=0.14,
        )
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}")

    # Starting CMCL training
    print("Starting CMCL training (Trainer will tokenize datasets internally)...")
    trainer = CMCLTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
