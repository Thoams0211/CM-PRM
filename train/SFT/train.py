#!/usr/bin/env python3
import argparse
from collections import Counter
import glob
import logging
import os
from pathlib import Path
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datasets
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_parquet_dataset(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise ValueError(f"No .parquet files found in {data_dir}")
    ds = datasets.load_dataset("parquet", data_files=files)
    # datasets returns a dict split if multiple parquet files; unify to dataset
    if isinstance(ds, dict):
        # merge all splits into a single dataset
        datasets_list = [v for v in ds.values()]
        merged = datasets.concatenate_datasets(datasets_list)
        return merged
    return ds


def make_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def preprocess_examples(examples, tokenizer, max_length):
    # 批量化实现：一次对一批文本调用 tokenizer（更快）
    instructions = examples["instruction"]
    outputs = examples["output"]
    # 构建 prompt 和 full 文本列表
    prompts = [make_prompt(instr) for instr in instructions]
    eos = tokenizer.eos_token
    full_texts = [p + out + eos for p, out in zip(prompts, outputs)]

    # 使用 tokenizer 对整批进行 tokenization（fast tokenizer 在 Rust 中更快）
    tokenized_full = tokenizer(
        full_texts, truncation=True, max_length=max_length, padding=False
    )
    tokenized_prompt = tokenizer(
        prompts, truncation=True, max_length=max_length, padding=False
    )

    input_ids_list = tokenized_full["input_ids"]
    attention_mask_list = tokenized_full.get(
        "attention_mask", [[1] * len(ids) for ids in input_ids_list]
    )

    labels_list = []
    for full_ids, prompt_ids in zip(input_ids_list, tokenized_prompt["input_ids"]):
        labels = full_ids.copy()
        prompt_len = len(prompt_ids)
        # 将 prompt 部分设为 -100（不参与 loss）
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def data_collator(tokenizer, features: list):
    # simple collator for causal LM with labels already prepared (with -100)
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_mask = [
        torch.tensor(f["attention_mask"], dtype=torch.long) for f in features
    ]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of training samples (use first N samples)")
    parser.add_argument("--tokenize_batch_size", type=int, default=2000, help="Batch size for dataset.map tokenization")
    parser.add_argument("--tokenize_num_proc", type=int, default=1, help="Number of processes for dataset.map tokenization (num_proc)")
    parser.add_argument("--print_samples", type=int, default=0, help="Print N decoded tokenized examples for debugging")
    parser.add_argument("--deepspeed", type=str, default=None, help="Deepspeed config json")
    parser.add_argument("--attn_implementation", type=str, default=None, help="Attention implementation name (e.g. 'flash') — sets model_kwargs['attn_implementation']")
    parser.add_argument("--use_lora", type=str, default="false", help="Enable LoRA (true/false)")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--bf16", type=str, default="false", help="Use bf16 (true/false)")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep")
    parser.add_argument("--log_with", type=str, default="tensorboard", help="Reporting backend for Trainer (e.g., 'tensorboard', 'wandb', 'none')")
    parser.add_argument("--logdir", type=str, default=None, help="Directory for tensorboard logs. If not specified, will use output_dir/logs")
    parser.add_argument("--max_training_samples", type=int, default=None, help="Limit the number of training samples (use first N samples)")
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume training from, or 'True' to resume from latest checkpoint in output_dir")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for learning rate scheduler")
    args = parser.parse_args()

    # helper to parse true/false strings (compatible with one-arg-per-line style)
    def str2bool(s):
        if isinstance(s, bool):
            return s
        if s is None:
            return False
        return str(s).lower() in ("1", "true", "yes")

    args.bf16 = str2bool(args.bf16)
    args.use_lora = str2bool(args.use_lora)

    # Attention implementation control (reference dpo_finetuning style)
    model_kwargs = {}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, revision=args.model_revision
    )
    if tokenizer.pad_token_id is None:
        # Use eos_token as pad_token (don't add new token, just reuse existing one)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
    
    # Ensure model config is aligned with tokenizer
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Optional LoRA (PEFT)
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Prepare dataset
    ds = load_parquet_dataset(args.data_dir)
    if args.max_samples is not None:
        ds = ds.select(range(args.max_samples))
        logger.info(f"Limited training dataset to {args.max_samples} samples")
    # ensure columns instruction & output exist
    if "instruction" not in ds.column_names or "output" not in ds.column_names:
        raise ValueError("Dataset must contain 'instruction' and 'output' columns")

    map_kwargs = {
        "batched": True,
        "batch_size": args.tokenize_batch_size,
        "remove_columns": ds.column_names,
    }
    # only set num_proc if >1 (num_proc may be unsupported on some platforms)
    if args.tokenize_num_proc and int(args.tokenize_num_proc) > 1:
        map_kwargs["num_proc"] = int(args.tokenize_num_proc)

    tokenized = ds.map(
        lambda examples: preprocess_examples(examples, tokenizer, args.max_length),
        **map_kwargs,
    )
    
    # 如果需要，打印若干经 tokenization 后的样本（将 token ids 解码回可读文本）
    n_print = 5
    for idx in range(n_print):
        item = tokenized[idx]
        input_ids = item["input_ids"]
        labels = item["labels"]
        # 找到第一个标签不为 -100 的位置（即 response 起始位置）
        start_pos = 0
        for i_label, lab in enumerate(labels):
            if lab != -100:
                start_pos = i_label
                break
        full_text = tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        response_text = tokenizer.decode(input_ids[start_pos:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print("==== Decoded sample %d ====" % idx)
        print("Full (prompt + response):")
        print(full_text)
        print("Response (decoded from first non -100 label):")
        print(response_text)
    

    # TrainingArguments with Deepspeed and bf16
    report_to = None if str(args.log_with).lower() in ("none", "", "null") else args.log_with
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to=report_to,
        logging_dir=args.logdir,  # 设置 tensorboard 日志目录
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized,
        data_collator=lambda features: data_collator(tokenizer, features),
    )

    # 处理断点续训
    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "true":
            # 如果设置为 "true"，尝试从 output_dir 中找到最新的 checkpoint
            checkpoint_dirs = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
            if checkpoint_dirs:
                # 按 checkpoint 编号排序，选择最新的
                checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
                resume_from_checkpoint = checkpoint_dirs[-1]
                logger.info(f"自动找到最新 checkpoint: {resume_from_checkpoint}")
            else:
                logger.warning(f"未在 {args.output_dir} 中找到 checkpoint，将从新开始训练")
        else:
            # 直接使用提供的路径
            resume_from_checkpoint = args.resume_from_checkpoint
            if os.path.exists(resume_from_checkpoint):
                logger.info(f"从 checkpoint 恢复训练: {resume_from_checkpoint}")
            else:
                logger.warning(f"Checkpoint 路径不存在: {resume_from_checkpoint}，将从新开始训练")
                resume_from_checkpoint = None

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model and tokenizer
    logger.info("Training completed. Saving final model and tokenizer...")
    if isinstance(model, PeftModel):
        # For LoRA, save the adapter and tokenizer
        model.save_pretrained(args.output_dir)
        logger.info(f"LoRA adapter saved to {args.output_dir}")
    else:
        # For full model, save the entire model
        model.save_pretrained(args.output_dir)
        logger.info(f"Full model saved to {args.output_dir}")
    
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()

