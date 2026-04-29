#!/usr/bin/env python3
"""PRMBench inference script (single-stage SFT-style prompting).

This script utilizes vLLM for high-throughput inference and supports
multiple sampling to evaluate process reward models. It only generates
and saves the results; evaluation is handled separately.
"""

import argparse
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


DEFAULT_INSTRUCTION_TEMPLATE = """You are an excellent math teacher. Please verify the correctness of the Now Step.

You first need to analyze the Now Step and the Previous Steps and then summarize based on your analysis.
Analysis:
You need to analyze the following aspects.
**Previous Steps Analysis**: You need to analyze the Previous Steps step by step. For each step, you need to first explain what the current step is doing, then you try to find any error in the current step.
**Now Step Analysis**: You first need to explain what the Now Step is doing, and then point out which part of the Question it is trying to solve or which part of the information it states.
**Data Source Analysis**: First you need to find out what data are used in the Now Step, and then you need to determine whether the source of the data is reasonable and correct. When you judge whether the source of a data is reasonable and correct, you need to specify the specific source of this data: such as which part of the question, or which content of the previous step; and then determine the source and current use is consistent, the Now Step is used correctly.
**Consistency Analysis**: You need to check that the Now Step is consistent with the contents of the Previous Steps, and then you need to check that all the information inside the Now Step is consistent.
**Calculation Analysis**: If the Now Step involves any calculations, such as addition, subtraction, multiplication, division, equations, modulo operations, etc., you will first need to perform a check on the calculation, such as a reverse operation, to see if the calculation was done correctly, and then analyze the results of your check to see if there was an error in the calculation.
Conclusion:
Please verify the correctness of the Now Step based on your analysis, if there is any error in the Now Step then the Now Step is wrong and vice versa the Now Step is correct. At the end of the Conclusion, when you give your final answer, write it in the form "Verification: Is the step correct (Yes/No)? X", where X is either Yes or No.

Question: {question}
Previous Steps: {previous_steps}
Now Step: {now_step}
Please carefully analyze the correctness of the Now Step.
Reply:"""

VERIFICATION_PATTERN = re.compile(
    r"Verification:\s*Is\s+the\s+step\s+correct\s+\(Yes/No\)\?\s*(Yes|No)",
    re.IGNORECASE,
)


def str2bool(v: Any) -> bool:
    """Converts a string or boolean to a boolean value."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for the PRMBench generation."""
    parser = argparse.ArgumentParser(description="PRMBench generation")
    parser.add_argument("--model_path", type=str, required=True, help="Model path for vLLM")
    parser.add_argument("--data_path", type=str, required=True, help="Input data path (JSON/JSONL file or directory)")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSON path")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--n", type=int, default=8, help="Number of output sequences to return for each prompt.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max generated tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--instruction_template", type=str, default=DEFAULT_INSTRUCTION_TEMPLATE, help="SFT-style instruction template")
    parser.add_argument("--save_raw_output", type=str2bool, default=False, help="Whether to save raw model outputs")
    parser.add_argument("--lora_adapter_path", type=str, default=None, help="Optional LoRA adapter path")
    parser.add_argument("--lora_name", type=str, default="cmcl_lora", help="LoRA adapter name")
    parser.add_argument("--lora_int_id", type=int, default=1, help="LoRA adapter integer id")
    # Kept top_p and top_k in argparse to prevent bash script errors, but they are unused in SamplingParams
    parser.add_argument("--top_p", type=float, default=1.0, help="Unused (relies on vLLM default)")
    parser.add_argument("--top_k", type=int, default=-1, help="Unused (relies on vLLM default)")
    return parser.parse_args()


def _load_one_file(file_path: str) -> List[Dict[str, Any]]:
    """Loads JSON or JSONL data from a single file."""
    p = Path(file_path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"JSON file is not a list: {file_path}")
        return data

    records: List[Dict[str, Any]] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSONL line {idx} in {file_path}: {exc}") from exc
    return records


def load_json_data(data_path: str) -> List[Dict[str, Any]]:
    """Loads JSON/JSONL data from a file or directory."""
    if os.path.isfile(data_path):
        return _load_one_file(data_path)

    if os.path.isdir(data_path):
        all_data: List[Dict[str, Any]] = []
        files = sorted(
            [
                os.path.join(data_path, x)
                for x in os.listdir(data_path)
                if x.endswith(".json") or x.endswith(".jsonl")
            ]
        )
        for fp in files:
            all_data.extend(_load_one_file(fp))
        return all_data

    raise ValueError(f"Data path does not exist: {data_path}")


def split_samples(original_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Splits original process samples into step-level samples for verification."""
    split_samples_list: List[Dict[str, Any]] = []
    for sample in original_samples:
        original_id = sample.get("id", sample.get("idx", "unknown"))
        modified_question = sample.get("modified_question", sample.get("problem", sample.get("question", "")))
        modified_process = sample.get("modified_process", sample.get("steps", []))
        error_steps = sample.get("error_steps", [])
        classification = sample.get("classification", sample.get("label", "unclassified"))

        if not modified_process:
            print(f"Warning: Sample {original_id} has empty modified_process, skipping.")
            continue

        previous_steps: List[str] = []
        for step_idx, current_step in enumerate(modified_process, start=1):
            new_sample = {
                "original_id": original_id,
                "step_id": step_idx,
                "question": modified_question,
                "previous_steps": previous_steps.copy(),
                "current_step": current_step,
                "error_steps": error_steps,
                "classification": classification,
                "is_error": step_idx in error_steps,
            }
            # Carry over any other metadata
            for key, value in sample.items():
                if key not in {"id", "idx", "modified_question", "modified_process", "problem", "question", "steps", "error_steps", "classification", "label"}:
                    new_sample[key] = deepcopy(value)
                    
            split_samples_list.append(new_sample)
            previous_steps.append(str(current_step))
    return split_samples_list


def build_instruction(sample: Dict[str, Any], instruction_template: str) -> str:
    """Builds the instruction text for a given sample based on the template."""
    prev_lines = [f"Step {i + 1}: {step}" for i, step in enumerate(sample.get("previous_steps", []))]
    previous_steps_text = "\n".join(prev_lines) if prev_lines else "None"
    now_step_text = f"Step {sample.get('step_id', 1)}: {sample.get('current_step', '')}"
    return instruction_template.format(
        question=sample.get("question", ""),
        previous_steps=previous_steps_text,
        now_step=now_step_text,
    )


def build_prompt(instruction: str) -> str:
    """Builds the final prompt string for the LLM."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def extract_verification_label(text: str) -> Optional[str]:
    """Extracts the Yes/No verification label from the generated text."""
    if not text:
        return None
    matches = VERIFICATION_PATTERN.findall(text)
    if not matches:
        return None
    label = matches[-1].strip().lower()
    return "Yes" if label == "yes" else "No"


def label_to_reward(label: Optional[str]) -> float:
    """Converts a textual label into a numerical reward score."""
    if label == "No":
        return 0.0
    if label == "Yes":
        return 1.0
    return 0.5


def init_vllm_model(model_path: str, tensor_parallel_size: int = 1, lora_adapter_path: Optional[str] = None) -> Any:
    """Initializes the vLLM engine."""
    from vllm import LLM
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
    llm_kwargs: Dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
    }

    if lora_adapter_path:
        adapter_dir = Path(lora_adapter_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"LoRA adapter path not found: {lora_adapter_path}")
        max_lora_rank = 32
        adapter_cfg = adapter_dir / "adapter_config.json"
        if adapter_cfg.exists():
            try:
                with open(adapter_cfg, "r", encoding="utf-8") as f:
                    max_lora_rank = json.load(f).get("r", 32)
            except Exception:
                max_lora_rank = 32
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max_lora_rank

    return LLM(**llm_kwargs)


def process_single_batch(
    batch_samples: List[Dict[str, Any]],
    llm: Any,
    sampling_params: Any,
    args: argparse.Namespace,
    lora_request: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Processes a single batch of samples using vLLM."""
    prompts: List[str] = []
    for sample in batch_samples:
        instruction = build_instruction(sample, args.instruction_template)
        prompts.append(build_prompt(instruction))

    results: List[Dict[str, Any]] = []

    generations = llm.generate(
        prompts=prompts, 
        sampling_params=sampling_params, 
        use_tqdm=False, 
        lora_request=lora_request
    )
    
    for sample, prompt, output in zip(batch_samples, prompts, generations):
        rewards: List[float] = []
        labels: List[Optional[str]] = []
        raw_texts: List[str] = []
        
        for seq_output in output.outputs:
            generated_text = seq_output.text
            full_text = prompt + generated_text
            label = extract_verification_label(full_text)
            
            rewards.append(label_to_reward(label))
            labels.append(label)
            if args.save_raw_output:
                raw_texts.append(full_text)

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.5
        
        result = {
            "reward": mean_reward,
            "all_rewards": rewards,
            "verification_labels": labels,
            "parse_failed": any(l is None for l in labels),
        }
        if args.save_raw_output:
            result["outputs"] = raw_texts
            
        results.append(result)
    return results


def main() -> None:
    """Main execution function."""
    args = parse_args()

    print(f"Loading data from: {args.data_path}")
    original_samples = load_json_data(args.data_path)
    print(f"Loaded {len(original_samples)} original samples.")

    print("Splitting samples into per-step examples...")
    split_samples_list = split_samples(original_samples)
    print(f"Generated {len(split_samples_list)} step-level samples.")

    print(f"Initializing model: {args.model_path}")
    llm = init_vllm_model(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        lora_adapter_path=args.lora_adapter_path,
    )

    lora_request = None
    if args.lora_adapter_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(
            lora_name=args.lora_name,
            lora_int_id=args.lora_int_id,
            lora_path=args.lora_adapter_path,
        )
        print(f"Using LoRA adapter: {args.lora_adapter_path}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_results: List[Dict[str, Any]] = []

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    print("Starting batch inference...")
    for i in tqdm(range(0, len(split_samples_list), args.batch_size), desc="Batch inference"):
        batch_samples = split_samples_list[i : i + args.batch_size]
        batch_results = process_single_batch(
            batch_samples=batch_samples,
            llm=llm,
            sampling_params=sampling_params,
            args=args,
            lora_request=lora_request,
        )

        for sample, result in zip(batch_samples, batch_results):
            final_sample = {
                "original_id": sample.get("original_id", "unknown"),
                "step_id": sample.get("step_id", 0),
                "is_error": sample.get("is_error", False),
                "classification": sample.get("classification", "unclassified"),
                "reward": result.get("reward", 0.5), 
                "all_rewards": result.get("all_rewards", []),
                "verification_labels": result.get("verification_labels", []),
                "parse_failed": result.get("parse_failed", True),
            }
            if "error" in result:
                final_sample["error"] = result["error"]
            if args.save_raw_output and "outputs" in result:
                final_sample["outputs"] = result["outputs"]
            final_results.append(final_sample)

        # Incrementally save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"Finished generation. Output saved to: {output_path}")
    print("Please run analyse.py to evaluate the metrics.")


if __name__ == "__main__":
    main()