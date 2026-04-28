#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

PROMPT_TEMPLATE = """
### Instruction:
You are an excellent math teacher. Please verify the correctness of the Now Step.

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
\n\n
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("process_bench")


def load_json_file(path: str) -> List[Dict[str, Any]]:
    """Loads a single benchmark file.

    Supports two formats:
    - JSON array (the whole file is a list).
    - JSONL (each line is a JSON object).

    Args:
        path: Path to the input file.

    Returns:
        A list of sample dictionaries.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is invalid JSON or cannot be parsed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = p.read_text(encoding="utf-8")
    text_stripped = text.lstrip()

    if text_stripped.startswith("["):
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError(f"File {path} parsed as JSON but is not a list.")

    results: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except Exception as e:
            raise ValueError(f"Failed to parse line {i} in {path} as JSON: {e}")
    return results


def expand_steps_into_examples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expands each dataset sample into multiple per-step examples.

    For each original sample containing a list of reasoning steps, produces
    one example per step.

    Args:
        samples: List of raw dataset samples loaded from the file.

    Returns:
        A flattened list of per-step example dictionaries.
    """
    examples: List[Dict[str, Any]] = []
    for samp in samples:
        question = samp.get("question") or samp.get("problem") or samp.get("text") or samp.get("prompt") or ""
        steps = samp.get("steps") or samp.get("reasoning") or samp.get("explanation") or []

        if isinstance(steps, str):
            steps = [steps]
        if not isinstance(steps, list):
            continue

        answer = samp.get("final_answer") or samp.get("answer") or samp.get("label") or ""
        explanation = samp.get("explanation") or ""
        base_id = samp.get("id") or samp.get("idx") or ""

        for i, now in enumerate(steps):
            previous = ["Step " + str(j + 1) + ": " + str(p) for j, p in enumerate(steps[:i])]
            prev = "\n".join(previous) if i > 0 else ""
            ex_id = f"{base_id}-step{i}"

            examples.append(
                {
                    "id": ex_id,
                    "orig_id": base_id,
                    "step_index": i,
                    "question": str(question),
                    "previous_steps": prev,
                    "now_step": "Step " + str(i + 1) + ": " + str(now),
                    "answer": str(answer),
                    "explanation": str(explanation),
                    "_orig_sample": samp,
                }
            )
    return examples


def build_prompt(sample: Dict[str, Any]) -> str:
    """Builds the final prompt from the template and sample fields.

    Args:
        sample: A dictionary containing 'question', 'previous_steps', 
            and 'now_step'.

    Returns:
        The fully constructed prompt string.
    """
    parts = [
        PROMPT_TEMPLATE.strip(),
        "",
        f"Question: {sample.get('question', '')}",
        "",
        f"Previous Steps: {sample.get('previous_steps', '')}",
        "",
        f"Now Step: {sample.get('now_step', '')}",
        "",
        "Please carefully analyze the correctness of the Now Step.\nReply: ",
        "",
        "\n### Response:\n",
    ]
    return "\n".join(parts)


def init_vllm(model: str, tensor_parallel_size: int = 1, lora_adapter_path: Optional[str] = None) -> LLM:
    """Initializes the vLLM instance.

    Args:
        model: Model path or identifier (vLLM/Transformers compatible).
        tensor_parallel_size: Number of GPUs to use for tensor parallelism.
        lora_adapter_path: Optional path to a LoRA adapter directory.

    Returns:
        The initialized LLM instance.

    Raises:
        FileNotFoundError: If the specified LoRA adapter path does not exist.
    """
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

    llm_kwargs: Dict[str, Any] = {
        "model": model,
        "tensor_parallel_size": tensor_parallel_size,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.90,
        "swap_space": 0,
        "disable_custom_all_reduce": True,
        "enable_prefix_caching": True,
    }

    if lora_adapter_path:
        adapter_path = Path(lora_adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter path not found: {lora_adapter_path}")

        adapter_config_path = adapter_path / "adapter_config.json"
        max_lora_rank = 32  # Default fallback
        if adapter_config_path.exists():
            try:
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    adapter_config = json.load(f)
                    max_lora_rank = adapter_config.get("r", 32)
                    logger.info("Read max_lora_rank=%s from adapter_config.json", max_lora_rank)
            except Exception as e:
                logger.warning("Failed to read adapter_config.json: %s; using default max_lora_rank=32", e)

        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max_lora_rank
        logger.info("LoRA enabled with adapter: %s", lora_adapter_path)
        logger.info("max_lora_rank: %s", max_lora_rank)

    logger.info("Initializing vLLM model: %s", model)
    logger.info("Using %d GPU(s) for tensor parallelism", tensor_parallel_size)

    return LLM(**llm_kwargs)


def extract_score_from_output(text: str) -> Optional[float]:
    """Parses model output text to extract a numeric correctness score.

    Args:
        text: The generated text from the model.

    Returns:
        1.0 if the step is verified as 'Yes', 0.0 if 'No', or None if it 
        cannot be parsed.
    """
    if not text:
        return None

    m = re.search(
        r"Verification:\s*Is the step correct\s*\(Yes/No\)\?\s*(Yes|No)\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return 1.0 if m.group(1).lower() == "yes" else 0.0

    return None


def main(argv: Optional[List[str]] = None) -> None:
    """Main execution function for the evaluation script."""
    parser = argparse.ArgumentParser(description="Run batched inference using vLLM.")
    parser.add_argument("--data-file", type=str, required=True, help="Input JSON/JSONL file path.")
    parser.add_argument("--model", type=str, required=True, help="Model name (vLLM / Transformers recognizable).")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path.")
    # Removed --batch_size as vLLM handles batching internally based on max_tokens and context window.
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=-1, help="Top-k sampling (default: -1, disabled).")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of generations per prompt.")
    parser.add_argument("--lora-adapter-path", type=str, default=None, help="Path to LoRA adapter checkpoint.")
    parser.add_argument("--lora-name", type=str, default="default_lora", help="LoRA adapter name.")
    parser.add_argument("--lora-int-id", type=int, default=1, help="LoRA adapter integer ID.")

    args = parser.parse_args(argv)

    data_path = Path(args.data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"data_file not found: {args.data_file}")
    
    logger.info("Reading file %s", str(data_path))
    all_samples = load_json_file(str(data_path))
    logger.info("Read %d samples", len(all_samples))

    examples = expand_steps_into_examples(all_samples)
    logger.info("Total number of samples (expanded per-step): %d", len(examples))

    prompts = [build_prompt(sample) for sample in examples]
    logger.info("Starting vLLM engine for %d prompts. vLLM will handle internal batching.", len(prompts))
    # print(prompts[0])  # Print the first prompt for verification
    # raise Exception("Stop")

    llm = init_vllm(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        lora_adapter_path=args.lora_adapter_path,
    )
    
    lora_request = None
    if args.lora_adapter_path:
        lora_request = LoRARequest(
            lora_name=args.lora_name,
            lora_int_id=args.lora_int_id,
            lora_path=args.lora_adapter_path,
        )
        logger.info(
            "Using LoRA adapter: %s (name: %s, id: %s)",
            args.lora_adapter_path,
            args.lora_name,
            args.lora_int_id,
        )

    sampling_kwargs: Dict[str, Any] = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "n": args.num_samples,
    }
    if args.top_k is not None and args.top_k >= 0:
        sampling_kwargs["top_k"] = args.top_k

    sampling_params = SamplingParams(**sampling_kwargs)
    
    # Submitting all prompts to vLLM at once to leverage continuous batching.
    if lora_request:
        req_outputs = llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_tqdm=True,  # Enables the built-in vLLM progress bar
            lora_request=lora_request,
        )
    else:
        req_outputs = llm.generate(
            prompts=prompts, 
            sampling_params=sampling_params, 
            use_tqdm=True,
        )

    # Process and map outputs back to the original examples
    output2json = []
    for sample, req_out in zip(examples, req_outputs):
        outs = [o.text for o in req_out.outputs]
        scores = [extract_score_from_output(t) for t in outs]
        valid_scores = [s for s in scores if s is not None]
        avg_score = (sum(valid_scores) / len(valid_scores)) if valid_scores else None

        output2json.append(
            {
                "id": sample.get("id"),
                "orig_id": sample.get("orig_id"),
                "step_index": sample.get("step_index"),
                "outputs": outs,
                "scores": scores,
                "avg_score": avg_score,
            }
        )

    # Write results to disk once, greatly reducing I/O overhead
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output2json, f, ensure_ascii=False, indent=4)

    logger.info("Inference completed. Processed %d results, saved to %s", len(output2json), args.output)


if __name__ == "__main__":
    main()