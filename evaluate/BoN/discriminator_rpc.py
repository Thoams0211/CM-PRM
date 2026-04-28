#!/usr/bin/env python3
"""BoN discriminator based on a new PRM-style model (single-stage analysis).

Compared with discriminator_genprm.py:
- Uses one unified prompt (no analyze/verify/execute multi-stage generation)
- Runs batched inference directly with vLLM
- Parses final verification label (Yes/No) to obtain reward
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vllm import LLM, SamplingParams


DEFAULT_PRM_INSTRUCTION_TEMPLATE = """###Instruction: \nYou are an excellent math teacher. Please verify the correctness of the Now Step.

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
Reply: \n###Response: \n"""

VERIFICATION_PATTERN = re.compile(
    r"Verification:\s*Is\s+the\s+step\s+correct\s+\(Yes/No\)\?\s*(Yes|No)",
    re.IGNORECASE,
)


class CandidateDiscriminatorRPC:
    """Selects the best candidate step using single-stage PRM analysis."""

    @staticmethod
    def _load_json_list(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
            return []
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return []

    @staticmethod
    def _save_json(path: str, data: List[Dict[str, Any]]) -> None:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _build_eval_prompts(self, samples: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[Any, int]]]:
    """Build evaluation prompts for the discriminator.
    
    Args:
        samples: List of samples to build prompts for.
        
    Returns:
        A tuple of lists:
            - prompts: List of prompts for the discriminator.
            - owner_pairs: List of tuples of sample IDs and candidate indices.
    
    """
        prompts: List[str] = []
        owner_pairs: List[Tuple[Any, int]] = []
        for sample in samples:
            sample_id = sample["id"]
            problem = sample["problem"]
            steps = sample.get("steps", [])
            for candidate_idx, candidate in enumerate(sample.get("candidates", [])):
                instruction = self.build_prm_instruction(problem, steps, candidate)
                prompts.append(f"<|user|>\n{instruction}\n<|assistant|>")
                owner_pairs.append((sample_id, candidate_idx))
        return prompts, owner_pairs

    @staticmethod
    def _group_results_by_sample(results: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
        grouped: Dict[Any, List[Dict[str, Any]]] = {}
        for result in results:
            grouped.setdefault(result["sample_id"], []).append(result)
        return grouped

    @staticmethod
    def _ground_truth_answer(sample: Dict[str, Any]) -> Optional[str]:
        answer = sample.get("answer")
        return str(answer).strip() if answer is not None else None

    def __init__(
        self,
        model_path: str,
        discriminator_path: str,
        generator_path: str,
        jsonl_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        batch_size: int = 32,
        prm_n: int = 8,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
    ) -> None:
        self.model_path = model_path
        self.discriminator_path = discriminator_path
        self.generator_path = generator_path
        self.jsonl_path = jsonl_path
        self.tensor_parallel_size = tensor_parallel_size
        self.batch_size = batch_size
        self.prm_n = prm_n
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        self.llm: Optional[LLM] = None

        self.ground_truth_answers: Dict[Any, str] = {}
        if self.jsonl_path and os.path.exists(self.jsonl_path):
            self.load_ground_truth_answers()

    def initialize_model(self) -> None:
        if self.llm is not None:
            return
        print(f"Loading model from {self.model_path} ...")
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            enable_prefix_caching=True,
            disable_custom_all_reduce=True,
        )
        print("Model loaded.")

    def load_ground_truth_answers(self) -> None:
        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    sample_id = data.get("id")
                    answer = data.get("answer")
                    if sample_id is not None and answer is not None:
                        self.ground_truth_answers[sample_id] = str(answer).strip()
            print(f"Loaded {len(self.ground_truth_answers)} ground truth answers from {self.jsonl_path}")
        except Exception as e:
            print(f"Warning: failed to load ground truth answers: {e}")

    @staticmethod
    def build_prm_instruction(problem: str, reasoning_chain: Sequence[str], candidate: str) -> str:
        """Build a PRM instruction for a single candidate.
        
        Args:
            problem: The problem to solve.
            reasoning_chain: The reasoning chain.
            candidate: The candidate to evaluate.
        
        Returns:
            A PRM instruction for the single candidate.
        """
        prev_lines = [f"Step {i + 1}: {step}" for i, step in enumerate(reasoning_chain)]
        previous_steps_text = "\n".join(prev_lines) if prev_lines else "None"
        now_step = f"Step {len(reasoning_chain) + 1}: {candidate}"
        return DEFAULT_PRM_INSTRUCTION_TEMPLATE.format(
            question=problem,
            previous_steps=previous_steps_text,
            now_step=now_step,
        )

    @staticmethod
    def extract_verification_label(text: str) -> Optional[str]:
        if not text:
            return None
        matches = VERIFICATION_PATTERN.findall(text)
        if not matches:
            return None
        label = matches[-1].strip().lower()
        return "Yes" if label == "yes" else "No"

    @staticmethod
    def label_to_score(label: Optional[str]) -> float:
        if label == "Yes":
            return 1.0
        if label == "No":
            return 0.0
        return 0.5

    @staticmethod
    def extract_boxed(text: str) -> Optional[str]:
        results = []
        i = 0
        while True:
            start = text.find(r"\boxed{", i)
            if start == -1:
                break
            j = start + len(r"\boxed{")
            depth = 1
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                content = text[start + len(r"\boxed{") : j - 1]
                results.append(content)
                i = j
            else:
                break
        if len(results) == 1:
            return results[0]
        return None

    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        answer = CandidateDiscriminatorRPC.extract_boxed(text or "")
        return answer.strip() if answer is not None else None

    @staticmethod
    def verify_answer(extracted_answer: str, ground_truth: str) -> bool:
        def normalize(ans: str) -> str:
            ans = ans.strip().lower()
            ans = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", ans)
            ans = re.sub(r"[{}$]", "", ans)
            return ans.strip()

        return normalize(extracted_answer) == normalize(ground_truth)

    def evaluate_candidates_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        """Evaluate candidates in batch.
        
        Args:
            samples: List of samples to evaluate.
            
        Returns:
            A list of dictionaries, each containing the evaluation results for a sample.
            The dictionary contains the following keys:
            - sample_id: The ID of the sample.
            - candidate_idx: The index of the candidate.
            - reward: The reward for the candidate.
            - analysis_outputs: The outputs of the analysis.
            - analysis_labels: The labels of the analysis.
        """

        self.initialize_model()
        assert self.llm is not None

        flat_prompts, owner_pairs = self._build_eval_prompts(samples)
        if not flat_prompts:
            return []

        print(f"Evaluating {len(flat_prompts)} candidates, batch_size={self.batch_size}, n={self.prm_n}")
        outputs = self.llm.generate(
            flat_prompts,
            SamplingParams(
                n=self.prm_n,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            ),
            use_tqdm=True,
        )

        aggregated_scores: Dict[Tuple[Any, int], List[float]] = {}
        aggregated_texts: Dict[Tuple[Any, int], List[str]] = {}
        for i, output in enumerate(outputs):
            key = owner_pairs[i]
            aggregated_scores.setdefault(key, [])
            aggregated_texts.setdefault(key, [])
            for cand in output.outputs:
                text = cand.text.strip()
                aggregated_scores[key].append(self.label_to_score(self.extract_verification_label(text)))
                aggregated_texts[key].append(text)

        results: List[Dict[str, Any]] = []
        for key, values in aggregated_scores.items():
            sample_id, candidate_idx = key
            texts = aggregated_texts.get(key, [])
            results.append(
                {
                    "sample_id": sample_id,
                    "candidate_idx": candidate_idx,
                    "reward": sum(values) / len(values) if values else 0.5,
                    "analysis_outputs": texts,
                    "analysis_labels": [self.extract_verification_label(text) for text in texts],
                }
            )
        return results

    def select_best_candidates(
        self,
        discriminator_samples: List[Dict[str, Any]],
        generator_samples: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> None:

    """Select the best candidates from the evaluation results.
    
    Args:
        discriminator_samples: List of discriminator samples.
        generator_samples: List of generator samples.
        results: List of evaluation results.

    """
        results_by_sample = self._group_results_by_sample(results)
        discriminator_by_id = {sample["id"]: sample for sample in discriminator_samples}

        for sample_id, sample_results in results_by_sample.items():
            d_sample = discriminator_by_id.get(sample_id)
            if not d_sample:
                continue
            d_sample["candidate_evaluations"] = [
                {
                    "candidate_idx": item.get("candidate_idx"),
                    "reward": item.get("reward"),
                    "analysis_labels": item.get("analysis_labels", []),
                    "analysis_outputs": item.get("analysis_outputs", []),
                }
                for item in sorted(sample_results, key=lambda x: x.get("candidate_idx", -1))
            ]

        for generator_sample in generator_samples:
            sample_id = generator_sample["id"]
            discriminator_sample = discriminator_by_id.get(sample_id)
            if not discriminator_sample:
                print(f"Warning: no discriminator data for sample {sample_id}, skipping")
                continue

            candidates = discriminator_sample.get("candidates", [])
            sample_results = results_by_sample.get(sample_id, [])
            if not candidates or not sample_results:
                print(f"Warning: no candidates/results for sample {sample_id}, skipping")
                continue

            best_result = max(sample_results, key=lambda item: item["reward"])
            best_idx = best_result["candidate_idx"]
            if best_idx >= len(candidates):
                print(f"Warning: invalid best candidate idx={best_idx} for sample {sample_id}")
                continue
            best_candidate = candidates[best_idx]
            extracted_answer = self.extract_answer(best_candidate)
            if extracted_answer:
                generator_sample["answer"] = extracted_answer
            print(f"Sample {sample_id}: choose candidate {best_idx} (reward={best_result['reward']:.4f})")

            generator_sample.setdefault("steps", [])
            generator_sample["steps"].append(best_candidate)

    def run(self) -> None:
        discriminator_samples = self._load_json_list(self.discriminator_path)
        if not discriminator_samples:
            raise ValueError(f"No data found in {self.discriminator_path}")

        generator_samples = self._load_json_list(self.generator_path)
        if not generator_samples:
            raise ValueError(f"No data found in {self.generator_path}")

        generator_by_id = {sample.get("id"): sample for sample in generator_samples}
        answered_ids = {
            sample.get("id")
            for sample in discriminator_samples
            if "answer" in sample
        }
        answered_ids.update(
            sample_id
            for sample_id, sample in generator_by_id.items()
            if isinstance(sample, dict) and "answer" in sample
        )

        samples_with_candidates = [
            sample
            for sample in discriminator_samples
            if sample.get("candidates") and sample.get("id") not in answered_ids
        ]

        print(f"Loaded discriminator samples: {len(discriminator_samples)}")
        print(f"Loaded generator samples: {len(generator_samples)}")
        if answered_ids:
            print(f"Skipped answered samples: {len(answered_ids)}")
        print(f"Samples with candidates to evaluate: {len(samples_with_candidates)}")

        if not samples_with_candidates:
            print("No pending samples to evaluate. Nothing to do.")
            return

        results = self.evaluate_candidates_batch(samples_with_candidates)
        if not results:
            raise ValueError("No evaluation results obtained")

        self.select_best_candidates(samples_with_candidates, generator_samples, results)
        self._save_json(self.generator_path, generator_samples)
        self._save_json(self.discriminator_path, discriminator_samples)
        print(f"Saved updated generator file to {self.generator_path}")
        print(f"Saved updated discriminator file (with analyses) to {self.discriminator_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate discriminator with single-stage PRM analysis")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--discriminator_path", type=str, default="buffer/discriminator.json")
    parser.add_argument("--generator_path", type=str, default="buffer/generator.json")
    parser.add_argument("--jsonl_path", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--prm_n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)

    args = parser.parse_args()

    discriminator = CandidateDiscriminatorRPC(
        model_path=args.model_path,
        discriminator_path=args.discriminator_path,
        generator_path=args.generator_path,
        jsonl_path=args.jsonl_path,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        prm_n=args.prm_n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    discriminator.run()


if __name__ == "__main__":
    main()
