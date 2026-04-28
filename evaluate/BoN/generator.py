"""
Generator for mathematical problem solving using vLLM.

This module handles initialization from JSONL files and step-by-step reasoning
generation using vLLM for batch inference.
"""
import os
import json
import argparse
from typing import Dict, Any, Optional, List
from vllm import LLM, SamplingParams

from policy_prompt import Qwen_Policy_Prompt


class MathProblemGenerator:
    """Generate step-by-step reasoning candidates with vLLM."""

    def __init__(
        self,
        model_path: str,
        buffer_path: str = "buffer/generator.json",
        tensor_parallel_size: int = 1,
        batch_size: int = 1,
        num_candidates: int = 8,
        discriminator_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.buffer_path = buffer_path
        self.discriminator_path = discriminator_path or "buffer/discriminator.json"
        self.tensor_parallel_size = tensor_parallel_size
        self.batch_size = batch_size
        self.num_candidates = num_candidates
        self.llm: Optional[LLM] = None
        self.data: Optional[List[Dict[str, Any]]] = None

    def initialize_model(self):
        if self.llm is None:
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=0.90,
                enable_chunked_prefill=True,
            )

    def _load_json_buffer(self, path: str) -> List[Dict[str, Any]]:
        if not path or not os.path.exists(path):
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
        except (json.JSONDecodeError, IOError):
            return []
        return []

    def _save_json_buffer(self, path: str, data: List[Dict[str, Any]]):
        if not path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_buffer(self) -> List[Dict[str, Any]]:
        return self._load_json_buffer(self.buffer_path)

    def save_buffer(self, data: List[Dict[str, Any]]):
        self._save_json_buffer(self.buffer_path, data)

    def load_discriminator_buffer(self) -> List[Dict[str, Any]]:
        return self._load_json_buffer(self.discriminator_path)

    def save_discriminator_buffer(self, data: List[Dict[str, Any]]):
        self._save_json_buffer(self.discriminator_path, data)

    def initialize_from_jsonl(self, jsonl_path: str):
        """Load data from JSONL file and initialize buffer
        
        Args:
            jsonl_path: Path to the JSONL file.
            
        Returns:
            None
        
        """
        buffer_data = self.load_buffer()
        if buffer_data:
            self.data = buffer_data
            return

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                problem_data = json.loads(line)
                samples.append(
                    {
                        "id": problem_data.get("id"),
                        "problem": problem_data.get("problem"),
                        "steps": [],
                    }
                )

        if not samples:
            raise ValueError(f"JSONL file is empty: {jsonl_path}")

        self.save_buffer(samples)
        print(f"Initialized {len(samples)} samples from JSONL: ids={[s['id'] for s in samples]}")

        discriminator_samples = [
            {"id": sample["id"], "problem": sample["problem"], "steps": []}
            for sample in samples
        ]
        self.save_discriminator_buffer(discriminator_samples)
        print(f"Saved {len(discriminator_samples)} samples to discriminator buffer: {self.discriminator_path}")

        self.data = samples

    def construct_prompt(self, problem: str, steps: List[str]) -> str:
        """Prompt for generating next step
        
        Args:
            problem: The problem to solve.
            steps: List of steps already generated.
            
        Returns:
            A prompt for generating the next step.
        
        """
        previous_steps = "\n\n".join(steps)
        prompt = Qwen_Policy_Prompt.format(question=problem, previous_steps=previous_steps)
        # prompt = prompt.replace("left-brackets", "{").replace("right-brackets", "}")
        return prompt + ("\n\n" if steps else "")

    def generate_next_step(self) -> List[List[str]]:
        """Generate next step for all samples"""
        if self.llm is None:
            self.initialize_model()
        if not self.data:
            raise ValueError("Data not initialized. Call initialize_from_jsonl() first.")

        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=512,
            n=self.num_candidates,
            stop=["<|im_end|>", "\n\n"],
        )
        all_candidates: List[List[str]] = []
        batch_size = max(1, self.batch_size)

        # Generate next step for all samples
        for start in range(0, len(self.data), batch_size):
            batch_samples = self.data[start : start + batch_size]
            prompts = [self.construct_prompt(sample["problem"], sample.get("steps", [])) for sample in batch_samples]
            outputs = self.llm.generate(prompts, sampling_params)
            # Keep vLLM outputs in the same order as `batch_samples` / `prompts`.
            batch_candidates = [
                [item.text.strip() for item in output.outputs if item.text.strip()]
                for output in outputs
            ]
            all_candidates.extend(batch_candidates)

        return all_candidates

    def _merge_buffers(self, generator_data: List[Dict[str, Any]], discriminator_data: List[Dict[str, Any]]):
        """Merge generator and discriminator data
        
        Args:
            generator_data: List of generator samples.
            discriminator_data: List of discriminator samples.
            
        Returns:
            A list of merged samples.
        
        """
        generator_by_id = {s.get("id"): s for s in generator_data} if generator_data else {}
        discriminator_by_id = {s.get("id"): s for s in discriminator_data} if discriminator_data else {}

        # Merge generator and discriminator data   
        merged_data = []
        for sid in set(generator_by_id) | set(discriminator_by_id):
            g = generator_by_id.get(sid, {})
            d = discriminator_by_id.get(sid, {})
            merged_sample = dict(d) if d else {}
            if "id" in g or "id" in d:
                merged_sample["id"] = g.get("id", d.get("id"))
            merged_sample["problem"] = g.get("problem", d.get("problem"))
            merged_sample["steps"] = g.get("steps", d.get("steps", []))
            if "answer" in g:
                merged_sample["answer"] = g["answer"]
            elif "answer" in d:
                merged_sample["answer"] = d["answer"]
            merged_data.append(merged_sample)

        return sorted(merged_data, key=lambda x: x.get("id", 0))

    def run_generation(self, jsonl_path: Optional[str] = None):
        """Run generation"""

        # Load generator data from buffer or JSONL file
        generator_data = self.load_buffer()
        if not generator_data and jsonl_path:
            self.initialize_from_jsonl(jsonl_path)
            generator_data = self.load_buffer()

        # Load discriminator data from buffer
        discriminator_data = self.load_discriminator_buffer()
        if not discriminator_data and not generator_data:
            raise ValueError(
                "No data available. Please provide a JSONL file for initialization or ensure generator/discriminator files exist."
            )

        # Merge generator and discriminator data
        merged_data = self._merge_buffers(generator_data, discriminator_data)

        # Get samples to process and skipped samples
        samples_to_process = [s for s in merged_data if "answer" not in s]
        skipped_samples = [s for s in merged_data if "answer" in s]
        if skipped_samples:
            print(f"Skipped {len(skipped_samples)} samples that already have answers: ids={[s['id'] for s in skipped_samples]}")
        if not samples_to_process:
            print("All samples already have answers. No generation needed.")
            return
        self.data = samples_to_process
        print(f"Generating {self.num_candidates} candidates per sample for {len(self.data)} samples using one prompt each...")
        
        # Generate next step for all samples
        candidates_list = self.generate_next_step()

        # Save generator samples
        generator_samples = []
        for sample, candidates in zip(self.data, candidates_list):
            generator_samples.append(
                {
                    "id": sample["id"],
                    "problem": sample["problem"],
                    "steps": sample.get("steps", []),
                    "candidates": candidates,
                }
            )
            print(f"Sample id={sample['id']}: Generated {len(candidates)} candidates")
            for index, candidate in enumerate(candidates[:3], start=1):
                print(f"  Candidate {index}: {candidate[:80]}...")

        # Save all samples to discriminator buffer
        all_samples = sorted(skipped_samples + generator_samples, key=lambda x: x.get("id", 0))
        self.save_discriminator_buffer(all_samples)
        print(
            f"Saved {len(all_samples)} samples to {self.discriminator_path} "
            f"({len(skipped_samples)} skipped, {len(generator_samples)} generated)"
        )


def main():
    parser = argparse.ArgumentParser(description="Mathematical problem step generator using vLLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the vLLM model")
    parser.add_argument("--jsonl_path", type=str, default=None, help="Path to JSONL file for initialization (if buffer is empty)")
    parser.add_argument("--buffer_path", type=str, default="buffer/generator.json", help="Path to buffer JSON file (default: buffer/generator.json)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference (default: 1)")
    parser.add_argument("--num_candidates", type=int, default=8, help="Number of candidate steps to generate per sample (default: 8)")
    parser.add_argument("--discriminator_path", type=str, default=None, help="Path to discriminator JSON file (default: None)")
    
    # Create generator
    generator = MathProblemGenerator(
        model_path=args.model_path,
        buffer_path=args.buffer_path,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        num_candidates=args.num_candidates,
        discriminator_path=args.discriminator_path
    )
    
    # Run generation
    generator.run_generation(
        jsonl_path=args.jsonl_path,
    )


if __name__ == "__main__":
    main()
