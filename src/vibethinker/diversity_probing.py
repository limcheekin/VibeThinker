"""
Domain-Aware Diversity Probing.
Implements the selection mechanism for the Spectrum Phase of SSP (Section 3.3).
Calculates Pass@K for intermediate SFT checkpoints to identify specialist models.
"""

import argparse
import json
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel

from vibethinker.monitor import MGPORewardCalculator


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate unbiased Pass@K estimator.

    Formula: 1 - product(1 - k / (n - i) for i in range(c))

    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: K value for Pass@K metric

    Returns:
        Pass@K probability estimate
    """
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


# Domain-specific prompt templates for improved context
DOMAIN_PROMPTS = {
    "algebra": (
        "Solve the following algebra problem step by step. "
        "Show your work and box the final answer:\n\n{problem}\n\nSolution:"
    ),
    "geometry": (
        "Solve the following geometry problem using geometric reasoning. "
        "Draw diagrams if helpful and box the final answer:\n\n{problem}\n\nSolution:"
    ),
    "calculus": (
        "Solve the following calculus problem step by step. "
        "Show all derivatives, integrals, or limits clearly and box the final answer:\n\n{problem}\n\nSolution:"
    ),
    "statistics": (
        "Solve the following statistics problem step by step. "
        "Show all calculations and box the final answer:\n\n{problem}\n\nSolution:"
    ),
    "code": (
        "Write a Python function to solve the following problem:\n\n{problem}\n\n```python\n"
    ),
    "math": ("Solve the following problem step by step:\n\n{problem}\n\nSolution:"),
}


class DiversityProber:
    """Probe model diversity using Pass@K metric."""

    def __init__(self, model_path: str, max_seq_length: int = 2048) -> None:
        """
        Initialize diversity prober with a model checkpoint.

        Args:
            model_path: Path to the model checkpoint
            max_seq_length: Maximum sequence length for the model

        Raises:
            ValueError: If model loading fails
        """
        print(f"Loading model for probing: {model_path}")
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_path,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                dtype=None,
            )
            FastLanguageModel.for_inference(self.model)
            self.reward_calc = MGPORewardCalculator()
        except Exception as e:
            raise ValueError(
                f"Failed to load model from {model_path}. "
                f"Ensure it contains full model weights (not just LoRA adapters). "
                f"If this is a LoRA checkpoint, use save_pretrained_merged() during training. "
                f"Original error: {e}"
            )

    def probe_domain(
        self,
        problems: List[Dict[str, Any]],
        k: int = 8,
        num_generations: int = 16,
        domain: str = "math",
    ) -> Dict[str, float]:
        """
        Evaluate the model on a domain-specific dataset using Pass@K.

        Args:
            problems: List of dicts with 'problem' and 'answer'.
            k: The 'K' in Pass@K (diversity metric).
            num_generations: Total samples (N) to generate per problem (must be >= k).

        Returns:
            Dictionary with pass@k, pass@1, and diversity_score metrics
        """
        print(f"Probing diversity (Pass@{k}) over {len(problems)} problems...")

        total_pass_at_k = 0.0
        total_pass_at_1 = 0.0

        for item in tqdm(problems, desc="Probing"):
            # Domain-aware prompt construction using templates
            prompt_template = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["math"])
            prompt_text = prompt_template.format(problem=item["problem"])

            inputs = self.tokenizer([prompt_text], return_tensors="pt").to("cuda")

            # Generate N solutions (spectrum)
            # Note: Paper implies higher temp/sampling for exploring the spectrum
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=num_generations,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            prompt_len = inputs.input_ids.shape[1]
            correct_count = 0

            for output in outputs:
                generated_text = self.tokenizer.decode(
                    output[prompt_len:], skip_special_tokens=True
                )
                # Binary reward verification
                score = self.reward_calc.evaluate_solution(
                    generated_text, item["answer"]
                )
                if score > 0.5:
                    correct_count += 1

            # Calculate metrics
            p_at_k = calculate_pass_at_k(num_generations, correct_count, k)
            p_at_1 = correct_count / num_generations

            total_pass_at_k += p_at_k
            total_pass_at_1 += p_at_1

        avg_pass_at_k = total_pass_at_k / len(problems)
        avg_pass_at_1 = total_pass_at_1 / len(problems)

        return {
            "pass@k": avg_pass_at_k,
            "pass@1": avg_pass_at_1,
            "diversity_score": avg_pass_at_k,  # This is the selection metric
        }

    def unload_model(self) -> None:
        """Explicitly unload model from GPU to free VRAM.

        Critical for multi-domain iteration to prevent OOM crashes.
        """
        import gc

        import torch

        # Move model to CPU first
        if hasattr(self, "model") and hasattr(self.model, "cpu"):
            self.model.cpu()

        # Delete model references
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if hasattr(self, "model"):
            self.unload_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to SFT checkpoint"
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to domain validation .jsonl"
    )
    parser.add_argument("--k", type=int, default=8, help="K for Pass@K")
    parser.add_argument(
        "--n", type=int, default=16, help="Number of generations per problem"
    )
    args = parser.parse_args()

    # Load validation data
    with open(args.data_path, "r") as f:
        data = [json.loads(line) for line in f]

    prober = DiversityProber(args.model_path)
    metrics = prober.probe_domain(data, k=args.k, num_generations=args.n)

    print("\n" + "=" * 60)
    print(f"Diversity Probing Results: {args.model_path}")
    print("-" * 60)
    print(f"Pass@{args.k} (Diversity): {metrics['pass@k']:.4f} <== SELECTION METRIC")
    print(f"Pass@1 (Accuracy):  {metrics['pass@1']:.4f}")
    print("=" * 60)
