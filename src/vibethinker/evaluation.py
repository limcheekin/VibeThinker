"""
Evaluation script for VibeThinker Phase 3 (AIME, MATH-500, GPQA, LiveCodeBench).
Implements symbolic answer evaluation and Pass@K metrics.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .inference_optimize import OptimizedInference
from .monitor import MGPORewardCalculator

BENCHMARKS = {
    "AIME": "data/aime24_25.jsonl",
    "MATH-500": "data/math_500.jsonl",
    "GPQA": "data/gpqa.jsonl",
    "LiveCodeBench": "data/livecodebench.jsonl",
}


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    problems: List[Dict[str, Any]],
    reward_calc: MGPORewardCalculator,
    n_samples: int = 1,
) -> Dict[str, Any]:
    inf = OptimizedInference(model, tokenizer)
    results = []
    for problem in problems:
        prompt = problem["prompt"]
        reference = problem["answer"]
        generations = [inf.generate_optimized(prompt) for _ in range(n_samples)]
        rewards = [reward_calc.evaluate_solution(gen, reference) for gen in generations]
        pass_at_k = float(any(r == 1.0 for r in rewards))
        results.append(
            {
                "prompt": prompt,
                "reference": reference,
                "generations": generations,
                "rewards": rewards,
                "pass@k": pass_at_k,
            }
        )
    accuracy = np.mean([r["pass@k"] for r in results])
    return {"results": results, "accuracy": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS.keys()))
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="outputs/test_output/")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)  # type: ignore
    reward_calc = MGPORewardCalculator()

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}
    for bench in args.benchmarks:
        print(f"Evaluating {bench}...")
        problems = load_benchmark(BENCHMARKS[bench])
        result = evaluate_model(
            model, tokenizer, problems, reward_calc, n_samples=args.n_samples
        )
        summary[bench] = result["accuracy"]
        with open(f"{args.output_dir}/{bench}_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"{bench} accuracy: {result['accuracy']:.4f}")
    with open(f"{args.output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
