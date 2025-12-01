#!/usr/bin/env python3
"""
Spectrum-to-Signal Pipeline Orchestrator.
Implements the SSP methodology from VibeThinker paper.
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List

from vibethinker.diversity_probing import DiversityProber
from vibethinker.model_fusion import fusion_weighted_average, load_expert_models


# 4 Domains from the paper
DOMAINS = ["algebra", "geometry", "calculus", "statistics"]
BASE_MODEL = "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit"


def get_best_checkpoint(
    domain: str, checkpoint_dir: str, data_path: str, k: int = 8, num_gen: int = 16
) -> str:
    """
    Select the intermediate checkpoint with highest Diversity (Pass@K).
    This implements the core SSP selection logic from the paper.
    
    Args:
        domain: Domain name
        checkpoint_dir: Directory containing checkpoints
        data_path: Path to validation data
        k: K value for Pass@K metric
        num_gen: Number of generations per problem
        
    Returns:
        Path to the best checkpoint
    """
    checkpoints = sorted(glob.glob(f"{checkpoint_dir}/checkpoint-*"))
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    best_ckpt = None
    best_score = -1.0
    
    # Load validation data (subset for efficiency)
    with open(data_path, 'r') as f:
        probing_data = [json.loads(line) for line in f][:64]
    
    print(f"\n{'='*60}")
    print(f"Probing {len(checkpoints)} checkpoints for {domain}...")
    print(f"{'='*60}")
    
    for ckpt in checkpoints:
        print(f"\nEvaluating: {ckpt}")
        
        # Load checkpoint and probe
        prober = DiversityProber(ckpt)
        metrics = prober.probe_domain(probing_data, k=k, num_generations=num_gen)
        
        # SSP Metric: Diversity Score (Pass@K)
        score = metrics['diversity_score']
        print(f"  Pass@{k} (Diversity): {score:.4f}")
        print(f"  Pass@1 (Accuracy):   {metrics['pass@1']:.4f}")
        
        if score > best_score:
            best_score = score
            best_ckpt = ckpt
            print(f"  ✓ New best checkpoint!")
        
        # Free memory
        del prober
    
    print(f"\n{'='*60}")
    print(f"Selected for {domain}: {best_ckpt}")
    print(f"Best Diversity Score: {best_score:.4f}")
    print(f"{'='*60}\n")
    
    return best_ckpt


def train_domain_specialist(
    domain: str,
    data_path: str,
    base_model: str,
    output_dir: str,
    max_steps: int = 1000,
) -> str:
    """
    Train a specialist for one domain.
    
    Returns:
        Path to checkpoint directory
    """
    print(f"\n{'='*60}")
    print(f"Training Specialist: {domain}")
    print(f"{'='*60}\n")
    
    checkpoint_dir = f"{output_dir}/{domain}_specialist"
    
    # Call training script
    cmd = (
        f"python scripts/train_sft_specialist.py "
        f"--domain {domain} "
        f"--data {data_path} "
        f"--base-model {base_model} "
        f"--out {output_dir} "
        f"--max-steps {max_steps}"
    )
    
    print(f"Running: {cmd}\n")
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        raise RuntimeError(f"Training failed for {domain} with exit code {exit_code}")
    
    return checkpoint_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VibeThinker Spectrum-to-Signal Pipeline Orchestrator"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing domain-specific data files (e.g., data/algebra.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints",
        help="Output directory for checkpoints and fused model",
    )
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="Base model to use for all specialists",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Training steps per specialist",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="K value for Pass@K diversity metric",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=16,
        help="Number of generations for diversity probing",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training phase (use existing checkpoints)",
    )
    
    args = parser.parse_args()
    
    selected_experts: List[str] = []
    
    # Phase 1: Train Specialists
    for domain in DOMAINS:
        data_path = f"{args.data_dir}/{domain}.jsonl"
        
        if not os.path.exists(data_path):
            print(f"WARNING: Data file not found: {data_path}")
            print(f"Skipping domain: {domain}")
            continue
        
        if not args.skip_training:
            checkpoint_dir = train_domain_specialist(
                domain=domain,
                data_path=data_path,
                base_model=args.base_model,
                output_dir=args.output_dir,
                max_steps=args.max_steps,
            )
        else:
            checkpoint_dir = f"{args.output_dir}/{domain}_specialist"
        
        # Phase 2: Select Best Checkpoint via Diversity Probing
        try:
            best_ckpt = get_best_checkpoint(
                domain=domain,
                checkpoint_dir=checkpoint_dir,
                data_path=data_path,
                k=args.k,
                num_gen=args.num_generations,
            )
            selected_experts.append(best_ckpt)
        except Exception as e:
            print(f"ERROR selecting checkpoint for {domain}: {e}")
            print(f"Skipping {domain}")
            continue
    
    # Phase 3: Fuse Selected Experts
    if selected_experts:
        print(f"\n{'='*60}")
        print(f"Fusing {len(selected_experts)} Selected Experts")
        print(f"{'='*60}\n")
        
        for i, expert in enumerate(selected_experts):
            print(f"  Expert {i+1}: {expert}")
        
        print("\nLoading experts...")
        experts = load_expert_models(selected_experts)
        
        print("Performing weighted average fusion...")
        weights = [1.0 / len(experts)] * len(experts)
        fused_model = fusion_weighted_average(experts, weights=weights)
        
        output_path = f"{args.output_dir}/vibethinker_spectrum_fused"
        print(f"\nSaving fused model to: {output_path}")
        fused_model.save_pretrained(output_path)
        
        print(f"\n{'='*60}")
        print(f"✓ Spectrum Phase Complete!")
        print(f"✓ Fused model saved to: {output_path}")
        print(f"{'='*60}\n")
    else:
        print("\nERROR: No experts were trained/selected!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
