"""
Merge domain-specific expert models into a unified spectrum model.

This implements the fusion strategy from the paper where domain specialists
(algebra, geometry, calculus, statistics) are combined.
"""

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM


def load_expert_models(expert_paths: List[str], device: str = "cuda") -> List[Any]:
    """Load all expert models."""
    experts: List[Any] = []
    for path in expert_paths:
        model: Any = AutoModelForCausalLM.from_pretrained(str(path))
        model.to(device)
        model.eval()
        experts.append(model)
    return experts


def fusion_weighted_average(
    expert_models: List[torch.nn.Module], weights: Optional[List[float]] = None
) -> torch.nn.Module:
    """
    Fuse expert models using weighted parameter averaging.
    """
    if weights is None:
        weights = [1.0 / len(expert_models)] * len(expert_models)
    assert len(weights) == len(expert_models)
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
    fused_model = AutoModelForCausalLM.from_pretrained(
        str(getattr(expert_models[0].config, "_name_or_path", ""))
    )
    with torch.no_grad():
        for name, param in fused_model.named_parameters():
            param.data.zero_()
            for expert, weight in zip(expert_models, weights):
                expert_param = dict(expert.named_parameters())[name]
                param.data += weight * expert_param.data
    return fused_model


def fusion_task_arithmetic(
    base_model: torch.nn.Module,
    expert_models: List[torch.nn.Module],
    scaling: float = 0.3,
) -> torch.nn.Module:
    """
    Task arithmetic fusion: fused = base + α * Σ(expert_i - base).
    """
    fused_model = AutoModelForCausalLM.from_pretrained(
        str(getattr(base_model.config, "_name_or_path", ""))
    )
    with torch.no_grad():
        for name, param in fused_model.named_parameters():
            base_param = dict(base_model.named_parameters())[name]
            task_vector_sum = torch.zeros_like(base_param)
            for expert in expert_models:
                expert_param = dict(expert.named_parameters())[name]
                task_vector_sum += expert_param.data - base_param.data
            param.data = base_param.data + scaling * task_vector_sum
    return fused_model


def validate_fusion(
    fused_model: Any, tokenizer: Any, test_problems: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Validate fused model performance across domains."""
    fused_model.eval()
    domain_accuracies: Dict[str, float] = {}
    for domain in ["algebra", "geometry", "calculus", "statistics"]:
        domain_problems = [p for p in test_problems if p["domain"] == domain]
        correct = 0
        for problem in domain_problems[:10]:
            prompt = f"Solve: {problem['question']}"
            inputs = tokenizer(prompt, return_tensors="pt").to(fused_model.device)
            with torch.no_grad():
                outputs = fused_model.generate(
                    inputs["input_ids"],
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if problem["answer"].lower() in response.lower():
                correct += 1
        accuracy = correct / min(len(domain_problems), 10)
        domain_accuracies[domain] = accuracy
    return domain_accuracies


if __name__ == "__main__":
    expert_paths = [
        "checkpoints/vibethinker_algebra",
        "checkpoints/vibethinker_geometry",
        "checkpoints/vibethinker_calculus",
        "checkpoints/vibethinker_statistics",
    ]
    print("Loading expert models...")
    experts = load_expert_models(expert_paths)
    print("Fusing models with weighted average...")
    fused_model = fusion_weighted_average(experts, weights=[0.25, 0.25, 0.25, 0.25])
    output_path = "checkpoints/vibethinker_spectrum_fused"
    if hasattr(fused_model, "save_pretrained") and callable(
        fused_model.save_pretrained
    ):
        fused_model.save_pretrained(output_path)
    else:
        raise TypeError(
            f"""
            fused_model does not have a callable save_pretrained method.
            Type: {type(fused_model)}"""
        )
    print(f"✓ Fused model saved to: {output_path}")
