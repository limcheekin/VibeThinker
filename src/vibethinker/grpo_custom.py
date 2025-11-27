"""
Custom GRPO implementation with MGPO entropy weighting.

This properly modifies the advantage calculation to incorporate entropy weighting,
rather than attempting to modify rewards (which gets normalized away).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MGPOBatchData:
    """Batch data with computed advantages."""

    query_tensors: torch.Tensor
    response_tensors: torch.Tensor
    rewards: torch.Tensor
    entropy_weights: torch.Tensor
    log_probabilities: torch.Tensor
    old_log_probabilities: torch.Tensor


class MGPOLoss:
    """
    MGPO (MaxEnt-Guided Policy Optimization) loss calculation.
    """

    def __init__(self, lambda_param: float = 4.0, epsilon: float = 0.2):
        """
        Initialize MGPO loss.
        """
        self.lambda_param = lambda_param
        self.epsilon = epsilon

    def compute_kl_entropy_weight(self, group_correctness: np.ndarray) -> float:
        """
        Compute KL divergence from maximum entropy distribution.
        """
        p_correct = np.mean(group_correctness)
        p_correct = np.clip(p_correct, 1e-6, 1 - 1e-6)
        p_0 = 0.5
        d_kl = p_correct * np.log(p_correct / p_0) + (1 - p_correct) * np.log(
            (1 - p_correct) / p_0
        )
        entropy_weight = np.exp(-self.lambda_param * d_kl)
        return float(entropy_weight)

    def compute_advantages(
        self, rewards: torch.Tensor, entropy_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute advantages with optional entropy weighting.
        """
        group_mean = rewards.mean(dim=1, keepdim=True)
        group_std = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards - group_mean) / group_std
        if entropy_weights is not None:
            w = entropy_weights.unsqueeze(1)
            advantages = w * advantages
        return advantages

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute clipped policy gradient loss.
        """
        log_prob_sum = log_probs.sum(dim=2)
        old_log_prob_sum = old_log_probs.sum(dim=2)
        ratio = torch.exp(log_prob_sum - old_log_prob_sum)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        loss = -torch.min(surr1, surr2)
        for i in range(loss.shape[0]):
            for j in range(loss.shape[1]):
                loss[i, j] = loss[i, j] / max(response_lengths[i, j].item(), 1)
        return loss.mean()


class MGPOTrainerWithEntropyWeighting:
    """
    Proper GRPO trainer with MGPO entropy weighting.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Any,
        reward_calculator: Any,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_calculator = reward_calculator
        self.device = device
        self.loss_fn = MGPOLoss(lambda_param=4.0, epsilon=0.2)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
        )

    def compute_log_probabilities(
        self, model_output: Any, response_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities of responses under current policy."""
        logits = model_output.logits
        batch_size, G, seq_len = response_ids.shape
        logits = logits.view(batch_size * G, seq_len, -1)
        response_ids_flat = response_ids.view(batch_size * G, seq_len)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.gather(2, response_ids_flat.unsqueeze(2)).squeeze(2)
        log_probs = log_probs.view(batch_size, G, seq_len)
        return log_probs

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute one training step with MGPO.
        """
        prompts = batch["prompts"]
        completions = batch["completions"]
        reference_answers = batch["reference_answers"]
        rewards_list, entropy_info = self.reward_calculator.compute_rewards(
            prompts, completions, reference_answers
        )
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        entropy_weights = torch.tensor(
            [info["entropy_weight"] for info in entropy_info],
            dtype=torch.float32,
            device=self.device,
        )
        self.model.eval()
        with torch.no_grad():
            old_log_probs_list: List[torch.Tensor] = []
            for i, comp_group in enumerate(completions):
                group_log_probs: List[torch.Tensor] = []
                for c in comp_group:
                    tokens = self.tokenizer(
                        c,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.config.max_completion_length,
                    )
                    with torch.no_grad():
                        outputs = self.model(
                            **{k: v.to(self.device) for k, v in tokens.items()}
                        )
                    log_p = self.compute_log_probabilities(
                        outputs, tokens["input_ids"].to(self.device)
                    )
                    group_log_probs.append(log_p)
                old_log_probs_list.append(torch.stack(group_log_probs))
            old_log_probs = torch.stack(old_log_probs_list)
        advantages = self.loss_fn.compute_advantages(rewards, entropy_weights)
        self.model.train()
        new_log_probs_list: List[torch.Tensor] = []
        for i, comp_group in enumerate(completions):
            group_log_probs = []
            for c in comp_group:
                tokens = self.tokenizer(
                    c,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_completion_length,
                )
                outputs = self.model(
                    **{k: v.to(self.device) for k, v in tokens.items()}
                )
                log_p = self.compute_log_probabilities(
                    outputs, tokens["input_ids"].to(self.device)
                )
                group_log_probs.append(log_p)
            new_log_probs_list.append(torch.stack(group_log_probs))
        new_log_probs = torch.stack(new_log_probs_list)
        loss: torch.Tensor = self.loss_fn.compute_policy_loss(
            new_log_probs,
            old_log_probs,
            advantages,
            response_lengths=torch.ones_like(advantages),
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {
            "loss": loss.item(),
            "reward_mean": rewards.mean().item(),
            "entropy_weight_mean": entropy_weights.mean().item(),
            "advantage_mean": advantages.mean().item(),
        }


if __name__ == "__main__":
    loss_fn = MGPOLoss(lambda_param=4.0)
    rewards = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    entropy_weights_list: List[float] = []
    for r in rewards:
        w = loss_fn.compute_kl_entropy_weight(r.numpy())
        entropy_weights_list.append(w)
    entropy_weights = torch.tensor(entropy_weights_list)
    advantages_no_weight = loss_fn.compute_advantages(rewards)
    advantages_with_weight = loss_fn.compute_advantages(rewards, entropy_weights)
    print("MGPO Loss Calculation Example")
    print("=" * 60)
    print(f"Rewards:\n{rewards}")
    print(f"Entropy weights: {entropy_weights}")
    print(f"Advantages (no weighting):\n{advantages_no_weight}")
    print(f"Advantages (with MGPO weighting):\n{advantages_with_weight}")
    print("Difference (should be larger for high-entropy problem):")
    print(advantages_with_weight / (advantages_no_weight + 1e-8))
