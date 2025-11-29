"""
Custom GRPO implementation with MGPO entropy weighting.
Includes vectorized processing for efficiency and robust padding handling.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
        self.lambda_param = lambda_param
        self.epsilon = epsilon

    def compute_kl_entropy_weight(self, group_correctness: np.ndarray) -> float:
        """Compute KL divergence from maximum entropy distribution."""
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
        """Compute advantages with optional entropy weighting."""
        # rewards shape: (B, G)
        group_mean = rewards.mean(dim=1, keepdim=True)
        group_std = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards - group_mean) / group_std
        if entropy_weights is not None:
            w = entropy_weights.unsqueeze(1)
            advantages = w * advantages
        return advantages

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,  # (B, G, Seq)
        old_log_probs: torch.Tensor,  # (B, G, Seq)
        advantages: torch.Tensor,  # (B, G)
        response_lengths: torch.Tensor,  # (B, G)
    ) -> torch.Tensor:
        """Compute clipped policy gradient loss."""
        # Sum log probs over the sequence dimension to get total log prob of completion
        # Assumes log_probs are already masked (0.0 for prompt/padding)
        log_prob_sum = log_probs.sum(dim=2)  # (B, G)
        old_log_prob_sum = old_log_probs.sum(dim=2)  # (B, G)

        ratio = torch.exp(log_prob_sum - old_log_prob_sum)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

        # Loss per sample
        loss_matrix = -torch.min(surr1, surr2)

        # Normalize by response length to prevent long responses dominating
        safe_lengths = torch.clamp(response_lengths, min=1.0)
        loss_matrix = loss_matrix / safe_lengths

        return loss_matrix.mean()


class MGPOTrainerWithEntropyWeighting:
    """
    Proper GRPO trainer with MGPO entropy weighting and vectorized processing.
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

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.loss_fn = MGPOLoss(lambda_param=4.0, epsilon=0.2)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
        )

    def compute_log_probabilities(
        self, model_output: Any, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute token-level log probabilities.
        Returns tensor of shape aligned with input_ids (Batch, ..., Seq).
        """
        logits = model_output.logits  # (..., Seq, Vocab)

        # Shift logits: logits[t] predicts input_ids[t+1]
        # Supports arbitrary batch dimensions using ellipsis ...
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Calculate log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather probabilities of actual tokens
        # Output Shape matches shift_labels: (..., Seq-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Prepend zero to match input sequence length
        # Dynamically create prefix shape to match leading dimensions
        prefix_shape = list(token_log_probs.shape)
        prefix_shape[-1] = 1

        prefix = torch.zeros(
            prefix_shape, device=token_log_probs.device, dtype=token_log_probs.dtype
        )
        # Use dim=-1 to concatenate along the sequence dimension regardless of batch rank
        return torch.cat([prefix, token_log_probs], dim=-1)

    def _process_batch_log_probs(
        self,
        prompts: List[str],
        completions: List[List[str]],
        is_training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized processing of prompts + completions.
        Returns: (stacked_log_probs, stacked_lengths) in shapes (B, G, Seq) and (B, G)
        """
        B = len(prompts)
        G = len(completions[0])

        # 1. Flatten inputs for vectorized tokenization
        flat_prompts = []
        flat_completions = []
        for p, group in zip(prompts, completions):
            flat_prompts.extend([p] * len(group))
            flat_completions.extend(group)

        # 2. Construct full texts
        full_texts = [p + c for p, c in zip(flat_prompts, flat_completions)]

        # 3. Tokenize Full Texts (Padding enabled)
        # Use max_seq_length from config if available, else max_completion_length
        max_len = getattr(self.config, "max_seq_length", 4096)

        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_new_tokens=max_len,
        ).to(self.device)

        # 4. Forward Pass
        if is_training:
            outputs = self.model(**inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)

        # 5. Compute Log Probs (Shape: B*G, Seq)
        log_probs = self.compute_log_probabilities(outputs, inputs.input_ids)

        # 6. Apply Masking (Prompt + Padding)

        # Calculate prompt lengths to mask them out
        # Note: We tokenize prompts separately to get exact lengths
        # return_tensors=None to allow variable length lists (no padding needed here)
        prompt_inputs = self.tokenizer(
            flat_prompts, return_tensors=None, padding=False, add_special_tokens=True
        )
        prompt_lengths = [len(ids) for ids in prompt_inputs["input_ids"]]

        # Create mask
        mask = torch.ones_like(log_probs)
        for i, p_len in enumerate(prompt_lengths):
            # Zero out prompt tokens
            # Ensure we don't index out of bounds if truncation happened
            safe_p_len = min(p_len, log_probs.shape[1])
            mask[i, :safe_p_len] = 0.0

        # Also zero out padding (where attention_mask is 0)
        mask = mask * inputs.attention_mask

        # Apply mask
        log_probs = log_probs * mask

        # 7. Calculate Response Lengths
        # Total non-zero mask entries per sequence
        response_lens = mask.sum(dim=1)

        # 8. Reshape back to (B, G, Seq)
        Seq = log_probs.shape[1]
        log_probs = log_probs.view(B, G, Seq)
        response_lens = response_lens.view(B, G)

        return log_probs, response_lens

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute one training step with MGPO.
        """
        prompts = batch["prompts"]
        completions = batch["completions"]
        reference_answers = batch["reference_answers"]

        # 1. Compute Rewards
        rewards_list, entropy_info = self.reward_calculator.compute_rewards(
            prompts, completions, reference_answers
        )
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        entropy_weights = torch.tensor(
            [info["entropy_weight"] for info in entropy_info],
            dtype=torch.float32,
            device=self.device,
        )

        # 2. Compute Old Log Probs (Reference Policy)
        # We assume generated completions came from the 'old' policy state
        self.model.eval()
        with torch.no_grad():
            old_log_probs, response_lengths = self._process_batch_log_probs(
                prompts, completions, is_training=False
            )

        # 3. Compute Advantages
        advantages = self.loss_fn.compute_advantages(rewards, entropy_weights)

        # 4. Compute New Log Probs (Current Policy) & Update
        self.model.train()
        new_log_probs, _ = self._process_batch_log_probs(
            prompts, completions, is_training=True
        )

        loss = self.loss_fn.compute_policy_loss(
            new_log_probs,
            old_log_probs,
            advantages,
            response_lengths=response_lengths,
        )

        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "reward_mean": rewards.mean().item(),
            "entropy_weight_mean": entropy_weights.mean().item(),
            "advantage_mean": advantages.mean().item(),
        }
