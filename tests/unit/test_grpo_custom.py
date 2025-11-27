import numpy as np
import pytest
import torch

from vibethinker.grpo_custom import MGPOLoss


@pytest.fixture
def mgpo_loss():
    return MGPOLoss(lambda_param=4.0, epsilon=0.2)


def test_compute_kl_entropy_weight(mgpo_loss):
    # High entropy (50% correct) -> weight should be close to 1.0
    high_entropy_correctness = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    high_entropy_weight = mgpo_loss.compute_kl_entropy_weight(high_entropy_correctness)
    assert 0.9 < high_entropy_weight <= 1.0

    # Low entropy (100% correct) -> weight should be close to 0.0
    low_entropy_correctness = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    low_entropy_weight = mgpo_loss.compute_kl_entropy_weight(low_entropy_correctness)
    assert 0.0 <= low_entropy_weight < 0.1


def test_compute_advantages(mgpo_loss):
    rewards = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]], dtype=torch.float32
    )

    advantages = mgpo_loss.compute_advantages(rewards)

    # The rewards [1.0, 1.0, 0.0, 0.0] have a mean of 0.5.
    # The unbiased standard deviation (torch.std default) is ~0.577.
    # The advantages are (rewards - mean) / std, so |0.5 / 0.577| ~= 0.866.
    assert torch.allclose(
        torch.abs(advantages[0]),
        torch.tensor([0.8660, 0.8660, 0.8660, 0.8660]),
        atol=1e-4,
    )

    # Second group has mean 0.5 and std 0, so advantages should be all 0
    assert torch.allclose(advantages[1], torch.tensor([0.0, 0.0, 0.0, 0.0]))


def test_compute_advantages_with_entropy_weighting(mgpo_loss):
    rewards = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    entropy_weights = torch.tensor([0.5], dtype=torch.float32)

    advantages = mgpo_loss.compute_advantages(rewards, entropy_weights)

    # Advantages should be scaled by the entropy weight
    base_advantages = mgpo_loss.compute_advantages(rewards)
    assert torch.allclose(advantages, base_advantages * 0.5)
