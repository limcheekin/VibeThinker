import numpy as np
import pytest
import torch

from vibethinker.grpo_custom import MGPOLoss


def test_mgpo_loss_kl_entropy_weight():
    loss_fn = MGPOLoss(lambda_param=4.0)

    # High entropy (close to 0.5) should have a weight close to 1.0
    high_entropy_correctness = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    high_entropy_weight = loss_fn.compute_kl_entropy_weight(high_entropy_correctness)
    assert high_entropy_weight == pytest.approx(1.0, abs=1e-2)

    # Low entropy (all correct) should have a low weight
    low_entropy_correctness = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    low_entropy_weight = loss_fn.compute_kl_entropy_weight(low_entropy_correctness)
    assert low_entropy_weight < 0.1

    # Low entropy (all incorrect) should also have a low weight
    all_incorrect_correctness = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    all_incorrect_weight = loss_fn.compute_kl_entropy_weight(all_incorrect_correctness)
    assert all_incorrect_weight < 0.1


def test_mgpo_loss_compute_advantages():
    loss_fn = MGPOLoss(lambda_param=4.0)
    rewards = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],  # High entropy
            [1.0, 1.0, 1.0, 0.0],  # Low entropy
        ],
        dtype=torch.float32,
    )
    entropy_weights = torch.tensor([0.9, 0.2], dtype=torch.float32)

    # Test without entropy weighting
    advantages_no_weight = loss_fn.compute_advantages(rewards)
    assert advantages_no_weight.shape == rewards.shape
    # Check that the mean of each row is close to 0
    assert torch.allclose(advantages_no_weight.mean(dim=1), torch.zeros(2), atol=1e-6)

    # Test with entropy weighting
    advantages_with_weight = loss_fn.compute_advantages(rewards, entropy_weights)
    assert advantages_with_weight.shape == rewards.shape
    # Check that the weighted advantages are different from the unweighted ones
    assert not torch.allclose(advantages_no_weight, advantages_with_weight)
    # Check that the second row (low entropy) is scaled down more
    assert torch.allclose(
        advantages_with_weight[0], advantages_no_weight[0] * entropy_weights[0]
    )
    assert torch.allclose(
        advantages_with_weight[1], advantages_no_weight[1] * entropy_weights[1]
    )
