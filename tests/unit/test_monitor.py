import numpy as np
import pytest

from vibethinker.monitor import CostCalculator, MGPORewardCalculator


def test_cost_calculator():
    # Test with default values (H100)
    calculator = CostCalculator(gpu_type="H100", region="us-west")
    cost = calculator.compute_cost(elapsed_seconds=3600)
    assert cost["compute_cost_usd"] == pytest.approx(4.0)
    assert cost["energy_cost_usd"] > 0
    assert cost["total_usd"] > 4.0

    # Test with a different GPU
    calculator = CostCalculator(gpu_type="L4", region="eu")
    cost = calculator.compute_cost(elapsed_seconds=7200)
    assert cost["compute_cost_usd"] == pytest.approx(0.7)
    assert cost["energy_cost_usd"] > 0
    assert cost["total_usd"] > 0.7


def test_mgpo_reward_calculator_symbolic_evaluation():
    reward_calc = MGPORewardCalculator(lambda_param=4.0)

    # Test cases for symbolic evaluation
    test_cases = [
        ("2*x + 3 = 7\nx = 2", "x = 2", 1.0),
        ("The answer is 42", "42", 1.0),
        ("x = 5", "x = 3", 0.0),
        ("x**2 = 4\nx=2", "x = 2", 1.0),
        ("sqrt(16) = 4", "4", 1.0),
        ("The final answer is y=2x+1", "y=2*x+1", 1.0),
    ]

    for gen, ref, expected in test_cases:
        reward = reward_calc.evaluate_solution(gen, ref)
        assert reward == expected


def test_mgpo_reward_calculator_entropy_weight():
    reward_calc = MGPORewardCalculator(lambda_param=4.0)

    # High entropy (close to 0.5) should have a weight close to 1.0
    high_entropy_correctness = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    high_entropy_weight = reward_calc.compute_kl_entropy_weight(
        high_entropy_correctness
    )
    assert high_entropy_weight == pytest.approx(1.0, abs=1e-2)

    # Low entropy (all correct) should have a low weight
    low_entropy_correctness = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    low_entropy_weight = reward_calc.compute_kl_entropy_weight(low_entropy_correctness)
    assert low_entropy_weight < 0.1

    # Low entropy (all incorrect) should also have a low weight
    all_incorrect_correctness = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    all_incorrect_weight = reward_calc.compute_kl_entropy_weight(
        all_incorrect_correctness
    )
    assert all_incorrect_weight < 0.1
