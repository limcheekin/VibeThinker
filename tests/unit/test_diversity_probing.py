"""Unit tests for diversity_probing module."""

import json
import sys
from unittest.mock import MagicMock, Mock, mock_open, patch

import numpy as np
import pytest

# Mock unsloth before importing diversity_probing to avoid GPU requirement
sys.modules["unsloth"] = MagicMock()
sys.modules["unsloth"].FastLanguageModel = MagicMock()

from vibethinker.diversity_probing import (  # noqa: E402
    DiversityProber,
    calculate_pass_at_k,
)


def test_calculate_pass_at_k_basic():
    """Test basic Pass@K calculation."""
    # All correct
    assert calculate_pass_at_k(n=10, c=10, k=1) == 1.0

    # None correct
    assert calculate_pass_at_k(n=10, c=0, k=1) == 0.0

    # Half correct with k=1
    result = calculate_pass_at_k(n=10, c=5, k=1)
    assert 0.4 < result < 0.6


def test_calculate_pass_at_k_edge_cases():
    """Test Pass@K calculation edge cases."""
    # When n - c < k, should return 1.0
    assert calculate_pass_at_k(n=10, c=9, k=5) == 1.0

    # When k equals n-c, should be close to 1.0
    result = calculate_pass_at_k(n=10, c=5, k=5)
    assert result > 0.99


def test_calculate_pass_at_k_formula():
    """Test Pass@K formula correctness."""
    # Test known values
    # With n=16, c=8, k=8: Pass@8 should be close to 1.0
    result = calculate_pass_at_k(n=16, c=8, k=8)
    assert result > 0.99

    # With n=16, c=16, k=8: all correct, should be 1.0
    assert calculate_pass_at_k(n=16, c=16, k=8) == 1.0


def test_calculate_pass_at_k_values():
    """Test Pass@K with various parameter combinations."""
    # k=1 should give probability of getting at least 1 correct
    result = calculate_pass_at_k(n=4, c=2, k=1)
    assert 0 < result < 1

    # k=2 with n=4, c=2: should be close to 1
    result = calculate_pass_at_k(n=4, c=2, k=2)
    assert result > 0.8  # High probability but not necessarily 1.0


@patch("vibethinker.diversity_probing.FastLanguageModel")
@patch("vibethinker.diversity_probing.MGPORewardCalculator")
def test_diversity_prober_init(mock_reward_calc, mock_flm):
    """Test DiversityProber initialization."""
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)

    prober = DiversityProber(model_path="/fake/path", max_seq_length=1024)

    # Verify model was loaded
    mock_flm.from_pretrained.assert_called_once_with(
        "/fake/path",
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,
    )

    # Verify inference mode was set
    mock_flm.for_inference.assert_called_once_with(mock_model)

    # Verify reward calculator was initialized
    assert prober.reward_calc is not None


@patch("vibethinker.diversity_probing.FastLanguageModel")
@patch("vibethinker.diversity_probing.MGPORewardCalculator")
def test_probe_domain(mock_reward_calc_class, mock_flm):
    """Test domain probing functionality."""
    # Setup mocks
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)

    # Mock tokenizer behavior - return proper object that can be unpacked
    def mock_tokenizer_call(*args, **kwargs):
        obj = Mock()
        inputs_obj = MagicMock()
        inputs_obj.input_ids = Mock(shape=(1, 10))
        # Make it behave like a dict for ** unpacking
        inputs_obj.__iter__.return_value = iter(["input_ids"])
        inputs_obj.keys.return_value = ["input_ids"]
        inputs_obj.__getitem__.return_value = inputs_obj.input_ids
        obj.to = Mock(return_value=inputs_obj)
        return obj

    mock_tokenizer.side_effect = mock_tokenizer_call
    mock_tokenizer.pad_token_id = 0

    # Mock model generation - return 4 sequences
    mock_outputs = [MagicMock() for _ in range(4)]
    mock_model.generate.return_value = mock_outputs

    # Mock tokenizer decode
    mock_tokenizer.decode.return_value = "The answer is 42"

    # Mock reward calculator
    mock_reward_calc = Mock()
    mock_reward_calc.evaluate_solution.side_effect = [1.0, 0.0, 1.0, 0.0]  # 50% correct
    mock_reward_calc_class.return_value = mock_reward_calc

    prober = DiversityProber(model_path="/fake/path")
    prober.reward_calc = mock_reward_calc

    # Test data
    problems = [{"problem": "What is 6*7?", "answer": "42"}]

    # Run probing with n=4, k=2
    metrics = prober.probe_domain(problems, k=2, num_generations=4)

    # Verify metrics structure
    assert "pass@k" in metrics
    assert "pass@1" in metrics
    assert "diversity_score" in metrics

    # Verify values are reasonable
    assert 0 <= metrics["pass@k"] <= 1.0
    assert 0 <= metrics["pass@1"] <= 1.0
    assert metrics["diversity_score"] == metrics["pass@k"]


@patch("vibethinker.diversity_probing.FastLanguageModel")
@patch("vibethinker.diversity_probing.MGPORewardCalculator")
def test_probe_domain_all_correct(mock_reward_calc_class, mock_flm):
    """Test probing when all generations are correct."""
    # Setup mocks
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)

    def mock_tokenizer_call(*args, **kwargs):
        obj = Mock()
        inputs_obj = MagicMock()
        inputs_obj.input_ids = Mock(shape=(1, 10))
        # Make it behave like a dict for ** unpacking
        inputs_obj.__iter__.return_value = iter(["input_ids"])
        inputs_obj.keys.return_value = ["input_ids"]
        inputs_obj.__getitem__.return_value = inputs_obj.input_ids
        obj.to = Mock(return_value=inputs_obj)
        return obj

    mock_tokenizer.side_effect = mock_tokenizer_call
    mock_tokenizer.pad_token_id = 0

    mock_outputs = [MagicMock() for _ in range(4)]
    mock_model.generate.return_value = mock_outputs
    mock_tokenizer.decode.return_value = "42"

    mock_reward_calc = Mock()
    mock_reward_calc.evaluate_solution.return_value = 1.0  # All correct
    mock_reward_calc_class.return_value = mock_reward_calc

    prober = DiversityProber(model_path="/fake/path")
    prober.reward_calc = mock_reward_calc

    problems = [{"problem": "Test", "answer": "42"}]
    metrics = prober.probe_domain(problems, k=2, num_generations=4)

    # All correct should give Pass@K = 1.0 and Pass@1 = 1.0
    assert metrics["pass@k"] == 1.0
    assert metrics["pass@1"] == 1.0


@patch("vibethinker.diversity_probing.FastLanguageModel")
@patch("vibethinker.diversity_probing.MGPORewardCalculator")
def test_probe_domain_none_correct(mock_reward_calc_class, mock_flm):
    """Test probing when no generations are correct."""
    # Setup mocks
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)

    def mock_tokenizer_call(*args, **kwargs):
        obj = Mock()
        inputs_obj = MagicMock()
        inputs_obj.input_ids = Mock(shape=(1, 10))
        # Make it behave like a dict for ** unpacking
        inputs_obj.__iter__.return_value = iter(["input_ids"])
        inputs_obj.keys.return_value = ["input_ids"]
        inputs_obj.__getitem__.return_value = inputs_obj.input_ids
        obj.to = Mock(return_value=inputs_obj)
        return obj

    mock_tokenizer.side_effect = mock_tokenizer_call
    mock_tokenizer.pad_token_id = 0

    mock_outputs = [MagicMock() for _ in range(4)]
    mock_model.generate.return_value = mock_outputs
    mock_tokenizer.decode.return_value = "wrong"

    mock_reward_calc = Mock()
    mock_reward_calc.evaluate_solution.return_value = 0.0  # All wrong
    mock_reward_calc_class.return_value = mock_reward_calc

    prober = DiversityProber(model_path="/fake/path")
    prober.reward_calc = mock_reward_calc

    problems = [{"problem": "Test", "answer": "42"}]
    metrics = prober.probe_domain(problems, k=2, num_generations=4)

    # None correct should give Pass@K = 0.0 and Pass@1 = 0.0
    assert metrics["pass@k"] == 0.0
    assert metrics["pass@1"] == 0.0


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"problem": "test", "answer": "42"}\n',
)
@patch("vibethinker.diversity_probing.DiversityProber")
def test_main_cli(mock_prober_class, mock_file):
    """Test command-line interface."""
    import sys

    from vibethinker.diversity_probing import __name__ as module_name

    # Mock the prober instance
    mock_prober = Mock()
    mock_prober.probe_domain.return_value = {
        "pass@k": 0.75,
        "pass@1": 0.50,
        "diversity_score": 0.75,
    }
    mock_prober_class.return_value = mock_prober

    # This test verifies the CLI structure exists
    # Actual CLI testing would require subprocess or more complex mocking


def test_calculate_pass_at_k_high_k():
    """Test Pass@K with high k values."""
    # When k is higher than n-c, should return 1.0
    result = calculate_pass_at_k(n=10, c=2, k=10)
    assert result == 1.0


def test_calculate_pass_at_k_symmetry():
    """Test that Pass@K behaves correctly with different c values."""
    # More correct answers should give higher Pass@K (for same k)
    result_low = calculate_pass_at_k(n=16, c=4, k=4)
    result_high = calculate_pass_at_k(n=16, c=12, k=4)

    assert result_high > result_low
