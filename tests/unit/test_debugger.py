from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from vibethinker.debugger import PerformanceInspector, TrainingDebugger


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)
    )
    return model


@pytest.fixture
def model_with_gradients(simple_model):
    """Create a model with gradients."""
    # Create dummy input and target
    x = torch.randn(2, 10)
    y = torch.randint(0, 2, (2,))

    # Forward pass
    output = simple_model(x)
    loss = torch.nn.functional.cross_entropy(output, y)

    # Backward pass to create gradients
    loss.backward()

    return simple_model


def test_training_debugger_init(tmp_path):
    debugger = TrainingDebugger(log_dir=str(tmp_path / "debug_logs"))
    assert debugger.log_dir.exists()
    assert debugger.prev_loss == 0.0


def test_check_gradient_health(model_with_gradients):
    debugger = TrainingDebugger()
    grad_stats = debugger.check_gradient_health(model_with_gradients)

    assert "total_norm" in grad_stats
    assert "layer_stats" in grad_stats
    assert "issues" in grad_stats
    assert grad_stats["total_norm"] > 0


def test_check_gradient_health_no_gradients(simple_model):
    debugger = TrainingDebugger()
    grad_stats = debugger.check_gradient_health(simple_model)

    assert grad_stats["total_norm"] == 0.0
    assert len(grad_stats["layer_stats"]) == 0


def test_check_loss_sanity_normal():
    debugger = TrainingDebugger()
    loss = torch.tensor(2.5)
    result = debugger.check_loss_sanity(loss, step=1)
    assert result is True


def test_check_loss_sanity_nan():
    debugger = TrainingDebugger()
    loss = torch.tensor(float("nan"))
    result = debugger.check_loss_sanity(loss, step=1)
    assert result is False


def test_check_loss_sanity_inf():
    debugger = TrainingDebugger()
    loss = torch.tensor(float("inf"))
    result = debugger.check_loss_sanity(loss, step=1)
    assert result is False


def test_check_loss_sanity_very_large():
    debugger = TrainingDebugger()
    loss = torch.tensor(150.0)
    result = debugger.check_loss_sanity(loss, step=1)
    # Should still return True but warn
    assert result is True


def test_check_activation_stats(simple_model):
    debugger = TrainingDebugger()

    # Add dummy outputs to Linear layers
    for module in simple_model.modules():
        if isinstance(module, torch.nn.Linear):
            module._last_output = torch.randn(2, module.out_features)

    stats = debugger.check_activation_stats(simple_model)

    assert "activation_stats" in stats
    assert "dead_neuron_count" in stats


def test_check_activation_stats_dead_neurons(simple_model):
    debugger = TrainingDebugger()

    # Create dead neuron scenario
    for module in simple_model.modules():
        if isinstance(module, torch.nn.Linear):
            module._last_output = torch.zeros(2, module.out_features)

    stats = debugger.check_activation_stats(simple_model)
    assert stats["dead_neuron_count"] > 0


def test_performance_inspector_exists():
    """Test that PerformanceInspector class exists."""
    assert PerformanceInspector is not None


def test_performance_inspector_has_profile_method():
    """Test that PerformanceInspector has expected methods."""
    assert hasattr(PerformanceInspector, "profile_gpu_memory")
    assert hasattr(PerformanceInspector, "benchmark_throughput")


def test_debug_generation():
    debugger = TrainingDebugger()

    # Create mock model and tokenizer
    mock_model = Mock()
    mock_tokenizer = Mock()

    # Setup tokenizer mock
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    # Setup model generate mock
    mock_output = Mock()
    mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model.generate.return_value = mock_output
    mock_model.eval = Mock()

    # Setup decode mock
    mock_tokenizer.decode.return_value = "This is a test output with multiple words"

    result = debugger.debug_generation(mock_model, mock_tokenizer, "Test prompt")

    assert "prompt" in result
    assert "generated_text" in result
    assert "length_tokens" in result
    assert "issues" in result
    assert result["prompt"] == "Test prompt"


def test_debug_generation_repetition():
    debugger = TrainingDebugger()

    mock_model = Mock()
    mock_model.eval = Mock()
    mock_tokenizer = Mock()

    # Setup mocks
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_output = Mock()
    mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model.generate.return_value = mock_output

    # Return repetitive text
    mock_tokenizer.decode.return_value = "word word word word word word"

    result = debugger.debug_generation(mock_model, mock_tokenizer, "Test")

    # Should detect high repetition
    assert any("repetition" in issue.lower() for issue in result["issues"])


def test_gradient_health_with_large_norm():
    debugger = TrainingDebugger()

    # Create model with very large gradients
    model = torch.nn.Linear(10, 5)
    model.weight.grad = torch.ones_like(model.weight) * 100
    model.bias.grad = torch.ones_like(model.bias) * 100

    grad_stats = debugger.check_gradient_health(model)

    # Should warn about large gradient norm
    assert any("large gradient" in issue.lower() for issue in grad_stats["issues"])


def test_gradient_health_with_small_norm():
    debugger = TrainingDebugger()

    # Create model with very small gradients
    model = torch.nn.Linear(10, 5)
    model.weight.grad = torch.ones_like(model.weight) * 1e-8
    model.bias.grad = torch.ones_like(model.bias) * 1e-8

    grad_stats = debugger.check_gradient_health(model)

    # Should warn about small gradient norm
    assert any("small gradient" in issue.lower() for issue in grad_stats["issues"])
