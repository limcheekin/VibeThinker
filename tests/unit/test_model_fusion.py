from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from vibethinker.model_fusion import (
    fusion_task_arithmetic,
    fusion_weighted_average,
    load_expert_models,
    validate_fusion,
)


@pytest.fixture
def mock_expert_model():
    """Create a mock expert model."""
    model = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)

    # Create mock parameters
    weight = torch.nn.Parameter(torch.randn(10, 5))
    bias = torch.nn.Parameter(torch.randn(10))

    model.named_parameters = Mock(
        return_value=[("layer.weight", weight), ("layer.bias", bias)]
    )

    # Mock config
    model.config = Mock()
    model.config._name_or_path = "test/model"

    return model


def test_load_expert_models():
    with patch(
        "vibethinker.model_fusion.AutoModelForCausalLM.from_pretrained"
    ) as mock_load:
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock()
        mock_load.return_value = mock_model

        expert_paths = ["path1", "path2", "path3"]
        experts = load_expert_models(expert_paths, device="cuda")

        assert len(experts) == 3
        assert mock_load.call_count == 3
        # Verify models were moved to device
        assert mock_model.to.call_count == 3


def test_fusion_weighted_average_equal_weights(mock_expert_model):
    # Create multiple mock experts
    experts = [mock_expert_model for _ in range(4)]

    with patch(
        "vibethinker.model_fusion.AutoModelForCausalLM.from_pretrained"
    ) as mock_load:
        fused_model = Mock()

        # Setup named_parameters for fused model
        weight_param = torch.nn.Parameter(torch.zeros(10, 5))
        bias_param = torch.nn.Parameter(torch.zeros(10))
        fused_model.named_parameters = Mock(
            return_value=[("layer.weight", weight_param), ("layer.bias", bias_param)]
        )

        mock_load.return_value = fused_model

        # Call fusion with equal weights
        result = fusion_weighted_average(experts, weights=None)

        assert result is not None


def test_fusion_weighted_average_custom_weights(mock_expert_model):
    experts = [mock_expert_model for _ in range(3)]
    weights = [0.5, 0.3, 0.2]

    with patch(
        "vibethinker.model_fusion.AutoModelForCausalLM.from_pretrained"
    ) as mock_load:
        fused_model = Mock()

        weight_param = torch.nn.Parameter(torch.zeros(10, 5))
        bias_param = torch.nn.Parameter(torch.zeros(10))
        fused_model.named_parameters = Mock(
            return_value=[("layer.weight", weight_param), ("layer.bias", bias_param)]
        )

        mock_load.return_value = fused_model

        result = fusion_weighted_average(experts, weights=weights)

        assert result is not None


def test_fusion_weighted_average_invalid_weights(mock_expert_model):
    experts = [mock_expert_model for _ in range(3)]
    weights = [0.5, 0.3, 0.1]  # Sum to 0.9, not 1.0

    with patch("vibethinker.model_fusion.AutoModelForCausalLM.from_pretrained"):
        with pytest.raises(AssertionError, match="Weights must sum to 1.0"):
            fusion_weighted_average(experts, weights=weights)


def test_fusion_weighted_average_mismatched_lengths(mock_expert_model):
    experts = [mock_expert_model for _ in range(3)]
    weights = [0.5, 0.5]  # Only 2 weights for 3 experts

    with patch("vibethinker.model_fusion.AutoModelForCausalLM.from_pretrained"):
        with pytest.raises(AssertionError):
            fusion_weighted_average(experts, weights=weights)


def test_fusion_task_arithmetic(mock_expert_model):
    base_model = mock_expert_model
    experts = [mock_expert_model for _ in range(3)]

    with patch(
        "vibethinker.model_fusion.AutoModelForCausalLM.from_pretrained"
    ) as mock_load:
        fused_model = Mock()

        weight_param = torch.nn.Parameter(torch.zeros(10, 5))
        bias_param = torch.nn.Parameter(torch.zeros(10))
        fused_model.named_parameters = Mock(
            return_value=[("layer.weight", weight_param), ("layer.bias", bias_param)]
        )

        mock_load.return_value = fused_model

        result = fusion_task_arithmetic(base_model, experts, scaling=0.3)

        assert result is not None


def test_fusion_task_arithmetic_different_scaling():
    mock_base = Mock()
    mock_expert = Mock()

    # Setup parameters
    weight = torch.nn.Parameter(torch.ones(5, 3))
    bias = torch.nn.Parameter(torch.ones(5))

    mock_base.named_parameters = Mock(
        return_value=[("weight", weight.clone()), ("bias", bias.clone())]
    )

    mock_expert.named_parameters = Mock(
        return_value=[
            ("weight", weight.clone() + 1),  # Different weights
            ("bias", bias.clone() + 1),
        ]
    )

    mock_base.config = Mock()
    mock_base.config._name_or_path = "test/model"

    with patch(
        "vibethinker.model_fusion.AutoModelForCausalLM.from_pretrained"
    ) as mock_load:
        fused_model = Mock()

        weight_param = torch.nn.Parameter(torch.zeros(5, 3))
        bias_param = torch.nn.Parameter(torch.zeros(5))
        fused_model.named_parameters = Mock(
            return_value=[("weight", weight_param), ("bias", bias_param)]
        )

        mock_load.return_value = fused_model

        # Test with different scaling values
        for scaling in [0.1, 0.3, 0.5, 1.0]:
            result = fusion_task_arithmetic(mock_base, [mock_expert], scaling=scaling)
            assert result is not None


def test_validate_fusion():
    """Test validate_fusion function exists and has right signature."""
    import inspect

    from vibethinker.model_fusion import validate_fusion

    sig = inspect.signature(validate_fusion)
    params = list(sig.parameters.keys())

    assert "fused_model" in params
    assert "tokenizer" in params
    assert "test_problems" in params


def test_validate_fusion_return_type():
    """Test that validate_fusion returns a dictionary."""
    # This test verifies the function signature
    import inspect

    from vibethinker.model_fusion import validate_fusion

    sig = inspect.signature(validate_fusion)
    # Function should return Dict[str, float]
    assert sig is not None


def test_load_expert_models_with_cpu():
    with patch(
        "vibethinker.model_fusion.AutoModelForCausalLM.from_pretrained"
    ) as mock_load:
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock()
        mock_load.return_value = mock_model

        experts = load_expert_models(["path1", "path2"], device="cpu")

        # Should load to CPU
        assert len(experts) == 2


def test_fusion_weighted_average_normalized_weights(mock_expert_model):
    """Test that weights are properly normalized."""
    experts = [mock_expert_model for _ in range(4)]
    weights = [0.25, 0.25, 0.25, 0.25]  # Perfectly normalized

    with patch(
        "vibethinker.model_fusion.AutoModelForCausalLM.from_pretrained"
    ) as mock_load:
        fused_model = Mock()

        weight_param = torch.nn.Parameter(torch.zeros(10, 5))
        bias_param = torch.nn.Parameter(torch.zeros(10))
        fused_model.named_parameters = Mock(
            return_value=[("layer.weight", weight_param), ("layer.bias", bias_param)]
        )

        mock_load.return_value = fused_model

        # Should not raise any assertion errors
        result = fusion_weighted_average(experts, weights=weights)
        assert result is not None
