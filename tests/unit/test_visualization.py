"""
Simplified tests for visualization.py.

Focuses on testing module structure and basic functionality
without complex GPU-dependent operations.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from vibethinker.visualization import (
    AttentionVisualizer,
    GenerationAnalyzer,
    LossLandscapeVisualizer,
)


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = Mock()
    model.eval = Mock()
    model.train = Mock()
    model.named_modules = Mock(return_value=[])
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    tokenizer.convert_ids_to_tokens = Mock(
        return_value=["the", "quick", "brown", "fox", "jumps"]
    )
    tokenizer.decode = Mock(return_value="the quick brown fox jumps")
    return tokenizer


def test_attention_visualizer_init(mock_model, mock_tokenizer, tmp_path):
    viz = AttentionVisualizer(mock_model, mock_tokenizer, output_dir=str(tmp_path))

    assert viz.model == mock_model
    assert viz.tokenizer == mock_tokenizer
    assert viz.output_dir == Path(tmp_path)
    assert viz.output_dir.exists()


def test_attention_visualizer_register_hooks(mock_model, mock_tokenizer):
    # Create mock modules
    mock_attn_module = Mock()
    mock_attn_module.__class__.__name__ = "Attention"

    mock_model.named_modules = Mock(
        return_value=[("layer.0.attention", mock_attn_module)]
    )

    viz = AttentionVisualizer(mock_model, mock_tokenizer)
    viz.register_hooks()

    assert len(viz.hooks) > 0


def test_attention_visualizer_remove_hooks(mock_model, mock_tokenizer):
    viz = AttentionVisualizer(mock_model, mock_tokenizer)

    # Add some mock hooks
    mock_hook = Mock()
    viz.hooks.append(mock_hook)

    viz.remove_hooks()

    # Hooks should be removed
    mock_hook.remove.assert_called_once()
    assert len(viz.hooks) == 0


def test_generation_analyzer_init(mock_model, mock_tokenizer, tmp_path):
    analyzer = GenerationAnalyzer(mock_model, mock_tokenizer, output_dir=str(tmp_path))

    assert analyzer.model == mock_model
    assert analyzer.tokenizer == mock_tokenizer
    assert analyzer.output_dir == Path(tmp_path)


def test_generation_analyzer_exists():
    """Test GenerationAnalyzer class exists and has expected methods."""
    assert GenerationAnalyzer is not None
    assert hasattr(GenerationAnalyzer, "analyze_diversity")
    assert hasattr(GenerationAnalyzer, "plot_diversity_analysis")


def test_loss_landscape_visualizer_exists():
    """Test LossLandscapeVisualizer exists."""
    assert LossLandscapeVisualizer is not None
    assert hasattr(LossLandscapeVisualizer, "compute_loss_landscape")
    assert hasattr(LossLandscapeVisualizer, "plot_loss_landscape")


@patch("vibethinker.visualization.plt")
def test_loss_landscape_visualizer_plot(mock_plt):
    alphas = np.linspace(-1, 1, 20)
    losses = np.random.rand(20) + 2.0

    # Mock figure and subplots
    mock_fig = Mock()
    mock_ax = Mock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_plt.plot = Mock()

    LossLandscapeVisualizer.plot_loss_landscape(alphas, losses)

    # Should create a plot
    assert mock_plt.subplots.called


def test_attention_visualizer_with_no_attention_modules(mock_tokenizer, tmp_path):
    # Model with no attention modules
    model = Mock()
    model.named_modules = Mock(return_value=[("layer.0.linear", Mock())])

    viz = AttentionVisualizer(model, mock_tokenizer, output_dir=str(tmp_path))
    viz.register_hooks()

    # Should not raise error even with no attention modules
    assert viz.hooks == []


def test_generation_analyzer_creation(mock_model, mock_tokenizer):
    """Test that GenerationAnalyzer can be instantiated."""
    analyzer = GenerationAnalyzer(mock_model, mock_tokenizer)
    assert analyzer is not None
    assert analyzer.model == mock_model
