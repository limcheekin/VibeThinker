"""
Tests for train_complete.py - using mocking to avoid unsloth dependency.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


# Mock unsloth before importing the module
@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock unsloth and other heavy dependencies."""
    with patch.dict(sys.modules, {
        "unsloth": MagicMock(),
        "unsloth.FastLanguageModel": MagicMock(),
        "torch": MagicMock(),
        "transformers": MagicMock(),
        "datasets": MagicMock(),
        "vibethinker.cost_analysis": MagicMock(),
        "vibethinker.debugger": MagicMock(),
        "vibethinker.grpo_custom": MagicMock(),
        "vibethinker.monitor": MagicMock(),
        "vibethinker.visualization": MagicMock(),
    }):
        # Ensure train_complete is re-imported if it was already imported
        if "vibethinker.train_complete" in sys.modules:
            del sys.modules["vibethinker.train_complete"]
        yield


def test_module_imports():
    """Test that all required modules can be imported."""
    from vibethinker import train_complete
    assert train_complete is not None


def test_function_exists():
    """Test that train_signal_phase_complete function exists."""
    from vibethinker.train_complete import train_signal_phase_complete
    assert callable(train_signal_phase_complete)


def test_function_signature():
    """Test function has expected parameters."""
    import inspect
    from vibethinker.train_complete import train_signal_phase_complete

    sig = inspect.signature(train_signal_phase_complete)
    params = list(sig.parameters.keys())

    assert "spectrum_model_path" in params
    assert "train_dataset" in params
    assert "val_dataset" in params
    assert "tokenizer" in params


def test_dependencies_imported():
    """Test that dependencies are properly imported in the module."""
    from vibethinker import train_complete

    # Check module has expected attributes
    assert hasattr(train_complete, "json")
    assert hasattr(train_complete, "os")


@patch("vibethinker.train_complete.FastLanguageModel")
@patch("vibethinker.train_complete.CostAnalyzer")
@patch("vibethinker.train_complete.TrainingMonitor")
@patch("vibethinker.train_complete.TrainingDebugger")
@patch("vibethinker.train_complete.GenerationAnalyzer")
@patch("vibethinker.train_complete.AttentionVisualizer")
@patch("vibethinker.train_complete.PerformanceInspector")
@patch("vibethinker.train_complete.MGPORewardCalculator")
@patch("vibethinker.train_complete.MGPOTrainerWithEntropyWeighting")
def test_train_signal_phase_complete_mocked(
    mock_trainer,
    mock_reward,
    mock_inspector,
    mock_attn_viz,
    mock_gen_analyzer,
    mock_debugger,
    mock_monitor,
    mock_cost,
    mock_flm,
):
    """Test the training function with mocked dependencies."""
    from vibethinker.train_complete import train_signal_phase_complete

    # Setup mocks
    mock_flm.from_pretrained.return_value = (MagicMock(), MagicMock())
    
    # Mock inspector return values
    mock_inspector.return_value.profile_gpu_memory.return_value = {"peak_memory_gb": 10.5}
    mock_inspector.return_value.benchmark_throughput.return_value = {"throughput_tokens_per_sec": 100}
    
    # Mock trainer return values
    mock_trainer.return_value.training_step.return_value = {
        "loss": 0.5,
        "reward_mean": 1.0,
        "entropy_weight_mean": 0.1
    }

    # Mock debugger return values
    mock_debugger.return_value.check_gradient_health.return_value = {"issues": [], "total_norm": 0.1}
    mock_debugger.return_value.check_loss_sanity.return_value = True

    # Mock dataset
    mock_dataset = MagicMock()
    mock_dataset.batch.return_value = []
    
    # Mock tokenizer
    mock_tokenizer = MagicMock()

    # Run function
    train_signal_phase_complete(
        spectrum_model_path="dummy/path",
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        tokenizer=mock_tokenizer,
        output_dir="outputs/test_output",
        max_steps=1
    )

    # Verify calls
    mock_cost.assert_called_once()
    mock_flm.from_pretrained.assert_called_once()
    mock_monitor.assert_called_once()
    mock_debugger.assert_called_once()
    mock_trainer.assert_called_once()
