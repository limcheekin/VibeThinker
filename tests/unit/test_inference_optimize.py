from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from vibethinker.inference_optimize import OptimizedInference


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = Mock()
    model.eval = Mock()
    model.generate = Mock()
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    tokenizer.decode = Mock(return_value="Generated response")
    return tokenizer


def test_optimized_inference_init(mock_model, mock_tokenizer):
    with patch("vibethinker.inference_optimize.torch.compile", return_value=mock_model):
        inference = OptimizedInference(mock_model, mock_tokenizer, device="cuda")

        assert inference.model is not None
        assert inference.tokenizer == mock_tokenizer
        assert inference.device == "cuda"
        # Model should be set to eval mode
        mock_model.eval.assert_called_once()


def test_optimized_inference_init_without_compile():
    mock_model = Mock()
    mock_model.eval = Mock()
    mock_tokenizer = Mock()

    # Test when torch.compile doesn't exist
    with patch("vibethinker.inference_optimize.torch", spec=["no_grad", "autocast"]):
        inference = OptimizedInference(mock_model, mock_tokenizer, device="cpu")
        assert inference.device == "cpu"


def test_generate_optimized():
    mock_model = Mock()
    mock_model.eval = Mock()
    mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))

    mock_tokenizer = Mock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_tokenizer.decode = Mock(return_value="This is the generated output")

    with patch("vibethinker.inference_optimize.torch.compile", return_value=mock_model):
        with patch("vibethinker.inference_optimize.torch.no_grad"):
            with patch("vibethinker.inference_optimize.torch.autocast"):
                inference = OptimizedInference(
                    mock_model, mock_tokenizer, device="cuda"
                )

                # Mock the to() method
                mock_inputs = Mock()
                mock_inputs.__getitem__ = Mock(return_value=torch.tensor([[1, 2, 3]]))
                mock_tokenizer.return_value = mock_inputs
                mock_inputs.to = Mock(
                    return_value={"input_ids": torch.tensor([[1, 2, 3]])}
                )

                result = inference.generate_optimized(
                    prompt="Test prompt", max_length=512, temperature=0.7
                )

                assert isinstance(result, str)
                assert result == "This is the generated output"


def test_generate_optimized_parameters():
    mock_model = Mock()
    mock_model.eval = Mock()
    mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))

    mock_tokenizer = Mock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode = Mock(return_value="output")

    mock_inputs = Mock()
    mock_inputs.to = Mock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
    mock_tokenizer.return_value = mock_inputs

    with patch("vibethinker.inference_optimize.torch.compile", return_value=mock_model):
        with patch("vibethinker.inference_optimize.torch.no_grad"):
            with patch("vibethinker.inference_optimize.torch.autocast"):
                inference = OptimizedInference(mock_model, mock_tokenizer)

                result = inference.generate_optimized(
                    prompt="Test",
                    max_length=256,
                    num_beams=4,
                    temperature=0.5,
                    use_cache=False,
                )

                # Verify generate was called with correct params
                call_args = mock_model.generate.call_args
                assert call_args is not None


def test_batch_generate():
    mock_model = Mock()
    mock_model.eval = Mock()
    mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3], [4, 5, 6]]))

    mock_tokenizer = Mock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode = Mock(side_effect=["Output 1", "Output 2"])

    # Mock the tokenizer call
    mock_inputs = Mock()
    mock_inputs.to = Mock(
        return_value={"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}
    )
    mock_tokenizer.return_value = mock_inputs

    with patch("vibethinker.inference_optimize.torch.compile", return_value=mock_model):
        with patch("vibethinker.inference_optimize.torch.no_grad"):
            with patch("vibethinker.inference_optimize.torch.autocast"):
                inference = OptimizedInference(mock_model, mock_tokenizer)

                prompts = ["Test 1", "Test 2"]
                results = inference.batch_generate(prompts, max_length=128)

                assert len(results) == 2
                assert results[0] == "Output 1"
                assert results[1] == "Output 2"


def test_batch_generate_with_kwargs():
    mock_model = Mock()
    mock_model.eval = Mock()
    mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))

    mock_tokenizer = Mock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode = Mock(return_value="output")

    mock_inputs = Mock()
    mock_inputs.to = Mock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
    mock_tokenizer.return_value = mock_inputs

    with patch("vibethinker.inference_optimize.torch.compile", return_value=mock_model):
        with patch("vibethinker.inference_optimize.torch.no_grad"):
            with patch("vibethinker.inference_optimize.torch.autocast"):
                inference = OptimizedInference(mock_model, mock_tokenizer)

                results = inference.batch_generate(
                    prompts=["Test"], max_length=256, temperature=0.8, top_p=0.95
                )

                assert len(results) == 1


def test_generate_with_temperature_zero():
    """Test that do_sample is False when temperature is 0."""
    mock_model = Mock()
    mock_model.eval = Mock()
    mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))

    mock_tokenizer = Mock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode = Mock(return_value="output")

    mock_inputs = Mock()
    mock_inputs.to = Mock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
    mock_tokenizer.return_value = mock_inputs

    with patch("vibethinker.inference_optimize.torch.compile", return_value=mock_model):
        with patch("vibethinker.inference_optimize.torch.no_grad"):
            with patch("vibethinker.inference_optimize.torch.autocast"):
                inference = OptimizedInference(mock_model, mock_tokenizer)

                result = inference.generate_optimized(prompt="Test", temperature=0.0)

                # Verify do_sample should be False
                call_kwargs = mock_model.generate.call_args.kwargs
                assert call_kwargs["do_sample"] is False


def test_generate_with_positive_temperature():
    """Test that do_sample is True when temperature > 0."""
    mock_model = Mock()
    mock_model.eval = Mock()
    mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))

    mock_tokenizer = Mock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode = Mock(return_value="output")

    mock_inputs = Mock()
    mock_inputs.to = Mock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
    mock_tokenizer.return_value = mock_inputs

    with patch("vibethinker.inference_optimize.torch.compile", return_value=mock_model):
        with patch("vibethinker.inference_optimize.torch.no_grad"):
            with patch("vibethinker.inference_optimize.torch.autocast"):
                inference = OptimizedInference(mock_model, mock_tokenizer)

                result = inference.generate_optimized(prompt="Test", temperature=0.7)

                # Verify do_sample should be True
                call_kwargs = mock_model.generate.call_args.kwargs
                assert call_kwargs["do_sample"] is True


def test_device_parameter():
    """Test different device settings."""
    mock_model = Mock()
    mock_model.eval = Mock()
    mock_tokenizer = Mock()

    for device in ["cuda", "cpu", "cuda:0", "cuda:1"]:
        with patch(
            "vibethinker.inference_optimize.torch.compile", return_value=mock_model
        ):
            inference = OptimizedInference(mock_model, mock_tokenizer, device=device)
            assert inference.device == device
