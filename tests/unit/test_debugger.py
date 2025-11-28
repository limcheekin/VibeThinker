import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from vibethinker.debugger import PerformanceInspector, TrainingDebugger


class TestDebugger(unittest.TestCase):
    def setUp(self):
        self.debugger = TrainingDebugger(log_dir="test_debug_logs")
        self.model = torch.nn.Linear(10, 2)

    def test_check_gradient_health(self):
        # Mock gradients
        self.model.weight.grad = torch.randn(2, 10)
        self.model.bias.grad = torch.randn(2)

        stats = self.debugger.check_gradient_health(self.model)
        self.assertIn("total_norm", stats)
        self.assertIn("layer_stats", stats)
        self.assertIn("issues", stats)
        self.assertGreaterEqual(stats["total_norm"], 0)

    def test_check_loss_sanity(self):
        self.assertTrue(self.debugger.check_loss_sanity(torch.tensor(1.0), 1))
        self.assertFalse(self.debugger.check_loss_sanity(torch.tensor(float("nan")), 2))

    def test_check_activation_stats(self):
        # Mock last output
        self.model._last_output = torch.randn(1, 2)

        stats = self.debugger.check_activation_stats(self.model)
        self.assertIn("activation_stats", stats)
        self.assertIn("dead_neuron_count", stats)
        self.assertEqual(stats["dead_neuron_count"], 0)

    def test_debug_generation(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mocks to return expected outputs
        mock_tokenizer.return_value = {"input_ids": torch.randint(0, 100, (1, 5))}
        mock_model.generate.return_value.sequences = [torch.randint(0, 100, (1, 10))]
        mock_tokenizer.decode.return_value = "This is a test sentence."

        result = self.debugger.debug_generation(mock_model, mock_tokenizer, "test")
        self.assertIn("generated_text", result)
        self.assertEqual(result["generated_text"], "This is a test sentence.")

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.reset_peak_memory_stats")
    @patch("torch.cuda.max_memory_allocated")
    @patch("torch.profiler.profile")
    @patch("torch.randint")
    def test_profile_gpu_memory(
        self,
        mock_randint,
        mock_profile,
        mock_max_memory_allocated,
        mock_reset_peak_memory_stats,
        mock_empty_cache,
        mock_is_available,
    ):
        mock_model = MagicMock()
        mock_randint.return_value = torch.randint(0, 50000, (4, 1024))
        mock_max_memory_allocated.return_value = 1024**3
        result = PerformanceInspector.profile_gpu_memory(mock_model)
        self.assertEqual(result["peak_memory_gb"], 1.0)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("time.time")
    @patch("torch.randint")
    def test_benchmark_throughput(
        self, mock_randint, mock_time, mock_synchronize, mock_is_available
    ):
        mock_model = MagicMock()
        mock_randint.return_value = torch.randint(0, 50000, (4, 512))
        mock_time.side_effect = [0, 1]  # Mock start and end times
        result = PerformanceInspector.benchmark_throughput(mock_model, MagicMock())
        self.assertGreater(result["throughput_tokens_per_sec"], 0)


if __name__ == "__main__":
    unittest.main()
