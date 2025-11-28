import unittest
from unittest.mock import MagicMock, patch

import torch

from vibethinker.inference_optimize import OptimizedInference


class TestInferenceOptimize(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.pad_token_id = 0

    def test_generate_optimized(self):
        # Mock the input and output tensors
        self.mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        self.mock_tokenizer.decode.return_value = "This is a test sentence."

        optimizer = OptimizedInference(self.mock_model, self.mock_tokenizer)
        result = optimizer.generate_optimized("test prompt")

        self.mock_model.generate.assert_called_once()
        self.assertEqual(result, "This is a test sentence.")

    def test_batch_generate(self):
        # Mock the input and output tensors
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])
        }
        self.mock_model.generate.return_value = torch.tensor(
            [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]]
        )
        self.mock_tokenizer.decode.side_effect = [
            "This is the first sentence.",
            "This is the second sentence.",
        ]

        optimizer = OptimizedInference(self.mock_model, self.mock_tokenizer)
        results = optimizer.batch_generate(["prompt1", "prompt2"])

        self.mock_model.generate.assert_called_once()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], "This is the first sentence.")
        self.assertEqual(results[1], "This is the second sentence.")


if __name__ == "__main__":
    unittest.main()
