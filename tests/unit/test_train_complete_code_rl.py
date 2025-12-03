"""
Unit tests for Code RL interface fix in train_complete.py
"""

import pytest
from datasets import Dataset

from vibethinker.train_complete import validate_dataset_format


class TestCodeRLInterface:
    """Test Code RL data loading and interface changes."""

    def test_validate_math_dataset_valid(self):
        """Test validation passes for valid math dataset."""
        dataset = Dataset.from_dict(
            {
                "problem": ["Solve x + 2 = 5"],
                "answer": ["3"],
            }
        )
        # Should not raise
        validate_dataset_format(dataset, "math")

    def test_validate_math_dataset_missing_answer(self):
        """Test validation fails for math dataset missing answer field."""
        dataset = Dataset.from_dict(
            {
                "problem": ["Solve x + 2 = 5"],
                "solution": ["3"],  # Wrong field name
            }
        )
        with pytest.raises(ValueError, match="must have 'answer' field"):
            validate_dataset_format(dataset, "math")

    def test_validate_code_dataset_valid_test_cases(self):
        """Test validation passes for code dataset with test_cases field."""
        dataset = Dataset.from_dict(
            {
                "problem": ["Write a function to add two numbers"],
                "test_cases": [[{"input": [1, 2], "output": 3}]],
            }
        )
        # Should not raise
        validate_dataset_format(dataset, "code")

    def test_validate_code_dataset_valid_tests(self):
        """Test validation passes for code dataset with tests field."""
        dataset = Dataset.from_dict(
            {
                "problem": ["Write a function to add two numbers"],
                "tests": [[{"input": [1, 2], "output": 3}]],
            }
        )
        # Should not raise
        validate_dataset_format(dataset, "code")

    def test_validate_code_dataset_missing_test_field(self):
        """Test validation fails for code dataset missing both test fields."""
        dataset = Dataset.from_dict(
            {
                "problem": ["Write a function to add two numbers"],
                "answer": ["def add(a, b): return a + b"],
            }
        )
        with pytest.raises(ValueError, match="must have 'test_cases' or 'tests'"):
            validate_dataset_format(dataset, "code")

    def test_validate_empty_dataset(self):
        """Test validation fails for empty dataset."""
        dataset = Dataset.from_dict({"problem": [], "answer": []})
        with pytest.raises(ValueError, match="Dataset is empty"):
            validate_dataset_format(dataset, "math")

    def test_validate_unknown_reward_type(self):
        """Test validation fails for unknown reward type."""
        dataset = Dataset.from_dict(
            {
                "problem": ["Test"],
                "answer": ["Test"],
            }
        )
        with pytest.raises(ValueError, match="Unknown reward_type"):
            validate_dataset_format(dataset, "unknown")
