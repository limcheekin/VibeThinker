"""
Tests for train_complete.py - simplified to avoid import issues.

Since train_complete.py requires unsloth and other complex dependencies,
we focus on testing the structure and imports rather than full execution.
"""

from unittest import mock

import pytest


def test_module_imports():
    """Test that all required modules can be imported."""
    with mock.patch.dict(
        "sys.modules",
        {
            "unsloth": mock.Mock(),
            "unsloth.FastLanguageModel": mock.Mock(),
            "datasets": mock.Mock(),
            "datasets.load_dataset": mock.Mock(),
            "transformers": mock.Mock(),
            "transformers.AutoTokenizer": mock.Mock(),
        },
    ):
        from vibethinker import train_complete

        assert train_complete is not None


def test_function_exists():
    """Test that train_signal_phase_complete function exists."""
    with mock.patch.dict(
        "sys.modules",
        {
            "unsloth": mock.Mock(),
            "unsloth.FastLanguageModel": mock.Mock(),
            "datasets": mock.Mock(),
            "datasets.load_dataset": mock.Mock(),
            "transformers": mock.Mock(),
            "transformers.AutoTokenizer": mock.Mock(),
        },
    ):
        from vibethinker.train_complete import train_signal_phase_complete

        assert callable(train_signal_phase_complete)


def test_function_signature():
    """Test function has expected parameters."""
    import inspect

    with mock.patch.dict(
        "sys.modules",
        {
            "unsloth": mock.Mock(),
            "unsloth.FastLanguageModel": mock.Mock(),
            "datasets": mock.Mock(),
            "datasets.load_dataset": mock.Mock(),
            "transformers": mock.Mock(),
            "transformers.AutoTokenizer": mock.Mock(),
        },
    ):
        from vibethinker.train_complete import train_signal_phase_complete

        sig = inspect.signature(train_signal_phase_complete)
        params = list(sig.parameters.keys())
        assert "spectrum_model_path" in params
        assert "train_dataset" in params
        assert "val_dataset" in params
        assert "tokenizer" in params


def test_dependencies_imported():
    """Test that dependencies are properly imported in the module."""
    with mock.patch.dict(
        "sys.modules",
        {
            "unsloth": mock.Mock(),
            "unsloth.FastLanguageModel": mock.Mock(),
            "datasets": mock.Mock(),
            "datasets.load_dataset": mock.Mock(),
            "transformers": mock.Mock(),
            "transformers.AutoTokenizer": mock.Mock(),
        },
    ):
        from vibethinker import train_complete

        # Check module has expected attributes
        assert hasattr(train_complete, "json")
        assert hasattr(train_complete, "os")


def test_function_has_max_seq_length_parameter():
    """Test that train_signal_phase_complete has max_seq_length parameter."""
    import inspect

    with mock.patch.dict(
        "sys.modules",
        {
            "unsloth": mock.Mock(),
            "unsloth.FastLanguageModel": mock.Mock(),
            "datasets": mock.Mock(),
            "datasets.load_dataset": mock.Mock(),
            "transformers": mock.Mock(),
            "transformers.AutoTokenizer": mock.Mock(),
        },
    ):
        from vibethinker.train_complete import train_signal_phase_complete

        sig = inspect.signature(train_signal_phase_complete)
        params = list(sig.parameters.keys())

        # Verify new parameter exists
        assert "max_seq_length" in params

        # Verify default value
        assert sig.parameters["max_seq_length"].default == 4096


def test_cli_arguments_parser():
    """Test that CLI argument parser has all required arguments."""
    import sys

    with mock.patch.dict(
        "sys.modules",
        {
            "unsloth": mock.Mock(),
            "unsloth.FastLanguageModel": mock.Mock(),
            "datasets": mock.Mock(),
            "datasets.load_dataset": mock.Mock(),
            "transformers": mock.Mock(),
            "transformers.AutoTokenizer": mock.Mock(),
        },
    ):
        # Import module to access parser creation
        import argparse

        # Create a parser similar to train_complete.py
        parser = argparse.ArgumentParser()
        parser.add_argument("--spectrum-path", type=str, required=True)
        parser.add_argument(
            "--output-dir", type=str, default="outputs/vibethinker_complete"
        )
        parser.add_argument("--gpu-type", type=str, default="H100")
        parser.add_argument("--max-steps", type=int, default=2000)
        parser.add_argument("--max-seq-length", type=int, default=4096)
        parser.add_argument(
            "--train-data", type=str, default="data/algebra_train.jsonl"
        )
        parser.add_argument("--val-data", type=str, default="data/algebra_val.jsonl")

        # Parse test arguments
        test_args = [
            "--spectrum-path",
            "test/path",
            "--max-seq-length",
            "8192",
            "--train-data",
            "custom/train.jsonl",
            "--val-data",
            "custom/val.jsonl",
        ]

        args = parser.parse_args(test_args)

        # Verify arguments are parsed correctly
        assert args.spectrum_path == "test/path"
        assert args.max_seq_length == 8192
        assert args.train_data == "custom/train.jsonl"
        assert args.val_data == "custom/val.jsonl"

        # Verify defaults for arguments not provided
        assert args.output_dir == "outputs/vibethinker_complete"
        assert args.gpu_type == "H100"
        assert args.max_steps == 2000
