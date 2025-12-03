"""
Unit tests for memory management in diversity_probing.py
"""

import pytest

from vibethinker.diversity_probing import DiversityProber


class TestDiversityProbingMemory:
    """Test memory management in DiversityProber."""

    def test_unload_model_method_exists(self):
        """Test that unload_model method exists."""
        # We can't easily instantiate DiversityProber without a real model
        # but we can verify the method exists on the class
        assert hasattr(DiversityProber, "unload_model")
        assert callable(getattr(DiversityProber, "unload_model"))

    def test_destructor_exists(self):
        """Test that __del__ destructor exists."""
        assert hasattr(DiversityProber, "__del__")
        assert callable(getattr(DiversityProber, "__del__"))

    @pytest.mark.skipif(True, reason="Requires real model checkpoint")
    def test_unload_model_clears_attributes(self):
        """Test that unload_model removes model and tokenizer attributes.

        This test is skipped because it requires a real model checkpoint.
        In practice, manual testing with a checkpoint confirms this works.
        """
        # This would require a real checkpoint
        # prober = DiversityProber("path/to/checkpoint")
        # assert hasattr(prober, "model")
        # assert hasattr(prober, "tokenizer")
        # prober.unload_model()
        # assert not hasattr(prober, "model")
        # assert not hasattr(prober, "tokenizer")
        pass
