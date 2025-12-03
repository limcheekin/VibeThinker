"""
Unit tests for gradient accumulation in grpo_custom.py
"""

import pytest
import torch

from vibethinker.grpo_custom import MGPOTrainerWithEntropyWeighting


class MockModel:
    """Mock model for testing."""

    def __init__(self):
        self.config = type("Config", (), {"_name_or_path": "test/model"})()
        self.param = torch.nn.Parameter(torch.randn(10, 10))

    def parameters(self):
        return [self.param]

    def eval(self):
        pass

    def cpu(self):
        return self


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"


class MockConfig:
    """Mock config for testing."""

    def __init__(self, gradient_accumulation_steps=1):
        self.learning_rate = 1e-4
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_seq_length = 2048


class MockRewardCalculator:
    """Mock reward calculator."""

    def compute_rewards(self, prompts, completions, reference_answers):
        # Return dummy rewards
        rewards = [[0.5 for _ in comp] for comp in completions]
        infos = [{"entropy_weight": 1.0} for _ in prompts]
        return rewards, infos


class TestGradientAccumulation:
    """Test gradient accumulation mechanism."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_accumulation_counter_initialized(self):
        """Test that accumulation counter is initialized."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig(gradient_accumulation_steps=4)
        reward_calc = MockRewardCalculator()

        # Skip actual reference model creation for unit test
        ref_model = MockModel()

        trainer = MGPOTrainerWithEntropyWeighting(
            model=model,
            tokenizer=tokenizer,
            config=config,
            reward_calculator=reward_calc,
            device="cpu",
            ref_model=ref_model,
        )

        assert hasattr(trainer, "_accumulation_counter")
        assert trainer._accumulation_counter == 0
        assert trainer.gradient_accumulation_steps == 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_should_update_weights(self):
        """Test should_update_weights method logic."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig(gradient_accumulation_steps=4)
        reward_calc = MockRewardCalculator()
        ref_model = MockModel()

        trainer = MGPOTrainerWithEntropyWeighting(
            model=model,
            tokenizer=tokenizer,
            config=config,
            reward_calculator=reward_calc,
            device="cpu",
            ref_model=ref_model,
        )

        # Test accumulation cycle
        trainer._accumulation_counter = 0
        assert not trainer.should_update_weights()  # Step 1 of 4

        trainer._accumulation_counter = 1
        assert not trainer.should_update_weights()  # Step 2 of 4

        trainer._accumulation_counter = 2
        assert not trainer.should_update_weights()  # Step 3 of 4

        trainer._accumulation_counter = 3
        assert trainer.should_update_weights()  # Step 4 of 4 - should update

        trainer._accumulation_counter = 4
        assert not trainer.should_update_weights()  # Next cycle begins

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_accumulation_steps_default(self):
        """Test that gradient_accumulation_steps defaults to 1."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig(gradient_accumulation_steps=1)
        reward_calc = MockRewardCalculator()
        ref_model = MockModel()

        trainer = MGPOTrainerWithEntropyWeighting(
            model=model,
            tokenizer=tokenizer,
            config=config,
            reward_calculator=reward_calc,
            device="cpu",
            ref_model=ref_model,
        )

        assert trainer.gradient_accumulation_steps == 1
        # With accumulation_steps=1, every step should update
        trainer._accumulation_counter = 0
        assert trainer.should_update_weights()
