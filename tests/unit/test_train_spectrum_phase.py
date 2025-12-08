"""Tests for train_spectrum_phase.py orchestrator script."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest


@pytest.fixture(autouse=True)
def mock_gpu_dependencies():
    """Mock GPU-dependent imports to allow tests to run without GPU."""
    # Mock unsloth before any imports
    sys.modules["unsloth"] = MagicMock()
    sys.modules["unsloth_zoo"] = MagicMock()

    # Mock vibethinker modules that depend on unsloth
    with patch.dict(
        "sys.modules",
        {
            "unsloth": MagicMock(),
            "unsloth.FastLanguageModel": MagicMock(),
        },
    ):
        yield


def test_get_best_checkpoint(tmp_path, mock_gpu_dependencies):
    """Test checkpoint selection logic."""
    # Create mock checkpoints
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    (checkpoint_dir / "checkpoint-100").mkdir()
    (checkpoint_dir / "checkpoint-200").mkdir()
    (checkpoint_dir / "checkpoint-300").mkdir()

    # Create mock validation data
    data_file = tmp_path / "val.jsonl"
    with open(data_file, "w") as f:
        for i in range(10):
            json.dump({"problem": f"P{i}", "answer": f"A{i}"}, f)
            f.write("\n")

    # Add scripts to path
    scripts_path = str(Path(__file__).parent.parent.parent / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    import train_spectrum_phase

    with patch.object(train_spectrum_phase, "DiversityProber") as MockProber:
        # Mock prober to return different scores
        prober_instance = Mock()
        prober_instance.probe_domain.side_effect = [
            {"diversity_score": 0.5, "pass@1": 0.3},
            {"diversity_score": 0.8, "pass@1": 0.4},  # Best
            {"diversity_score": 0.6, "pass@1": 0.5},
        ]
        MockProber.return_value = prober_instance

        from train_spectrum_phase import get_best_checkpoint

        best = get_best_checkpoint(
            domain="test",
            checkpoint_dir=str(checkpoint_dir),
            data_path=str(data_file),
            k=8,
            num_gen=16,
        )

        assert "checkpoint-200" in best
        assert prober_instance.probe_domain.call_count == 3


def test_train_domain_specialist(mock_gpu_dependencies):
    """Test domain specialist training invocation."""
    scripts_path = str(Path(__file__).parent.parent.parent / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    from train_spectrum_phase import train_domain_specialist

    with patch("os.system") as mock_system:
        mock_system.return_value = 0  # Success

        checkpoint_dir = train_domain_specialist(
            domain="algebra",
            data_path="data/algebra_train.jsonl",
            base_model="unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
            output_dir="checkpoints",
            max_steps=100,
        )

        assert checkpoint_dir == "checkpoints/algebra_specialist"
        assert mock_system.called

        # Check command structure
        call_args = mock_system.call_args[0][0]
        assert "train_sft_specialist.py" in call_args
        assert "--domain algebra" in call_args
        assert "--max-steps 100" in call_args


def test_train_domain_specialist_failure(mock_gpu_dependencies):
    """Test domain specialist training handles failures."""
    scripts_path = str(Path(__file__).parent.parent.parent / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    from train_spectrum_phase import train_domain_specialist

    with patch("os.system") as mock_system:
        mock_system.return_value = 1  # Failure

        with pytest.raises(RuntimeError, match="Training failed"):
            train_domain_specialist(
                domain="algebra",
                data_path="data/algebra_train.jsonl",
                base_model="unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
                output_dir="checkpoints",
                max_steps=100,
            )


def test_main_full_pipeline(tmp_path, mock_gpu_dependencies):
    """Test full pipeline orchestration."""
    scripts_path = str(Path(__file__).parent.parent.parent / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    # Create mock data files
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    for domain in ["algebra", "geometry"]:
        data_file = data_dir / f"{domain}_train.jsonl"
        with open(data_file, "w") as f:
            for i in range(5):
                json.dump({"problem": f"P{i}", "answer": f"A{i}"}, f)
                f.write("\n")

    from train_spectrum_phase import main

    # Mock argparse
    mock_args = Mock()
    mock_args.data_dir = str(data_dir)
    mock_args.output_dir = str(tmp_path / "checkpoints")
    mock_args.base_model = "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit"
    mock_args.max_steps = 10
    mock_args.k = 8
    mock_args.num_generations = 16
    mock_args.skip_training = True  # Skip actual training

    import train_spectrum_phase

    with patch("argparse.ArgumentParser") as MockParser:
        parser_instance = Mock()
        parser_instance.parse_args.return_value = mock_args
        MockParser.return_value = parser_instance

        with patch.object(train_spectrum_phase, "DiversityProber"):
            with patch("train_spectrum_phase.get_best_checkpoint") as mock_select:
                mock_select.return_value = (
                    "checkpoints/algebra_specialist/checkpoint-100"
                )

                with patch.object(
                    train_spectrum_phase, "load_expert_models"
                ) as mock_load:
                    mock_experts = [Mock(), Mock()]
                    mock_load.return_value = mock_experts

                    with patch.object(
                        train_spectrum_phase, "fusion_weighted_average"
                    ) as mock_fusion:
                        mock_fused = Mock()
                        mock_fused.save_pretrained = Mock()
                        mock_fusion.return_value = mock_fused

                        # Create checkpoint directories
                        for domain in ["algebra", "geometry"]:
                            ckpt_dir = tmp_path / "checkpoints" / f"{domain}_specialist"
                            ckpt_dir.mkdir(parents=True)
                            (ckpt_dir / "checkpoint-100").mkdir()

                        result = main()

                        assert result == 0  # Success
                        assert mock_fusion.called
                        assert mock_fused.save_pretrained.called


def test_main_no_data_files(tmp_path, mock_gpu_dependencies):
    """Test main handles missing data files gracefully."""
    scripts_path = str(Path(__file__).parent.parent.parent / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    from train_spectrum_phase import main

    # Create empty data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    mock_args = Mock()
    mock_args.data_dir = str(data_dir)
    mock_args.output_dir = str(tmp_path / "checkpoints")
    mock_args.base_model = "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit"
    mock_args.max_steps = 10
    mock_args.k = 8
    mock_args.num_generations = 16
    mock_args.skip_training = False

    with patch("argparse.ArgumentParser") as MockParser:
        parser_instance = Mock()
        parser_instance.parse_args.return_value = mock_args
        MockParser.return_value = parser_instance

        result = main()

        assert result == 1  # Failure - no experts


def test_domains_constant(mock_gpu_dependencies):
    """Test that DOMAINS constant is correctly defined."""
    scripts_path = str(Path(__file__).parent.parent.parent / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    from train_spectrum_phase import BASE_MODEL, DOMAINS

    assert DOMAINS == ["algebra", "geometry", "calculus", "statistics", "code"]
    assert "Qwen" in BASE_MODEL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
