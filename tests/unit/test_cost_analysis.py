import json

import pytest

from vibethinker.cost_analysis import (
    CostAnalyzer,
    TrainingStageProfile,
    compare_gpu_options,
)


def test_estimate_training_time():
    analyzer = CostAnalyzer(gpu_type="H100")
    time_est = analyzer.estimate_training_time(
        num_steps=1000, batch_size=4, seq_len=1024
    )
    assert time_est["estimated_hours"] > 0
    assert time_est["total_tokens"] == 1000 * 4 * 1024


def test_estimate_cost():
    analyzer = CostAnalyzer(gpu_type="H100")
    cost_est = analyzer.estimate_cost(num_steps=1000, batch_size=4, seq_len=1024)
    assert cost_est["total_cost"] > 0
    assert "compute_cost" in cost_est
    assert "energy_cost" in cost_est


def test_generate_full_pipeline_estimate():
    analyzer = CostAnalyzer(gpu_type="H100")
    pipeline_est = analyzer.generate_full_pipeline_estimate()
    assert len(pipeline_est["stages"]) == 7
    assert pipeline_est["total_estimated_cost"] > 0
    assert pipeline_est["total_estimated_hours"] > 0


def test_unknown_gpu_type():
    analyzer = CostAnalyzer(gpu_type="UNKNOWN_GPU")
    # Should default to H100 specs
    assert analyzer.gpu_spec["memory_gb"] == 80
    cost_est = analyzer.estimate_cost(num_steps=1000, batch_size=4, seq_len=1024)
    assert cost_est["total_cost"] > 0


def test_estimate_tokens_per_second():
    analyzer = CostAnalyzer(gpu_type="H100")
    tokens_per_sec = analyzer.estimate_tokens_per_second(
        batch_size=4, seq_len=1024, model_params=0.6e9
    )
    assert tokens_per_sec > 0
    assert isinstance(tokens_per_sec, float)


def test_cost_with_multiple_gpus():
    analyzer = CostAnalyzer(gpu_type="H100")
    cost_est = analyzer.estimate_cost(
        num_steps=1000, batch_size=4, gpu_count=4, seq_len=1024
    )
    # Cost should scale with GPU count
    single_gpu_est = analyzer.estimate_cost(
        num_steps=1000, batch_size=4, gpu_count=1, seq_len=1024
    )
    assert cost_est["compute_cost"] == single_gpu_est["compute_cost"] * 4


def test_training_stage_profile():
    profile = TrainingStageProfile(
        stage_name="test_stage",
        gpu_type="H100",
        gpu_count=1,
        batch_size=4,
        num_steps=1000,
        estimated_gpu_hours=10.0,
        estimated_compute_cost=40.0,
        estimated_energy_cost=5.0,
        estimated_total_cost=45.0,
        peak_memory_gb=48.0,
        avg_throughput_samples_sec=2.5,
    )
    assert profile.stage_name == "test_stage"
    assert profile.estimated_total_cost == 45.0


def test_different_gpu_types():
    for gpu_type in ["H100", "A100", "H800", "L4"]:
        analyzer = CostAnalyzer(gpu_type=gpu_type)
        cost_est = analyzer.estimate_cost(num_steps=100, batch_size=4, seq_len=1024)
        assert cost_est["total_cost"] > 0
        assert cost_est["compute_hours"] > 0


def test_different_regions():
    for region in ["us-west", "us-east", "eu-west", "asia"]:
        analyzer = CostAnalyzer(gpu_type="H100", region=region)
        cost_est = analyzer.estimate_cost(num_steps=100, batch_size=4, seq_len=1024)
        assert cost_est["energy_cost"] > 0


def test_cost_per_step():
    analyzer = CostAnalyzer(gpu_type="H100")
    cost_est = analyzer.estimate_cost(num_steps=1000, batch_size=4, seq_len=1024)
    assert "cost_per_step" in cost_est
    assert cost_est["cost_per_step"] == cost_est["total_cost"] / 1000


def test_print_cost_report(capsys):
    analyzer = CostAnalyzer(gpu_type="H100", region="us-west")
    analyzer.print_cost_report()
    captured = capsys.readouterr()
    assert "VIBETHINKER TRAINING COST ANALYSIS" in captured.out
    assert "GPU Type: H100" in captured.out


def test_compare_gpu_options(capsys):
    compare_gpu_options()
    captured = capsys.readouterr()
    assert "GPU COST COMPARISON" in captured.out
    assert "L4" in captured.out
    assert "A100" in captured.out
    assert "H100" in captured.out
