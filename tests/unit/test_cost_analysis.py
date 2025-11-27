import pytest

from vibethinker.cost_analysis import CostAnalyzer


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
