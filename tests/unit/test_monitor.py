import time
from unittest.mock import Mock, call, patch

import numpy as np
import pytest

from vibethinker.monitor import (
    CodeRewardCalculator,
    CostCalculator,
    CPUMetrics,
    CPUMonitor,
    GPUMetrics,
    GPUMonitor,
    MGPORewardCalculator,
    TrainingMetrics,
    TrainingMonitor,
)


def test_evaluate_solution_correct():
    calculator = MGPORewardCalculator()
    assert calculator.evaluate_solution("The answer is 42", "42") == 1.0
    assert calculator.evaluate_solution("x = 2", "x = 2") == 1.0
    assert (
        calculator.evaluate_solution("2*x + 3 = 7", "x = 2") == 0.0
    )  # Should not evaluate equations


def test_evaluate_solution_incorrect():
    calculator = MGPORewardCalculator()
    assert calculator.evaluate_solution("The answer is 43", "42") == 0.0
    assert calculator.evaluate_solution("x = 3", "x = 2") == 0.0


def test_symbolic_evaluation():
    calculator = MGPORewardCalculator()
    assert calculator.evaluate_solution("2 + 2", "4") == 1.0
    assert calculator.evaluate_solution("x*x", "x**2") == 1.0
    assert calculator.evaluate_solution("a*(b+c)", "a*b + a*c") == 1.0
    assert calculator.evaluate_solution("sin(pi/2)", "1") == 1.0


def test_cost_calculator():
    # Note: This is a basic test. In a real scenario, you'd want to
    # mock the time and subprocess calls to have deterministic tests.
    from vibethinker.monitor import CostCalculator

    calculator = CostCalculator(gpu_type="H100")
    cost = calculator.compute_cost(elapsed_seconds=3600)  # 1 hour
    assert cost["compute_cost_usd"] == 4.0
    assert cost["energy_cost_usd"] > 0


def test_mgpo_reward_calculator_init():
    calc = MGPORewardCalculator(lambda_param=5.0)
    assert calc.lambda_param == 5.0


def test_extract_answer():
    """Test answer extraction from various formats."""
    calc = MGPORewardCalculator()

    # Test various answer formats
    assert calc._extract_answer("The answer is: 42") == "42"
    assert calc._extract_answer("Therefore, x = 2") == "x = 2"
    assert calc._extract_answer("Final answer: 100") == "100"

    # Test when no pattern matches - should return last line
    text = "Step 1\nStep 2\nx = 5"
    # When there's "=" in the text, it should match that pattern
    # which extracts everything after "="
    assert calc._extract_answer(text) == "5"


def test_compute_kl_entropy_weight():
    """Test entropy weight computation."""
    calc = MGPORewardCalculator(lambda_param=4.0)

    # High entropy (50% correct)
    high_entropy = np.array([1.0, 0.0, 1.0, 0.0])
    weight_high = calc.compute_kl_entropy_weight(high_entropy)
    assert 0.9 < weight_high <= 1.0

    # Low entropy (all correct)
    low_entropy = np.array([1.0, 1.0, 1.0, 1.0])
    weight_low = calc.compute_kl_entropy_weight(low_entropy)
    assert 0.0 <= weight_low < 0.1


def test_compute_rewards():
    """Test reward computation for a batch."""
    calc = MGPORewardCalculator()

    prompts = ["Problem 1", "Problem 2"]
    completions = [
        ["answer1", "answer2", "answer3", "answer4"],
        ["solution1", "solution2", "solution3", "solution4"],
    ]
    references = ["answer1", "solution2"]

    rewards, entropy_info = calc.compute_rewards(prompts, completions, references)

    assert len(rewards) == 2
    assert len(entropy_info) == 2
    assert len(rewards[0]) == 4
    assert "entropy_weight" in entropy_info[0]
    assert "accuracy" in entropy_info[0]


def test_gpu_monitor_init():
    monitor = GPUMonitor(gpu_id=0)
    assert monitor.gpu_id == 0
    assert isinstance(monitor.available, bool)


@patch("vibethinker.monitor.subprocess.run")
def test_gpu_monitor_available(mock_run):
    """Test GPU monitor when nvidia-smi is available."""
    mock_run.return_value = Mock(returncode=0, stdout="Tesla V100")

    monitor = GPUMonitor()
    assert monitor.available is True


@patch("vibethinker.monitor.subprocess.run")
def test_gpu_monitor_unavailable(mock_run):
    """Test GPU monitor when nvidia-smi is not available."""
    mock_run.side_effect = FileNotFoundError()

    monitor = GPUMonitor()
    assert monitor.available is False


@patch("vibethinker.monitor.subprocess.run")
def test_gpu_monitor_get_metrics(mock_run):
    """Test getting GPU metrics."""
    mock_run.return_value = Mock(returncode=0, stdout="50, 8000, 16000, 250.5, 75.0\n")

    monitor = GPUMonitor()
    monitor.available = True
    metrics = monitor.get_metrics()

    assert isinstance(metrics, GPUMetrics)
    assert metrics.gpu_utilization_pct == 50.0
    assert metrics.gpu_memory_used_mb == 8000.0
    assert metrics.gpu_power_w == 250.5


def test_gpu_monitor_get_metrics_unavailable():
    """Test getting GPU metrics when nvidia-smi is unavailable."""
    monitor = GPUMonitor()
    monitor.available = False

    metrics = monitor.get_metrics()

    assert metrics.gpu_memory_used_mb == 0
    assert metrics.gpu_utilization_pct == 0


@patch("vibethinker.monitor.psutil.cpu_percent")
@patch("vibethinker.monitor.psutil.virtual_memory")
def test_cpu_monitor_get_metrics(mock_vm, mock_cpu):
    """Test getting CPU metrics."""
    mock_cpu.return_value = 45.5
    mock_vm.return_value = Mock(used=1024 * 1024 * 1024 * 4, percent=50.0)  # 4GB

    monitor = CPUMonitor()
    metrics = monitor.get_metrics()

    assert isinstance(metrics, CPUMetrics)
    assert metrics.cpu_utilization_pct == 45.5
    assert metrics.cpu_memory_pct == 50.0


def test_cost_calculator_init():
    """Test cost calculator initialization."""
    calc = CostCalculator(gpu_type="H100", region="us-west")

    assert calc.gpu_type == "H100"
    assert calc.hourly_rate == 4.0
    assert calc.power_draw == 350
    assert calc.energy_cost == 0.12


def test_cost_calculator_different_gpus():
    """Test cost calculator with different GPU types."""
    for gpu_type in ["A100", "H100", "A6000", "V100", "L4"]:
        calc = CostCalculator(gpu_type=gpu_type)
        cost = calc.compute_cost(elapsed_seconds=3600)

        assert cost["compute_cost_usd"] > 0
        assert cost["energy_cost_usd"] > 0
        assert cost["total_usd"] > 0


def test_cost_calculator_unknown_gpu():
    """Test with unknown GPU type (should use defaults)."""
    calc = CostCalculator(gpu_type="UNKNOWN")
    assert calc.hourly_rate == 2.50  # Default


def test_training_monitor_init(tmp_path):
    monitor = TrainingMonitor(output_dir=str(tmp_path / "monitoring"))

    assert monitor.output_dir.exists()
    assert monitor.metrics_history == []
    assert monitor.start_time > 0


@patch("vibethinker.monitor.GPUMonitor")
@patch("vibethinker.monitor.CPUMonitor")
def test_training_monitor_record_step(mock_cpu_mon, mock_gpu_mon, tmp_path):
    """Test recording a training step."""
    # Setup mocks
    mock_gpu_metrics = GPUMetrics(
        timestamp=time.time(),
        gpu_memory_used_mb=8000,
        gpu_memory_pct=50.0,
        gpu_power_w=250.0,
        gpu_temp_c=70.0,
        gpu_utilization_pct=80.0,
    )

    mock_cpu_metrics = CPUMetrics(
        timestamp=time.time(),
        cpu_memory_used_mb=4000,
        cpu_memory_pct=40.0,
        cpu_utilization_pct=60.0,
    )

    mock_gpu_mon_instance = Mock()
    mock_gpu_mon_instance.get_metrics.return_value = mock_gpu_metrics
    mock_gpu_mon.return_value = mock_gpu_mon_instance

    mock_cpu_mon_instance = Mock()
    mock_cpu_mon_instance.get_metrics.return_value = mock_cpu_metrics
    mock_cpu_mon.return_value = mock_cpu_mon_instance

    monitor = TrainingMonitor(output_dir=str(tmp_path / "monitoring"))

    monitor.record_step(
        step=10, loss=2.5, learning_rate=1e-5, gradient_norm=0.5, samples_processed=100
    )

    assert len(monitor.metrics_history) == 1
    assert monitor.metrics_history[0].step == 10
    assert monitor.metrics_history[0].loss == 2.5


def test_training_monitor_generate_report(tmp_path):
    """Test generating training report."""
    monitor = TrainingMonitor(output_dir=str(tmp_path / "monitoring"))

    # Add some fake metrics
    for i in range(10):
        mock_gpu = GPUMetrics(time.time(), 8000, 50.0, 250.0, 70.0, 80.0)
        mock_cpu = CPUMetrics(time.time(), 4000, 40.0, 60.0)

        metrics = TrainingMetrics(
            step=i,
            loss=5.0 - i * 0.1,
            learning_rate=1e-5,
            gradient_norm=0.5,
            throughput_samples_per_sec=10.0,
            elapsed_time_sec=i * 10,
            gpu_metrics=mock_gpu,
            cpu_metrics=mock_cpu,
            estimated_cost_usd=i * 0.01,
        )
        monitor.metrics_history.append(metrics)

    monitor.generate_report("test_checkpoint")

    # Check that report file was created
    report_file = monitor.output_dir / "report_test_checkpoint.txt"
    assert report_file.exists()


@patch("vibethinker.monitor.plt")
def test_training_monitor_plot_curves(mock_plt, tmp_path):
    """Test plotting training curves."""
    monitor = TrainingMonitor(output_dir=str(tmp_path / "monitoring"))

    # Add some metrics
    for i in range(10):
        mock_gpu = GPUMetrics(time.time(), 8000, 50.0, 250.0, 70.0, 80.0)
        mock_cpu = CPUMetrics(time.time(), 4000, 40.0, 60.0)

        metrics = TrainingMetrics(
            step=i,
            loss=5.0 - i * 0.1,
            learning_rate=1e-5,
            gradient_norm=0.5,
            throughput_samples_per_sec=10.0,
            elapsed_time_sec=i * 10,
            gpu_metrics=mock_gpu,
            cpu_metrics=mock_cpu,
            estimated_cost_usd=i * 0.01,
        )
        monitor.metrics_history.append(metrics)

    # Mock the subplots to return fig and properly indexable axes
    mock_fig = Mock()
    # Create 2x2 grid of Mock axes that can be indexed
    import numpy as np

    mock_axes_array = np.array([[Mock(), Mock()], [Mock(), Mock()]])
    mock_plt.subplots.return_value = (mock_fig, mock_axes_array)

    monitor.plot_training_curves("test")

    # Verify subplots was called
    assert mock_plt.subplots.called


def test_training_monitor_empty_report(tmp_path):
    """Test generating report with no metrics."""
    monitor = TrainingMonitor(output_dir=str(tmp_path / "monitoring"))

    # Should handle empty metrics gracefully
    monitor.generate_report()


def test_training_monitor_empty_plot(tmp_path):
    """Test plotting with no metrics."""
    monitor = TrainingMonitor(output_dir=str(tmp_path / "monitoring"))

    # Should handle empty metrics gracefully
    monitor.plot_training_curves()


def test_evaluate_solution_with_sympy_error():
    """Test evaluation when sympy raises an error."""
    calc = MGPORewardCalculator()

    # Some complex expression that might cause issues
    result = calc.evaluate_solution("invalid$$expression", "42")

    # Should fall back to string comparison
    assert result == 0.0


def test_compute_rewards_with_different_correctness():
    """Test rewards with varying correctness levels."""
    calc = MGPORewardCalculator()

    prompts = ["P1"]
    completions = [["a1", "a1", "a1", "wrong"]]  # 75% correct
    references = ["a1"]

    rewards, entropy_info = calc.compute_rewards(prompts, completions, references)

    assert sum(rewards[0]) == 3.0  # 3 correct out of 4
    assert entropy_info[0]["num_correct"] == 3
    assert entropy_info[0]["num_total"] == 4
    assert entropy_info[0]["accuracy"] == 0.75


@patch("vibethinker.monitor.subprocess.run")
def test_gpu_monitor_parse_error(mock_run):
    """Test GPU monitor handles parsing errors gracefully."""
    mock_run.return_value = Mock(returncode=0, stdout="invalid,data,format\n")

    monitor = GPUMonitor()
    monitor.available = True

    # Should handle parse error and return zero metrics
    with patch("builtins.print"):  # Suppress error printing
        metrics = monitor.get_metrics()

    assert metrics.gpu_utilization_pct == 0.0


def test_cost_calculator_compute_cost_zero_time():
    """Test cost calculation with zero elapsed time."""
    calc = CostCalculator()
    cost = calc.compute_cost(elapsed_seconds=0)

    assert cost["compute_hours"] == 0.0
    assert cost["compute_cost_usd"] == 0.0


def test_mgpo_reward_calculator_empty_completions():
    """Test reward calculation with empty completions."""
    calc = MGPORewardCalculator()

    prompts = ["P1"]
    completions = [[]]  # Empty completions
    references = ["ref"]

    rewards, entropy_info = calc.compute_rewards(prompts, completions, references)

    assert len(rewards[0]) == 0


# CodeRewardCalculator Tests


def test_code_reward_calculator_init():
    """Test CodeRewardCalculator initialization."""
    calc = CodeRewardCalculator(timeout=3.0)
    assert calc.timeout == 3.0


def test_evaluate_code_correct():
    """Test correct code evaluation."""
    calc = CodeRewardCalculator()

    code = """
def add(a, b):
    return a + b
"""

    test_cases = [
        {"input": [1, 2], "output": 3},
        {"input": [5, 7], "output": 12},
    ]

    result = calc.evaluate_code(code, test_cases)
    assert result == 1.0


def test_evaluate_code_syntax_error():
    """Test code evaluation with syntax error."""
    calc = CodeRewardCalculator()

    code = """
def add(a, b
    return a + b  # Missing closing parenthesis
"""

    test_cases = [{"input": [1, 2], "output": 3}]

    result = calc.evaluate_code(code, test_cases)
    assert result == 0.0


def test_evaluate_code_runtime_error():
    """Test code evaluation with runtime error."""
    calc = CodeRewardCalculator()

    code = """
def divide(a, b):
    return a / b
"""

    test_cases = [{"input": [1, 0], "output": 0}]  # Division by zero

    result = calc.evaluate_code(code, test_cases)
    assert result == 0.0


def test_evaluate_code_timeout():
    """Test code evaluation with timeout."""
    calc = CodeRewardCalculator(timeout=0.1)

    code = """
def infinite_loop():
    while True:
        pass
"""

    test_cases = [{"input": [], "output": None}]

    result = calc.evaluate_code(code, test_cases)
    assert result == 0.0


def test_evaluate_code_test_case_failure():
    """Test code evaluation when test cases fail."""
    calc = CodeRewardCalculator()

    code = """
def add(a, b):
    return a - b  # Wrong operation
"""

    test_cases = [{"input": [1, 2], "output": 3}]

    result = calc.evaluate_code(code, test_cases)
    assert result == 0.0


def test_compute_rewards_code():
    """Test batch reward computation for code."""
    calc = CodeRewardCalculator()

    prompts = ["Write add function", "Write multiply function"]
    completions = [
        ["def add(a, b):\n    return a + b", "def add(a, b):\n    return a - b"],
        ["def mul(a, b):\n    return a * b", "def mul(a, b):\n    return a + b"],
    ]
    test_suites = [
        [{"input": [1, 2], "output": 3}],
        [{"input": [2, 3], "output": 6}],
    ]

    rewards, infos = calc.compute_rewards(prompts, completions, test_suites)

    assert len(rewards) == 2
    assert len(infos) == 2
    assert len(rewards[0]) == 2
    assert len(rewards[1]) == 2
    assert all(info["entropy_weight"] == 1.0 for info in infos)


def test_extract_code():
    """Test code extraction from markdown."""
    calc = CodeRewardCalculator()

    # Test with markdown code block
    text_with_markdown = """
Here is the solution:

```python
def add(a, b):
    return a + b
```
"""

    extracted = calc._extract_code(text_with_markdown)
    assert "def add(a, b):" in extracted
    assert "```" not in extracted

    # Test without markdown
    plain_code = "def add(a, b):\n    return a + b"
    extracted = calc._extract_code(plain_code)
    assert extracted == plain_code


def test_evaluate_code_no_functions():
    """Test evaluation when code defines no functions."""
    calc = CodeRewardCalculator()

    code = """
x = 5
y = 10
"""

    test_cases = [{"input": [], "output": 15}]

    result = calc.evaluate_code(code, test_cases)
    assert result == 0.0  # No callable function found


def test_evaluate_code_partial_pass():
    """Test that code must pass all test cases."""
    calc = CodeRewardCalculator()

    code = """
def add(a, b):
    if a == 1 and b == 2:
        return 3
    return 0
"""

    test_cases = [
        {"input": [1, 2], "output": 3},  # Passes
        {"input": [2, 3], "output": 5},  # Fails
    ]

    result = calc.evaluate_code(code, test_cases)
    assert result == 0.0  # Should fail because not all tests pass
