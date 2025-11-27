import pytest

from vibethinker.monitor import MGPORewardCalculator


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
