import ast
import multiprocessing
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil


@dataclass
class GPUMetrics:
    """GPU resource metrics."""

    timestamp: float
    gpu_memory_used_mb: float
    gpu_memory_pct: float
    gpu_power_w: float
    gpu_temp_c: float
    gpu_utilization_pct: float


@dataclass
class CPUMetrics:
    """CPU resource metrics."""

    timestamp: float
    cpu_memory_used_mb: float
    cpu_memory_pct: float
    cpu_utilization_pct: float


@dataclass
class TrainingMetrics:
    """Training performance metrics."""

    step: int
    loss: float
    learning_rate: float
    gradient_norm: Optional[float]
    throughput_samples_per_sec: float
    elapsed_time_sec: float
    gpu_metrics: GPUMetrics
    cpu_metrics: CPUMetrics
    estimated_cost_usd: float


class GPUMonitor:
    """Monitor GPU utilization and power consumption."""

    def __init__(self, gpu_id: int = 0) -> None:
        self.gpu_id = gpu_id
        self.available = False
        self._check_nvidia_smi()

    def _check_nvidia_smi(self) -> None:
        """Verify nvidia-smi is available."""
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                check=True,
            )
            self.available = True
        except FileNotFoundError:
            print("WARNING: nvidia-smi not found. GPU metrics unavailable.")
            self.available = False

    def get_metrics(self) -> GPUMetrics:
        """Get current GPU metrics."""
        if not self.available:
            return GPUMetrics(
                timestamp=time.time(),
                gpu_memory_used_mb=0,
                gpu_memory_pct=0,
                gpu_power_w=0,
                gpu_temp_c=0,
                gpu_utilization_pct=0,
            )
        try:
            query = (
                "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
            )
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_id}",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            parts = result.stdout.strip().split(",")
            gpu_util = float(parts[0])
            mem_used = float(parts[1])
            mem_total = float(parts[2])
            power = float(parts[3]) if len(parts) > 3 else 0.0
            temp = float(parts[4]) if len(parts) > 4 else 0.0
            return GPUMetrics(
                timestamp=time.time(),
                gpu_memory_used_mb=mem_used,
                gpu_memory_pct=100.0 * mem_used / mem_total if mem_total > 0 else 0,
                gpu_power_w=power,
                gpu_temp_c=temp,
                gpu_utilization_pct=gpu_util,
            )
        except Exception as e:
            print(f"Error querying GPU metrics: {e}")
            return GPUMetrics(
                timestamp=time.time(),
                gpu_memory_used_mb=0,
                gpu_memory_pct=0,
                gpu_power_w=0,
                gpu_temp_c=0,
                gpu_utilization_pct=0,
            )


class CPUMonitor:
    """Monitor CPU and system memory."""

    def get_metrics(self) -> CPUMetrics:
        """Get current CPU metrics."""
        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        return CPUMetrics(
            timestamp=time.time(),
            cpu_memory_used_mb=mem.used / (1024**2),
            cpu_memory_pct=mem.percent,
            cpu_utilization_pct=cpu_pct,
        )


class CostCalculator:
    """Calculate training costs based on GPU type and compute."""

    GPU_RATES: Dict[str, float] = {
        "A100": 2.50,
        "H100": 4.00,
        "A6000": 1.80,
        "V100": 1.50,
        "L4": 0.35,
    }
    POWER_DRAW: Dict[str, int] = {
        "A100": 250,
        "H100": 350,
        "A6000": 300,
        "V100": 250,
        "L4": 72,
    }
    ENERGY_COST_PER_KWH: Dict[str, float] = {
        "us-west": 0.12,
        "us-east": 0.14,
        "eu": 0.25,
        "asia": 0.20,
    }

    def __init__(self, gpu_type: str = "H100", region: str = "us-west") -> None:
        self.gpu_type = gpu_type
        self.hourly_rate = self.GPU_RATES.get(gpu_type, 2.50)
        self.power_draw = self.POWER_DRAW.get(gpu_type, 250)
        self.energy_cost = self.ENERGY_COST_PER_KWH.get(region, 0.15)

    def compute_cost(self, elapsed_seconds: float) -> Dict[str, float]:
        """Calculate total training cost."""
        hours = elapsed_seconds / 3600
        compute_cost = hours * self.hourly_rate
        energy_kwh = (self.power_draw / 1000) * hours * 0.85
        energy_cost = energy_kwh * self.energy_cost
        return {
            "compute_hours": hours,
            "compute_cost_usd": compute_cost,
            "energy_kwh": energy_kwh,
            "energy_cost_usd": energy_cost,
            "total_usd": compute_cost + energy_cost,
        }


class TrainingMonitor:
    """Main training monitor with logging and visualization."""

    def __init__(self, output_dir: str = "monitoring", gpu_id: int = 0) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.gpu_monitor = GPUMonitor(gpu_id)
        self.cpu_monitor = CPUMonitor()
        self.cost_calc = CostCalculator()
        self.metrics_history: List[TrainingMetrics] = []
        self.start_time = time.time()

    def record_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        gradient_norm: Optional[float] = None,
        samples_processed: int = 0,
    ) -> None:
        """Record metrics for a training step."""
        elapsed = time.time() - self.start_time
        throughput = samples_processed / elapsed if elapsed > 0 else 0
        gpu_metrics = self.gpu_monitor.get_metrics()
        cpu_metrics = self.cpu_monitor.get_metrics()
        cost = self.cost_calc.compute_cost(elapsed)["total_usd"]
        metrics = TrainingMetrics(
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            throughput_samples_per_sec=throughput,
            elapsed_time_sec=elapsed,
            gpu_metrics=gpu_metrics,
            cpu_metrics=cpu_metrics,
            estimated_cost_usd=cost,
        )
        self.metrics_history.append(metrics)
        if step % 10 == 0:
            self._log_step(metrics)

    def _log_step(self, metrics: TrainingMetrics) -> None:
        """Print step summary."""
        print(
            f"Step {metrics.step} | Loss: {metrics.loss:.4f} | "
            f"GPU Mem: {metrics.gpu_metrics.gpu_memory_pct:.1f}% | "
            f"Cost: ${metrics.estimated_cost_usd:.2f}"
        )

    def generate_report(self, checkpoint_name: str = "final") -> None:
        """Generate comprehensive training report."""
        if not self.metrics_history:
            print("No metrics recorded.")
            return
        output_file = self.output_dir / f"report_{checkpoint_name}.txt"
        with open(output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("VIBETHINKER TRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")
            metrics = self.metrics_history
            f.write("TRAINING SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total steps: {metrics[-1].step}\n")
            f.write(f"Total time: {metrics[-1].elapsed_time_sec / 3600:.2f} hours\n")
            f.write(f"Final loss: {metrics[-1].loss:.6f}\n")
            f.write(f"Best loss: {min(m.loss for m in metrics):.6f}\n")
            f.write(f"Total estimated cost: ${metrics[-1].estimated_cost_usd:.2f}\n\n")
            f.write("GPU UTILIZATION\n")
            f.write("-" * 80 + "\n")
            gpu_util = [m.gpu_metrics.gpu_utilization_pct for m in metrics]
            gpu_mem = [m.gpu_metrics.gpu_memory_pct for m in metrics]
            f.write(f"Average GPU utilization: {np.mean(gpu_util):.1f}%\n")
            f.write(f"Max GPU memory: {np.max(gpu_mem):.1f}%\n")
            f.write(f"Avg GPU memory: {np.mean(gpu_mem):.1f}%\n\n")
            f.write("THROUGHPUT\n")
            f.write("-" * 80 + "\n")
            throughputs = [m.throughput_samples_per_sec for m in metrics]
            f.write(f"Average: {np.mean(throughputs):.1f} samples/sec\n")
            f.write(f"Max: {np.max(throughputs):.1f} samples/sec\n\n")
            f.write("COST ANALYSIS\n")
            f.write("-" * 80 + "\n")
            final_metrics = metrics[-1]
            elapsed_hours = final_metrics.elapsed_time_sec / 3600
            compute_cost = elapsed_hours * self.cost_calc.hourly_rate
            energy_kwh = (self.cost_calc.power_draw / 1000) * elapsed_hours * 0.85
            energy_cost = energy_kwh * self.cost_calc.energy_cost
            f.write(f"GPU hourly rate: ${self.cost_calc.hourly_rate:.2f}\n")
            f.write(f"Compute cost: ${compute_cost:.2f}\n")
            f.write(f"Energy consumption: {energy_kwh:.1f} kWh\n")
            f.write(f"Energy cost: ${energy_cost:.2f}\n")
            f.write(f"Total cost: ${compute_cost + energy_cost:.2f}\n\n")
        print(f"Report saved to: {output_file}")

    def plot_training_curves(self, checkpoint_name: str = "final") -> None:
        """Generate training visualization plots."""
        if not self.metrics_history:
            print("No metrics to plot.")
            return
        metrics = self.metrics_history
        steps = [m.step for m in metrics]
        losses = [m.loss for m in metrics]
        gpu_util = [m.gpu_metrics.gpu_utilization_pct for m in metrics]
        gpu_mem = [m.gpu_metrics.gpu_memory_pct for m in metrics]
        costs = [m.estimated_cost_usd for m in metrics]
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].plot(steps, losses, linewidth=2)
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(steps, gpu_util, linewidth=2, color="green")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Utilization (%)")
        axes[0, 1].set_title("GPU Utilization")
        axes[0, 1].set_ylim([0, 100])
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(steps, gpu_mem, linewidth=2, color="orange")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Memory (%)")
        axes[1, 0].set_title("GPU Memory Usage")
        axes[1, 0].set_ylim([0, 100])
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(steps, costs, linewidth=2, color="red")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Cost (USD)")
        axes[1, 1].set_title("Cumulative Training Cost")
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        output_file = self.output_dir / f"training_curves_{checkpoint_name}.png"
        plt.savefig(output_file, dpi=150)
        print(f"Training curves saved to: {output_file}")
        plt.close()


class MGPORewardCalculator:
    """
    Reward calculator for MGPO with symbolic answer evaluation.
    """

    def __init__(self, lambda_param: float = 4.0) -> None:
        """
        Initialize reward calculator.
        """
        self.lambda_param = lambda_param

    def evaluate_solution(self, generated: str, reference: str) -> float:
        """
        Evaluate a single generated solution against reference answer.
        """
        try:
            import sympy

            generated_clean = self._extract_answer(generated)
            reference_clean = self._extract_answer(reference)
            try:
                gen_expr = sympy.sympify(generated_clean)
                ref_expr = sympy.sympify(reference_clean)
                if sympy.simplify(gen_expr - ref_expr) == 0:
                    return 1.0
            except (sympy.SympifyError, TypeError, ValueError):
                if generated_clean.strip() == reference_clean.strip():
                    return 1.0
            return 0.0
        except Exception:
            return 1.0 if generated.strip() == reference.strip() else 0.0

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from solution text."""
        import re

        patterns = [
            r"answer is:?\s*(.+?)(?:\n|$)",
            r"final answer:?\s*(.+?)(?:\n|$)",
            r"therefore,?\s*(.+?)(?:\n|$)",
            r"=\s*(.+?)(?:\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        lines = text.strip().split("\n")
        return lines[-1].strip() if lines else text.strip()

    def compute_kl_entropy_weight(self, group_correctness: np.ndarray) -> float:
        """
        Compute KL divergence from maximum entropy distribution.
        """
        p_correct = np.mean(group_correctness)
        p_correct = np.clip(p_correct, 1e-6, 1 - 1e-6)
        p_0 = 0.5
        d_kl = p_correct * np.log(p_correct / p_0) + (1 - p_correct) * np.log(
            (1 - p_correct) / p_0
        )
        entropy_weight = np.exp(-self.lambda_param * d_kl)
        return float(entropy_weight)

    def compute_rewards(
        self,
        prompts: List[str],
        completions: List[List[str]],
        reference_answers: List[str],
    ) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
        """
        Compute rewards and entropy weights for a batch.
        """
        batch_rewards: List[List[float]] = []
        batch_entropy_info: List[Dict[str, Any]] = []
        for i, (prompt, completion_group, reference) in enumerate(
            zip(prompts, completions, reference_answers)
        ):
            group_correctness = np.array(
                [self.evaluate_solution(comp, reference) for comp in completion_group]
            )
            entropy_weight = self.compute_kl_entropy_weight(group_correctness)
            group_rewards = group_correctness.tolist()
            batch_rewards.append(group_rewards)
            batch_entropy_info.append(
                {
                    "entropy_weight": entropy_weight,
                    "accuracy": float(np.mean(group_correctness)),
                    "num_correct": int(np.sum(group_correctness)),
                    "num_total": len(group_correctness),
                }
            )
        return batch_rewards, batch_entropy_info



# Safe builtins for code execution - removes dangerous operations
SAFE_BUILTINS = {
    'abs': abs,
    'all': all,
    'any': any,
    'bool': bool,
    'dict': dict,
    'enumerate': enumerate,
    'float': float,
    'int': int,
    'len': len,
    'list': list,
    'max': max,
    'min': min,
    'pow': pow,
    'range': range,
    'round': round,
    'set': set,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'tuple': tuple,
    'zip': zip,
    # Note: Deliberately excluding: open, input, compile, eval, exec, __import__
}

# Whitelist of safe modules that can be used in code execution
SAFE_MODULES = {'math', 'numpy'}


def _create_safe_globals() -> Dict[str, Any]:
    """
    Create restricted globals for code execution.
    
    Returns:
        Dictionary with safe builtins and allowed module imports
    """
    safe_globals: Dict[str, Any] = {'__builtins__': SAFE_BUILTINS}
    
    # Allow safe module imports
    for module_name in SAFE_MODULES:
        try:
            safe_globals[module_name] = __import__(module_name)
        except ImportError:
            pass  # Module not available, skip
    
    return safe_globals


def _unsafe_code_execution_worker(
    generated_code: str,
    test_cases: List[Dict[str, Any]],
    result_queue: "multiprocessing.Queue[float]",
) -> None:
    """
    Worker function for code execution with restricted environment.
    
    Note: Despite the name, this is safer than raw exec() due to:
    - Restricted builtins (no file I/O, no imports, etc.)
    - Multiprocessing isolation
    - Timeout enforcement by parent process
    """
    try:
        # Use safe globals with restricted builtins
        safe_globals = _create_safe_globals()
        local_env: Dict[str, Any] = {}
        
        # Execute in restricted environment
        exec(generated_code, safe_globals, local_env)
        
        # Heuristic: find last defined function
        func_names = [k for k in local_env if callable(local_env[k])]
        if not func_names:
            result_queue.put(0.0)
            return

        func_name = func_names[-1]
        func = local_env[func_name]
        for test in test_cases:
            if func(*test["input"]) != test["output"]:
                result_queue.put(0.0)
                return
        result_queue.put(1.0)
    except Exception:
        result_queue.put(0.0)


class CodeRewardCalculator:
    """
    Reward calculator for Code Generation (Stage 3).
    Executes generated code against test cases with safety timeouts.
    """

    def __init__(self, timeout: float = 2.0) -> None:
        """
        Initialize code reward calculator.

        Args:
            timeout: Execution timeout in seconds
        """
        self.timeout = timeout

    def evaluate_code(
        self, generated_code: str, test_cases: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate python code against test cases.

        Args:
            generated_code: The generated Python code to evaluate
            test_cases: List of test cases with 'input' and 'output' keys

        Returns:
            1.0 if all tests pass, 0.0 otherwise
        """
        # 1. Syntax Check
        try:
            ast.parse(generated_code)
        except SyntaxError:
            return 0.0

        # 2. Execution Sandbox (Multiprocessing for timeout safety)
        queue: "multiprocessing.Queue[float]" = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_unsafe_code_execution_worker,
            args=(generated_code, test_cases, queue),
        )

        try:
            p.start()
            p.join(self.timeout)

            if p.is_alive():
                p.terminate()
                p.join()
                return 0.0

            if not queue.empty():
                return queue.get()
            return 0.0

        except Exception:
            if p.is_alive():
                p.terminate()
                p.join()
            return 0.0

    def compute_rewards(
        self,
        prompts: List[str],
        completions: List[List[str]],
        test_suites: List[List[Dict[str, Any]]],
    ) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
        """
        Compute rewards and info for a batch of code completions.

        Args:
            prompts: List of prompts (not used but kept for API consistency)
            completions: List of completion groups
            test_suites: List of test suites corresponding to each prompt

        Returns:
            Tuple of (rewards, infos) where rewards are binary scores
        """
        rewards: List[List[float]] = []
        infos: List[Dict[str, Any]] = []
        for comp_group, tests in zip(completions, test_suites):
            clean_comps = [self._extract_code(c) for c in comp_group]
            scores = np.array([self.evaluate_code(c, tests) for c in clean_comps])
            rewards.append(scores.tolist())
            # Code stage typically uses standard RL or generic entropy weight
            infos.append({"entropy_weight": 1.0})
        return rewards, infos

    def _extract_code(self, text: str) -> str:
        """
        Extract Python code from markdown code blocks.

        Args:
            text: Text that may contain code blocks

        Returns:
            Extracted code or original text
        """
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        return text


if __name__ == "__main__":
    monitor = TrainingMonitor(output_dir="monitoring")
    for step in range(0, 1001, 10):
        loss = 5.0 * np.exp(-step / 500) + np.random.normal(0, 0.01)
        monitor.record_step(
            step=step, loss=loss, learning_rate=5e-6, samples_processed=step * 8
        )
    monitor.generate_report("demo")
    monitor.plot_training_curves("demo")
    reward_calc = MGPORewardCalculator(lambda_param=4.0)
    test_cases = [
        ("2x + 3 = 7\nx = 2", "x = 2", 1.0),
        ("The answer is 42", "42", 1.0),
        ("x = 5", "x = 3", 0.0),
    ]
    print("Testing MGPORewardCalculator:")
    for gen, ref, expected in test_cases:
        reward = reward_calc.evaluate_solution(gen, ref)
        status = "✓" if reward == expected else "✗"
        print(f"{status} Generated: '{gen}' vs Reference: '{ref}' -> Reward: {reward}")
