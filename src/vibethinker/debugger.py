"""
Comprehensive debugging tools for VibeThinker training.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


class TrainingDebugger:
    """Debug training process and identify issues."""

    def __init__(self, log_dir: str = "debug_logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.prev_loss: float = 0.0

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_dir / "debug.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def check_gradient_health(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Inspect gradient statistics."""
        grad_stats: Dict[str, Dict[str, float]] = {}
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_stats[name] = {
                    "norm": grad_norm,
                    "mean": param.grad.data.mean().item(),
                    "std": param.grad.data.std().item(),
                    "min": param.grad.data.min().item(),
                    "max": param.grad.data.max().item(),
                }
                total_norm += grad_norm**2
        total_norm = np.sqrt(total_norm)
        issues: List[str] = []
        if total_norm > 10.0:
            issues.append(f"WARNING: Very large gradient norm ({total_norm:.4f})")
        elif total_norm < 1e-6:
            issues.append(f"WARNING: Very small gradient norm ({total_norm:.6f})")
        for name, stats in grad_stats.items():
            if np.isnan(stats["norm"]) or np.isinf(stats["norm"]):
                issues.append(f"ERROR: NaN/Inf gradient in {name}")
        return {
            "total_norm": total_norm,
            "layer_stats": grad_stats,
            "issues": issues,
        }

    def check_loss_sanity(self, loss: torch.Tensor, step: int) -> bool:
        """Check if loss value is reasonable."""
        loss_val = loss.item()
        if np.isnan(loss_val) or np.isinf(loss_val):
            self.logger.error(f"Step {step}: Loss is NaN/Inf: {loss_val}")
            return False
        if loss_val > 100:
            self.logger.warning(f"Step {step}: Loss is very large: {loss_val:.4f}")
        if not hasattr(self, "prev_loss"):
            self.prev_loss = loss_val
        if loss_val > self.prev_loss * 1.5:
            self.logger.warning(
                f"Step {step}: Loss increased by >50% "
                f"({self.prev_loss:.4f} -> {loss_val:.4f})"
            )
        self.prev_loss = loss_val
        return True

    def check_activation_stats(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Check activations for dead neurons or saturation."""
        activation_stats: Dict[str, Dict[str, Any]] = {}
        dead_neuron_count = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if hasattr(module, "_last_output"):
                    output = module._last_output
                    stats: Dict[str, Any] = {
                        "mean": output.mean().item(),
                        "std": output.std().item(),
                        "min": output.min().item(),
                        "max": output.max().item(),
                    }
                    if output.std() < 1e-6:
                        stats["warning"] = "Potential dead neurons"
                        dead_neuron_count += 1
                    activation_stats[name] = stats
        return {
            "activation_stats": activation_stats,
            "dead_neuron_count": dead_neuron_count,
        }

    def debug_generation(
        self, model: Any, tokenizer: Any, prompt: str, max_length: int = 256
    ) -> Dict[str, Any]:
        """Debug generation quality and issues."""
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=0.7,
            )
            text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        issues: List[str] = []
        words = text.split()
        if len(words) > 0 and len(set(words)) / len(words) < 0.3:
            issues.append("High token repetition detected")
        if len(words) < 10:
            issues.append("Generated text too short")
        elif len(words) > max_length * 0.95:
            issues.append("Hit max_length limit (model may be truncated)")
        if any(token in text.lower() for token in ["<unk>", "[unk]", "error"]):
            issues.append("Contains unknown tokens or errors")
        return {
            "prompt": prompt,
            "generated_text": text,
            "length_tokens": len(words),
            "issues": issues,
        }


class PerformanceInspector:
    """Inspect and optimize training performance."""

    @staticmethod
    def profile_gpu_memory(
        model: Any, batch_size: int = 4, seq_len: int = 1024
    ) -> Dict[str, Any]:
        """Profile GPU memory usage."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dummy_input = torch.randint(0, 50000, (batch_size, seq_len)).cuda()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("profile_logs"),
        ):
            with torch.no_grad():
                _ = model(input_ids=dummy_input)
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        return {
            "peak_memory_gb": peak_memory,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }

    @staticmethod
    def benchmark_throughput(
        model: Any, tokenizer: Any, num_iterations: int = 10
    ) -> Dict[str, Any]:
        """Benchmark training throughput."""
        import time

        model.train()
        dummy_input = torch.randint(0, 50000, (4, 512)).cuda()
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_ids=dummy_input, labels=dummy_input)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(input_ids=dummy_input, labels=dummy_input)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        tokens_per_sec = (num_iterations * 4 * 512) / elapsed
        return {
            "throughput_tokens_per_sec": tokens_per_sec,
            "time_per_iteration_ms": (elapsed / num_iterations) * 1000,
        }


if __name__ == "__main__":
    debugger = TrainingDebugger()
    print("✓ Debugging tools ready for integration.")
    print("  - TrainingDebugger: Monitor gradients, loss, activations")
    print("  - PerformanceInspector: Profile memory and throughput")
