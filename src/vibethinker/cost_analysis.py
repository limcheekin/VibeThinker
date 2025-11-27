
"""
Complete cost analysis and resource profiling for VibeThinker training.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
import json


@dataclass
class TrainingStageProfile:
    """Profile data for a training stage."""

    stage_name: str
    gpu_type: str
    gpu_count: int
    batch_size: int
    num_steps: int
    estimated_gpu_hours: float
    estimated_compute_cost: float
    estimated_energy_cost: float
    estimated_total_cost: float
    peak_memory_gb: float
    avg_throughput_samples_sec: float


class CostAnalyzer:
    """Comprehensive cost analysis for VibeThinker training."""

    GPU_SPECS: Dict[str, Dict[str, Any]] = {
        "H100": {
            "memory_gb": 80,
            "power_draw_w": 350,
            "cloud_price_hourly": 4.00,
            "flops_fp32": 989e12,
        },
        "A100": {
            "memory_gb": 40,
            "power_draw_w": 250,
            "cloud_price_hourly": 2.50,
            "flops_fp32": 312e12,
        },
        "H800": {
            "memory_gb": 80,
            "power_draw_w": 350,
            "cloud_price_hourly": 3.50,
            "flops_fp32": 989e12,
        },
        "L4": {
            "memory_gb": 24,
            "power_draw_w": 72,
            "cloud_price_hourly": 0.35,
            "flops_fp32": 121e12,
        },
    }

    ENERGY_COST_PER_KWH: Dict[str, float] = {
        "us-west": 0.12,
        "us-east": 0.14,
        "eu-west": 0.25,
        "asia": 0.20,
    }

    def __init__(self, gpu_type: str = "H100", region: str = "us-west") -> None:
        self.gpu_type = gpu_type
        self.region = region
        self.gpu_spec = self.GPU_SPECS.get(gpu_type, self.GPU_SPECS["H100"])
        self.energy_cost_per_kwh = self.ENERGY_COST_PER_KWH.get(region, 0.15)

    def estimate_tokens_per_second(
        self, batch_size: int, seq_len: int, model_params: float = 0.6e9
    ) -> float:
        """
        Estimate tokens/second throughput.
        """
        utilization = 0.5
        overhead_factor = 2.0
        tokens_per_sec: float = (
            (self.gpu_spec["flops_fp32"] * utilization)
            / (model_params * overhead_factor)
        )
        batch_scaling = min(batch_size / 16, 2.0)
        tokens_per_sec *= batch_scaling
        return tokens_per_sec

    def estimate_training_time(
        self, num_steps: int, batch_size: int, seq_len: int = 1024
    ) -> Dict[str, float]:
        """
        Estimate training duration for a phase.
        """
        tokens_per_sec = self.estimate_tokens_per_second(batch_size, seq_len)
        total_tokens = float(num_steps * batch_size * seq_len)
        time_seconds = total_tokens / tokens_per_sec
        time_hours = time_seconds / 3600
        time_days = time_hours / 24
        return {
            "estimated_seconds": time_seconds,
            "estimated_hours": time_hours,
            "estimated_days": time_days,
            "tokens_per_second": tokens_per_sec,
            "total_tokens": total_tokens,
        }

    def estimate_cost(
        self, num_steps: int, batch_size: int, gpu_count: int = 1, seq_len: int = 1024
    ) -> Dict[str, float]:
        """
        Estimate total training cost.
        """
        time_est = self.estimate_training_time(num_steps, batch_size, seq_len)
        hours = time_est["estimated_hours"]
        compute_cost = hours * self.gpu_spec["cloud_price_hourly"] * gpu_count
        power_kw = (self.gpu_spec["power_draw_w"] * 0.85) / 1000 * gpu_count
        energy_kwh = power_kw * hours
        energy_cost = energy_kwh * self.energy_cost_per_kwh
        misc_cost = hours * 0.10 * gpu_count
        total_cost = compute_cost + energy_cost + misc_cost
        return {
            "compute_hours": hours,
            "compute_cost": compute_cost,
            "energy_kwh": energy_kwh,
            "energy_cost": energy_cost,
            "misc_cost": misc_cost,
            "total_cost": total_cost,
            "cost_per_step": total_cost / num_steps,
        }

    def generate_full_pipeline_estimate(self) -> Dict[str, Any]:
        """
        Generate cost estimate for entire VibeThinker pipeline.
        """
        stages: Dict[str, Dict[str, Any]] = {
            "spectrum_algebra": {"steps": 1500, "batch_size": 4, "description": "SFT - Algebra specialist"},
            "spectrum_geometry": {"steps": 1500, "batch_size": 4, "description": "SFT - Geometry specialist"},
            "spectrum_calculus": {"steps": 1500, "batch_size": 4, "description": "SFT - Calculus specialist"},
            "spectrum_statistics": {"steps": 1500, "batch_size": 4, "description": "SFT - Statistics specialist"},
            "signal_stage1": {"steps": 500, "batch_size": 2, "description": "RL - 4K context window"},
            "signal_stage2": {"steps": 500, "batch_size": 2, "description": "RL - 8K context window"},
            "signal_stage3": {"steps": 500, "batch_size": 2, "description": "RL - 32K context window"},
        }
        total_cost = 0.0
        stage_profiles: List[TrainingStageProfile] = []
        for stage_name, config in stages.items():
            cost_est = self.estimate_cost(
                config["steps"], config["batch_size"], gpu_count=1, seq_len=2048
            )
            profile = TrainingStageProfile(
                stage_name=stage_name,
                gpu_type=self.gpu_type,
                gpu_count=1,
                batch_size=config["batch_size"],
                num_steps=config["steps"],
                estimated_gpu_hours=cost_est["compute_hours"],
                estimated_compute_cost=cost_est["compute_cost"],
                estimated_energy_cost=cost_est["energy_cost"],
                estimated_total_cost=cost_est["total_cost"],
                peak_memory_gb=self.gpu_spec["memory_gb"] * 0.6,
                avg_throughput_samples_sec=(
                    cost_est["compute_hours"]
                    * 3600
                    / config["steps"]
                    / config["batch_size"]
                ),
            )
            stage_profiles.append(profile)
            total_cost += cost_est["total_cost"]
        return {
            "gpu_type": self.gpu_type,
            "region": self.region,
            "stages": stage_profiles,
            "total_estimated_cost": total_cost,
            "total_estimated_hours": sum(s.estimated_gpu_hours for s in stage_profiles),
            "total_estimated_days": sum(s.estimated_gpu_hours for s in stage_profiles) / 24,
        }

    def print_cost_report(self) -> None:
        """Print formatted cost report."""
        estimate = self.generate_full_pipeline_estimate()
        print("\n" + "=" * 80)
        print("VIBETHINKER TRAINING COST ANALYSIS")
        print("=" * 80)
        print(f"GPU Type: {estimate['gpu_type']}")
        print(f"Region: {estimate['region']}")
        print(f"Hourly Rate: ${self.gpu_spec['cloud_price_hourly']:.2f}")
        print(f"Energy Cost: ${self.energy_cost_per_kwh:.3f}/kWh")
        print("\n" + "-" * 80)
        print(f"{'Stage':<25} {'Steps':>8} {'Hours':>10} {'Cost':>12}")
        print("-" * 80)
        for stage in estimate["stages"]:
            print(
                f"{stage.stage_name:<25} "
                f"{stage.num_steps:>8} "
                f"{stage.estimated_gpu_hours:>10.1f} "
                f"${stage.estimated_total_cost:>11.2f}"
            )
        print("-" * 80)
        print(
            f"{'TOTAL':<25} "
            f"{sum(s.num_steps for s in estimate['stages']):>8} "
            f"{estimate['total_estimated_hours']:>10.1f} "
            f"${estimate['total_estimated_cost']:>11.2f}"
        )
        print("-" * 80)
        print(
            f"\nEstimated Duration: {estimate['total_estimated_days']:.1f} days "
            f"({estimate['total_estimated_hours']:.1f} hours)"
        )
        print("\n" + "=" * 80)
        print("COMPARISON WITH PAPER")
        print("=" * 80)
        original_cost_1_5b = 7800
        original_gpu_h800 = 3900
        our_cost = estimate["total_estimated_cost"]
        our_hours = estimate["total_estimated_hours"]
        print(
            f"VibeThinker (1.5B): ${original_cost_1_5b:.0f} on H800 "
            f"({original_gpu_h800:.0f} hours)"
        )
        print(
            f"VibeThinker-0.6B (ours): ${our_cost:.0f} on {self.gpu_type} "
            f"({our_hours:.0f} hours)"
        )
        print(f"Cost reduction: {(1 - our_cost / original_cost_1_5b) * 100:.1f}%")
        print("=" * 80 + "\n")

def compare_gpu_options() -> None:
    """Compare costs across different GPU types."""
    print("\n" + "=" * 100)
    print("GPU COST COMPARISON (7000 training steps, 2K context)")
    print("=" * 100)
    print(
        f"{'GPU Type':<15} {'Hourly Rate':>15} {'Est. Hours':>15} "
        f"{'Est. Cost':>15} {'Cost/Step':>15}"
    )
    print("-" * 100)
    for gpu_type in ["L4", "A100", "H800", "H100"]:
        analyzer = CostAnalyzer(gpu_type=gpu_type)
        cost_est = analyzer.estimate_cost(
            num_steps=7000, batch_size=4, gpu_count=1, seq_len=2048
        )
        print(
            f"{gpu_type:<15} "
            f"${analyzer.gpu_spec['cloud_price_hourly']:>14.2f} "
            f"{cost_est['compute_hours']:>14.1f}h "
            f"${cost_est['total_cost']:>14.2f} "
            f"${cost_est['cost_per_step']:>14.4f}"
        )
    print("=" * 100 + "\n")

if __name__ == "__main__":
    analyzer = CostAnalyzer(gpu_type="H100", region="us-west")
    estimate = analyzer.generate_full_pipeline_estimate()
    analyzer.print_cost_report()
    compare_gpu_options()
    estimate_json = {
        "gpu_type": estimate["gpu_type"],
        "region": estimate["region"],
        "total_estimated_cost": estimate["total_estimated_cost"],
        "total_estimated_hours": estimate["total_estimated_hours"],
        "total_estimated_days": estimate["total_estimated_days"],
        "stages": [
            {
                "name": s.stage_name,
                "steps": s.num_steps,
                "hours": s.estimated_gpu_hours,
                "cost": s.estimated_total_cost,
            }
            for s in estimate["stages"]
        ],
    }
    with open("cost_estimate.json", "w") as f:
        json.dump(estimate_json, f, indent=2)
    print("Cost estimate saved to: cost_estimate.json")
