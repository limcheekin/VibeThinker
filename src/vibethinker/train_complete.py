"""
Complete VibeThinker training pipeline with monitoring, visualization,
proper MGPO, and cost tracking.
"""

import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from typing import Any, Tuple

from vibethinker.cost_analysis import CostAnalyzer
from vibethinker.debugger import TrainingDebugger, PerformanceInspector
from vibethinker.grpo_custom import MGPOTrainerWithEntropyWeighting
from vibethinker.monitor import MGPORewardCalculator, TrainingMonitor
from vibethinker.visualization import AttentionVisualizer, GenerationAnalyzer


def train_signal_phase_complete(
    spectrum_model_path: str,
    train_dataset: Any,
    val_dataset: Any,
    tokenizer: Any,
    output_dir: str = "outputs/vibethinker_complete",
    gpu_type: str = "H100",
    max_steps: int = 2000,
) -> Tuple[Any, Any, Any]:
    """
    Complete training with all debugging, visualization, and cost tracking.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

    print("\n" + "=" * 70)
    print("COST ANALYSIS")
    print("=" * 70)
    cost_analyzer = CostAnalyzer(gpu_type=gpu_type)
    cost_analyzer.generate_full_pipeline_estimate()
    cost_analyzer.print_cost_report()

    print("\nLoading model...")
    model, _ = FastLanguageModel.from_pretrained(
        spectrum_model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    print("Initializing monitors and debuggers...")
    monitor = TrainingMonitor(output_dir=f"{output_dir}/monitoring")
    debugger = TrainingDebugger(log_dir=f"{output_dir}/debug_logs")
    analyzer = GenerationAnalyzer(model, tokenizer, output_dir=f"{output_dir}/generation_viz")
    attention_viz = AttentionVisualizer(
        model, tokenizer, output_dir=f"{output_dir}/attention"
    )
    inspector = PerformanceInspector()

    print("\n" + "=" * 70)
    print("BASELINE PERFORMANCE BENCHMARKS")
    print("=" * 70)
    memory_profile = inspector.profile_gpu_memory(model, batch_size=4, seq_len=1024)
    print(f"Peak GPU Memory: {memory_profile['peak_memory_gb']:.2f} GB")
    throughput = inspector.benchmark_throughput(model, tokenizer)
    print(f"Throughput: {throughput['throughput_tokens_per_sec']:.0f} tokens/sec")

    class TrainingConfig:
        learning_rate = 5e-6
        adam_beta1 = 0.9
        adam_beta2 = 0.99
        max_completion_length = 1024
        max_steps = max_steps
        eval_every = 200
        log_every = 10

    config = TrainingConfig()

    print("\nInitializing MGPO trainer...")
    reward_calc = MGPORewardCalculator(lambda_param=4.0)
    trainer = MGPOTrainerWithEntropyWeighting(
        model=model,
        tokenizer=tokenizer,
        config=config,
        reward_calculator=reward_calc,
        device="cuda",
    )

    print("\n" + "=" * 70)
    print("STARTING SIGNAL PHASE TRAINING WITH MGPO")
    print("=" * 70)
    global_step = 0
    samples_processed = 0
    for epoch in range(int(max_steps / len(train_dataset)) + 1):
        for batch_idx, batch in enumerate(train_dataset.batch(4)):
            if global_step >= max_steps:
                break
            try:
                batch_dict = {
                    "prompts": batch["problem"],
                    "completions": [[]],
                    "reference_answers": batch["answer"],
                }
                metrics = trainer.training_step(batch_dict)
                grad_health = debugger.check_gradient_health(model)
                if grad_health["issues"]:
                    for issue in grad_health["issues"]:
                        debugger.logger.warning(issue)
                loss_ok = debugger.check_loss_sanity(
                    torch.tensor(metrics["loss"]), global_step
                )
                if not loss_ok:
                    debugger.logger.error(
                        f"Loss sanity check failed at step {global_step}"
                    )
                    break
                samples_processed += len(batch["problem"])
                monitor.record_step(
                    step=global_step,
                    loss=metrics["loss"],
                    learning_rate=config.learning_rate,
                    gradient_norm=grad_health["total_norm"],
                    samples_processed=samples_processed,
                )
                global_step += 1
                if global_step % config.eval_every == 0:
                    print(f"\n{'='*70}")
                    print(f"Step {global_step} / {max_steps}")
                    print(f"Loss: {metrics['loss']:.4f}")
                    print(f"Reward Mean: {metrics['reward_mean']:.4f}")
                    print(f"Entropy Weight Mean: {metrics['entropy_weight_mean']:.4f}")
                    print(f"Gradient Norm: {grad_health['total_norm']:.6f}")
                    cost = monitor.metrics_history[-1].estimated_cost_usd
                    print(f"Cumulative Cost: ${cost:.2f}")
                    print(f"{'='*70}")
                    checkpoint_path = f"{output_dir}/checkpoints/step-{global_step}"
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    debugger.logger.info(f"Checkpoint saved: {checkpoint_path}")
                    if global_step % (config.eval_every * 2) == 0:
                        test_prompt = "Solve: 2x + 3 = 7"
                        analysis = analyzer.analyze_diversity(
                            test_prompt, num_generations=8
                        )
                        analyzer.plot_diversity_analysis(
                            analysis, problem_name=f"step{global_step}"
                        )
                    if global_step % (config.eval_every * 4) == 0:
                        try:
                            attention_viz.visualize_attention(
                                test_prompt, layer_idx=-1, head_idx=0
                            )
                        except Exception as e:
                            debugger.logger.warning(f"Attention viz failed: {e}")
            except Exception as e:
                debugger.logger.error(
                    f"Training error at step {global_step}: {e}", exc_info=True
                )
                raise
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    final_path = f"{output_dir}/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    monitor.generate_report("signal_phase_complete")
    monitor.plot_training_curves("signal_phase_complete")
    final_analysis = {
        "total_steps": global_step,
        "total_samples": samples_processed,
        "final_cost": (
            monitor.metrics_history[-1].estimated_cost_usd
            if monitor.metrics_history
            else 0
        ),
        "final_loss": metrics["loss"],
        "gpu_type": gpu_type,
        "model_size": "0.6B",
    }
    with open(f"{output_dir}/final_analysis.json", "w") as f:
        json.dump(final_analysis, f, indent=2)
    debugger.logger.info(f"✓ Training complete! Model saved to: {final_path}")
    debugger.logger.info(f"✓ Reports saved to: {output_dir}")
    return final_path, monitor, debugger


if __name__ == "__main__":
    train_dataset = load_dataset(
        "json", data_files="data/algebra_train.jsonl", split="train"
    )
    val_dataset = load_dataset("json", data_files="data/algebra_val.jsonl", split="train")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    train_signal_phase_complete(
        spectrum_model_path="checkpoints/vibethinker_spectrum",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        output_dir="outputs/vibethinker_complete",
        gpu_type="H100",
        max_steps=2000,
    )
