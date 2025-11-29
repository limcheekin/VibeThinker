"""
Complete VibeThinker training pipeline with monitoring, visualization,
proper MGPO, and cost tracking.
"""

import argparse
import json
import os
from typing import Any, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from vibethinker.cost_analysis import CostAnalyzer
from vibethinker.debugger import PerformanceInspector, TrainingDebugger
from vibethinker.grpo_custom import MGPOTrainerWithEntropyWeighting
from vibethinker.monitor import MGPORewardCalculator, TrainingMonitor
from vibethinker.visualization import GenerationAnalyzer


def train_signal_phase_complete(
    spectrum_model_path: str,
    train_dataset: Any,
    val_dataset: Any,
    tokenizer: Any,
    output_dir: str = "outputs/vibethinker_complete",
    gpu_type: str = "H100",
    max_steps: int = 2000,
) -> Tuple[str, Any, Any]:
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
    # 1. Load Base Model in 4-bit
    model, _ = FastLanguageModel.from_pretrained(
        spectrum_model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    # 2. CRITICAL: Add LoRA Adapters.
    # 4-bit models are frozen; we need LoRA to have trainable parameters.
    print("Applying LoRA adapters for training...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print("Initializing monitors and debuggers...")
    monitor = TrainingMonitor(output_dir=f"{output_dir}/monitoring")
    debugger = TrainingDebugger(log_dir=f"{output_dir}/debug_logs")
    analyzer = GenerationAnalyzer(
        model, tokenizer, output_dir=f"{output_dir}/generation_viz"
    )
    inspector = PerformanceInspector()

    print("\n" + "=" * 70)
    print("BASELINE PERFORMANCE BENCHMARKS")
    print("=" * 70)
    # Ensure model is in training mode for profiling
    FastLanguageModel.for_training(model)
    memory_profile = inspector.profile_gpu_memory(model, batch_size=4, seq_len=1024)
    print(f"Peak GPU Memory: {memory_profile['peak_memory_gb']:.2f} GB")

    # Note: Throughput benchmark might be slow, optional
    throughput = inspector.benchmark_throughput(model, tokenizer)
    print(f"Throughput: {throughput['throughput_tokens_per_sec']:.0f} tokens/sec")

    class TrainingConfig:
        def __init__(self, max_steps: int) -> None:
            self.learning_rate: float = 5e-6
            self.adam_beta1: float = 0.9
            self.adam_beta2: float = 0.99
            self.max_completion_length: int = 1024
            self.max_steps: int = max_steps
            self.eval_every: int = 200
            self.log_every: int = 10

    config = TrainingConfig(max_steps)

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

    # Use DataLoader for correct batching (fix for .batch() error)
    batch_size = 4
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    global_step = 0
    samples_processed = 0

    # Determine epochs needed
    steps_per_epoch = len(dataloader)
    if steps_per_epoch == 0:
        raise ValueError("Dataset is empty!")
    num_epochs = (max_steps // steps_per_epoch) + 1

    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            if global_step >= max_steps:
                break
            try:
                # --- ACTOR PHASE: Generate Completions ---
                prompts = batch["problem"]
                reference_answers = batch["answer"]

                G = 4  # Number of generations per prompt for GRPO
                all_completions = []

                # Switch to eval mode for generation (disables dropout, etc.)
                model.eval()

                for problem in prompts:
                    # Construct Prompt
                    prompt_text = (
                        "Solve the following problem step by step:\n\n"
                        f"{problem}\n\nSolution:"
                    )

                    # Tokenize
                    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
                    input_len = inputs["input_ids"].shape[1]

                    problem_completions = []
                    for _ in range(G):
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_length=config.max_completion_length,
                                temperature=0.8,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )

                        # ROBUST DECODING: Slice off the input prompt tokens
                        generated_tokens = outputs[0][input_len:]
                        completion = tokenizer.decode(
                            generated_tokens, skip_special_tokens=True
                        ).strip()
                        problem_completions.append(completion)

                    all_completions.append(problem_completions)

                # --- LEARNER PHASE: Update Policy ---
                # Switch back to train mode (enables LoRA gradients)
                model.train()

                batch_dict = {
                    "prompts": prompts,
                    "completions": all_completions,
                    "reference_answers": reference_answers,
                }

                metrics = trainer.training_step(batch_dict)

                # --- MONITORING ---
                grad_health = debugger.check_gradient_health(model)
                if grad_health["issues"]:
                    # Just warn, don't crash unless severe
                    pass

                loss_val = metrics["loss"]
                loss_ok = debugger.check_loss_sanity(
                    torch.tensor(loss_val), global_step
                )
                if not loss_ok:
                    debugger.logger.error(
                        f"Loss sanity check failed at step {global_step}"
                    )
                    # Decide whether to break or skip. Skipping step here:
                    continue

                samples_processed += len(prompts)
                monitor.record_step(
                    step=global_step,
                    loss=loss_val,
                    learning_rate=config.learning_rate,
                    gradient_norm=grad_health["total_norm"],
                    samples_processed=samples_processed,
                )

                global_step += 1
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

                # --- EVALUATION & CHECKPOINTING ---
                if global_step % config.eval_every == 0:
                    print(f"\n{'=' * 70}")
                    print(f"Step {global_step} / {max_steps}")
                    print(f"Loss: {loss_val:.4f}")
                    print(f"Reward Mean: {metrics['reward_mean']:.4f}")
                    print(f"Entropy Weight Mean: {metrics['entropy_weight_mean']:.4f}")
                    cost = monitor.metrics_history[-1].estimated_cost_usd
                    print(f"Cumulative Cost: ${cost:.2f}")
                    print(f"{'=' * 70}")

                    checkpoint_path = f"{output_dir}/checkpoints/step-{global_step}"
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)

                    # Visualization
                    if global_step % (config.eval_every * 2) == 0:
                        test_prompt = "Solve: 2x + 3 = 7"
                        analysis = analyzer.analyze_diversity(
                            test_prompt, num_generations=8
                        )
                        analyzer.plot_diversity_analysis(
                            analysis, problem_name=f"step{global_step}"
                        )

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
        "gpu_type": gpu_type,
    }
    with open(f"{output_dir}/final_analysis.json", "w") as f:
        json.dump(final_analysis, f, indent=2)

    return final_path, monitor, debugger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectrum-path", type=str, required=True)
    parser.add_argument(
        "--output-dir", type=str, default="outputs/vibethinker_complete"
    )
    parser.add_argument("--gpu-type", type=str, default="H100")
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()

    train_dataset = load_dataset(
        "json", data_files="data/algebra_train.jsonl", split="train"
    )
    val_dataset = load_dataset(
        "json", data_files="data/algebra_val.jsonl", split="train"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.spectrum_path)  # type: ignore[no-untyped-call]

    train_signal_phase_complete(
        spectrum_model_path=args.spectrum_path,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        gpu_type=args.gpu_type,
        max_steps=args.max_steps,
    )
