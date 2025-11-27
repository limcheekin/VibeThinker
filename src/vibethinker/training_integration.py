"""
Integration of MGPO trainer into the main training loop.
"""

import os
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from vibethinker.grpo_custom import MGPOTrainerWithEntropyWeighting
from vibethinker.monitor import MGPORewardCalculator, TrainingMonitor
from vibethinker.visualization import GenerationAnalyzer


def train_signal_phase_with_mgpo(
    spectrum_model_path: str,
    train_dataset: Any,
    tokenizer: Any,
    max_steps: int = 2000,
    eval_every: int = 200,
    output_dir: str = "outputs/vibethinker_rl_mgpo",
) -> str:
    """
    Train signal phase with proper MGPO implementation.
    """
    from unsloth import FastLanguageModel

    os.makedirs(output_dir, exist_ok=True)
    model, _ = FastLanguageModel.from_pretrained(
        spectrum_model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    class Config:
        learning_rate = 5e-6
        adam_beta1 = 0.9
        adam_beta2 = 0.99
        max_completion_length = 1024
        max_steps = max_steps

    config = Config()
    reward_calc = MGPORewardCalculator(lambda_param=4.0)
    trainer = MGPOTrainerWithEntropyWeighting(
        model=model,
        tokenizer=tokenizer,
        config=config,
        reward_calculator=reward_calc,
        device="cuda",
    )
    monitor = TrainingMonitor(output_dir="monitoring")
    analyzer = GenerationAnalyzer(model, tokenizer)
    print("\n" + "=" * 70)
    print("SIGNAL PHASE TRAINING WITH MGPO")
    print("=" * 70)
    dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True,
    )
    global_step = 0
    samples_processed = 0
    for epoch in range(int(max_steps / len(dataloader)) + 1):
        for batch_idx, batch in enumerate(dataloader):
            if global_step >= max_steps:
                break
            G = 8
            all_completions: List[List[str]] = []
            for problem in batch["problem"]:
                prompt_text = (
                    "Solve the following problem step by step:\n\n"
                    f"{problem}\n\nSolution:"
                )
                problem_completions: List[str] = []
                for _ in range(G):
                    inputs = tokenizer(prompt_text, return_tensors="pt").to(
                        model.device
                    )
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs["input_ids"],
                            max_length=config.max_completion_length,
                            temperature=0.7,
                            top_p=0.95,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    completion = completion[len(prompt_text):].strip()
                    problem_completions.append(completion)
                all_completions.append(problem_completions)
            batch_dict: Dict[str, Any] = {
                "prompts": batch["problem"],
                "completions": all_completions,
                "reference_answers": batch["answer"],
            }
            metrics = trainer.training_step(batch_dict)
            monitor.record_step(
                step=global_step,
                loss=metrics["loss"],
                learning_rate=config.learning_rate,
                samples_processed=samples_processed,
            )
            samples_processed += len(batch["problem"])
            global_step += 1
            if global_step % eval_every == 0:
                print(f"\n{'='*70}")
                print(f"Step {global_step}: Loss = {metrics['loss']:.4f}")
                print(f"Reward Mean: {metrics['reward_mean']:.4f}")
                print(f"Entropy Weight Mean: {metrics['entropy_weight_mean']:.4f}")
                print(f"Advantage Mean: {metrics['advantage_mean']:.4f}")
                print(f"{'='*70}")
                checkpoint_path = f"{output_dir}/checkpoint-{global_step}"
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                if global_step % (eval_every * 2) == 0:
                    test_prompt = "Solve: 2x + 3 = 7"
                    analysis = analyzer.analyze_diversity(
                        test_prompt, num_generations=8
                    )
                    analyzer.plot_diversity_analysis(analysis, f"step{global_step}")
    final_path = f"{output_dir}/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    monitor.generate_report("signal_phase")
    monitor.plot_training_curves("signal_phase")
    print(f"\n✓ Training complete! Model saved to: {final_path}")
    return final_path


if __name__ == "__main__":
    print("MGPO training integration ready.")
