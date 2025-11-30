#!/usr/bin/env python3
import argparse

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer  # type: ignore[attr-defined]
from unsloth import FastLanguageModel


def train_specialist(
    domain: str, data_path: str, output_dir: str, max_steps: int, base_model_id: str
) -> None:
    # Spectrum Phase (SFT) typically uses a smaller context than the Signal Phase (RL)
    # Adjust this based on your GPU VRAM. 4096 is standard for SFT.
    max_seq_length = 4096

    print(f"Loading Base Model: {base_model_id} for domain: {domain}...")

    # 1. Load Model using Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # Efficient loading
        dtype=None,
    )

    # 2. Add LoRA Adapters
    # "target_modules" are set for Qwen/Llama architectures.
    # Unsloth handles this automatically usually, but specifying them ensures coverage.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # optimize VRAM
        random_state=3407,
    )

    # 3. Load Data
    dataset = load_dataset("json", data_files=data_path, split="train")

    # 4. Standardize Prompt Format (CoT style)
    def formatting_prompts_func(examples: dict[str, list[str]]) -> str:
        problems = examples["problem"]
        answers = examples["answer"]
        texts = []
        for p, a in zip(problems, answers):
            # Standard Question-Answer format with EOS token
            text = f"Problem: {p}\nAnswer: {a}" + tokenizer.eos_token
            texts.append(text)
        # Return joined text as single string (required by SFTTrainer)
        return "\n\n".join(texts)

    # 5. Initialize Trainer
    print(f"Starting training for {max_steps} steps...")

    # Use SFTConfig which extends TrainingArguments with SFT-specific parameters
    sft_config = SFTConfig(
        # SFT-specific parameters
        dataset_text_field="text",
        max_length=max_seq_length,
        # Training arguments
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch size = 32
        warmup_steps=50,
        max_steps=max_steps,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=f"{output_dir}/{domain}_specialist",
        # CRITICAL for VibeThinker: Save intermediate checkpoints
        # so diversity_probing.py can select the best one later.
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,  # Keep only last 3 to save disk space
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        args=sft_config,
    )

    # 6. Train
    trainer.train()

    # 7. Save Final Model
    final_path = f"{output_dir}/{domain}_specialist/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Training complete. Checkpoints saved in {output_dir}/{domain}_specialist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train VibeThinker Specialist (Spectrum Phase)"
    )

    parser.add_argument(
        "--domain", required=True, help="Domain name (e.g., algebra, geometry)"
    )
    parser.add_argument(
        "--data", required=True, help="Path to the .jsonl training data file"
    )

    # Model Configuration
    parser.add_argument(
        "--base-model",
        default="unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
        help="HuggingFace model ID or local path to the base model",
    )

    parser.add_argument(
        "--out", default="checkpoints", help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Total training steps"
    )

    args = parser.parse_args()

    train_specialist(args.domain, args.data, args.out, args.max_steps, args.base_model)
