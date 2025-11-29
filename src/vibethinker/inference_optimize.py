"""
Optimization techniques for VibeThinker inference.
"""

from typing import Any, List

import torch


class OptimizedInference:
    """Optimized inference wrapper for VibeThinker."""

    def __init__(self, model: Any, tokenizer: Any, device: str = "cuda") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.eval()

        if hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def generate_optimized(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        num_beams: int = 1,
        temperature: float = 0.7,
        use_cache: bool = True,
    ) -> str:
        """
        Generate with optimizations.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    use_cache=use_cache,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        return str(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

    def batch_generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Batch generation for higher throughput."""
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                outputs = self.model.generate(
                    inputs["input_ids"],
                    **kwargs,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        return [
            str(self.tokenizer.decode(out, skip_special_tokens=True)) for out in outputs
        ]
