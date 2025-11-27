import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List
from pathlib import Path


class AttentionVisualizer:
    """Visualize attention patterns in transformer models."""

    def __init__(
        self, model: Any, tokenizer: Any, output_dir: str = "attention_viz"
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.attention_maps: Dict[str, Any] = {}
        self.hooks: List[Any] = []

    def register_hooks(self) -> None:
        """Register hooks to capture attention weights."""

        def create_hook(name: str) -> Any:
            def hook(module: Any, input: Any, output: Any) -> None:
                if isinstance(output, tuple):
                    attention = output[0]
                else:
                    attention = output
                self.attention_maps[name] = attention.detach().cpu()

            return hook

        for name, module in self.model.named_modules():
            if "self_attn" in name or "attention" in name.lower():
                hook = module.register_forward_hook(create_hook(name))
                self.hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def visualize_attention(
        self,
        text: str,
        layer_idx: int = -1,
        head_idx: int = 0,
        max_seq_len: int = 128,
    ) -> np.ndarray:
        """
        Visualize attention patterns for a given input.
        """
        tokens = self.tokenizer(
            text, return_tensors="pt", max_length=max_seq_len, truncation=True
        )
        self.register_hooks()
        with torch.no_grad():
            _ = self.model(**tokens)
        self.remove_hooks()
        layer_names = list(self.attention_maps.keys())
        if layer_idx < 0:
            layer_idx = len(layer_names) + layer_idx
        target_layer = layer_names[layer_idx]
        attention = self.attention_maps[target_layer]
        attention_np: np.ndarray = attention[0, head_idx, :, :].numpy()
        token_ids = tokens["input_ids"][0].tolist()
        token_strings = [self.tokenizer.decode([tid]) for tid in token_ids]
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            attention_np,
            xticklabels=token_strings,
            yticklabels=token_strings,
            cmap="viridis",
            ax=ax,
            cbar_kws={"label": "Attention Weight"},
        )
        ax.set_title(f"Attention Layer {layer_idx}, Head {head_idx}")
        ax.set_xlabel("Query Tokens")
        ax.set_ylabel("Key Tokens")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        output_file = self.output_dir / f"attention_layer{layer_idx}_head{head_idx}.png"
        plt.savefig(output_file, dpi=150)
        print(f"Attention map saved to: {output_file}")
        plt.close()
        return attention_np


class GenerationAnalyzer:
    """Analyze model generation patterns and diversity."""

    def __init__(
        self, model: Any, tokenizer: Any, output_dir: str = "generation_viz"
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def analyze_diversity(
        self, prompt: str, num_generations: int = 16, max_length: int = 256
    ) -> Dict[str, Any]:
        """Analyze diversity of multiple generations."""
        solutions: List[str] = []
        for _ in range(num_generations):
            output = self.model.generate(
                self.tokenizer(prompt, return_tensors="pt").input_ids,
                max_length=max_length,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )
            solution = self.tokenizer.decode(output[0], skip_special_tokens=True)
            solutions.append(solution)
        unique_count = len(set(solutions))
        unique_pct = 100.0 * unique_count / num_generations
        lengths = [len(s.split()) for s in solutions]
        tokenized = [set(s.split()) for s in solutions]
        overlaps = []
        for i in range(len(tokenized)):
            for j in range(i + 1, len(tokenized)):
                intersection = len(tokenized[i] & tokenized[j])
                union = len(tokenized[i] | tokenized[j])
                jaccard = intersection / union if union > 0 else 0
                overlaps.append(jaccard)
        avg_overlap = np.mean(overlaps) if overlaps else 0.0
        return {
            "num_generations": num_generations,
            "unique_solutions": unique_count,
            "uniqueness_pct": unique_pct,
            "avg_length": np.mean(lengths),
            "length_std": np.std(lengths),
            "avg_token_overlap": avg_overlap,
            "solutions": solutions,
        }

    def plot_diversity_analysis(
        self, analysis: Dict[str, Any], problem_name: str = ""
    ) -> None:
        """Visualize diversity analysis."""
        solutions = analysis["solutions"]
        lengths = [len(s) for s in solutions]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(lengths, bins=15, alpha=0.7, color="blue", edgecolor="black")
        axes[0].axvline(
            np.mean(lengths),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(lengths):.1f}",
        )
        axes[0].set_xlabel("Solution Length (tokens)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"Generation Length Distribution {problem_name}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        metrics_text = (
            f"Unique Solutions: {analysis['unique_solutions']}/"
            f"{analysis['num_generations']} "
            f"({analysis['uniqueness_pct']:.1f}%)\n"
            f"Avg Token Overlap: {analysis['avg_token_overlap']:.3f}\n"
            f"Length Std Dev: {analysis['length_std']:.2f}"
        )
        axes[1].text(
            0.5,
            0.5,
            metrics_text,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )
        axes[1].axis("off")
        plt.tight_layout()
        output_file = self.output_dir / f"diversity_{problem_name}.png"
        plt.savefig(output_file, dpi=150)
        print(f"Diversity plot saved to: {output_file}")
        plt.close()


class LossLandscapeVisualizer:
    """Visualize loss landscape around current model weights."""

    @staticmethod
    def compute_loss_landscape(
        model: Any,
        tokenizer: Any,
        dataset: Any,
        directions: int = 20,
        magnitude: float = 1.0,
        device: str = "cuda",
    ) -> Any:
        """
        Compute loss landscape by perturbing weights in random directions.
        """
        original_weights = {
            name: param.clone() for name, param in model.named_parameters()
        }
        directions_dict = {
            name: torch.randn_like(param) / (param.numel() ** 0.5)
            for name, param in model.named_parameters()
        }
        losses: List[float] = []
        alphas = np.linspace(-magnitude, magnitude, directions)
        for alpha in alphas:
            for name, param in model.named_parameters():
                param.data = (
                    original_weights[name] + alpha * directions_dict[name]
                ).to(device)
            model.eval()
            batch_loss = 0.0
            count = 0
            with torch.no_grad():
                for sample in dataset.take(5):
                    inputs = tokenizer(sample["problem"], return_tensors="pt").to(
                        device
                    )
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    batch_loss += outputs.loss.item()
                    count += 1
            avg_loss = batch_loss / max(count, 1)
            losses.append(avg_loss)
        for name, param in model.named_parameters():
            param.data = original_weights[name]
        return alphas, losses

    @staticmethod
    def plot_loss_landscape(alphas: Any, losses: Any) -> None:
        """Plot loss landscape."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(alphas, losses, marker="o", linewidth=2, markersize=4)
        ax.set_xlabel("Weight Perturbation Magnitude (α)")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Landscape Around Current Weights")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("loss_landscape.png", dpi=150)
        print("Loss landscape saved to: loss_landscape.png")
        plt.close()


if __name__ == "__main__":
    print("Visualization tools ready for integration into training pipeline.")
