I would like to reproducing the methodology (SSP, MGPO) of the paper on a smaller scale by using the `unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit` model. The objective is to see how does it compare to the result of the 1.5B model of the paper.

What do you think of the following comments of the gaps_implementation_plan.md? You can refer to the paper @VibeThinker-1.5B.pdf for more details.

````
Based on the verification of the URL you provided, I must **retract my claim** that "Qwen3" is a hallucination.

I have verified that **`unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit`** is a valid model repository that fits the timeframe of the technical report (Nov 2025). My previous assessment incorrectly flagged this as an error based on outdated knowledge training cutoffs relative to the report's timeline.

### Corrected Assessment
**You are correct.** You should use `unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit` as the base model. This 0.6B model is extremely lightweight, which simplifies the hardware requirements significantly compared to a 1.5B model.

### Updated Implementation Plan
With the base model confirmed, here is the finalized, correct plan to reproduce the paper.

#### 1. Base Model Configuration (Use User's Choice)
**File:** `scripts/train_sft_specialist.py`
Use the exact ID provided. No changes needed to the ID, but ensure the script uses it as the default.

```python
parser.add_argument(
    "--base-model",
    default="unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit", 
    help="HuggingFace model ID"
)
```

#### 2. Fix RL Stability (Critical - Still Required)
**File:** `src/vibethinker/grpo_custom.py`
Even with the correct model, the code **lacks the Reference Model** required for the KL penalty in GRPO/RLHF. Without this, the model will collapse (reward hack).
*   **Action:** Implement the `ref_model` logic.
*   **Optimization:** Since `Qwen3-0.6B` is tiny (~0.5GB VRAM), you can safely load a separate copy as the reference model without running out of memory, or use the adapter disabling trick.

```python
# In MGPOTrainer.__init__
self.ref_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
    # ... args ...
)
```

#### 3. Fix Safety (Critical - Still Required)
**File:** `src/vibethinker/monitor.py`
The use of `exec()` is still unsafe.
*   **Action:** Wrap code execution in `RestrictedPython` or strict globals/builtins sanitization.

#### 4. Implement Spectrum Orchestrator (Critical - Still Required)
**File:** `scripts/train_spectrum_phase.py`
The "Spectrum-to-Signal" principle (training 4 specialists -> probing -> selecting -> fusing) is described in the paper but **missing from the codebase**. You still need to create this script.

**Revised Orchestrator Script:**
```python
import os
import glob
import json
from vibethinker.model_fusion import load_expert_models, fusion_weighted_average
from vibethinker.diversity_probing import DiversityProber

# 4 Domains from the paper
DOMAINS = ["algebra", "geometry", "calculus", "statistics"]
BASE_MODEL = "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit"

def get_best_checkpoint(domain, checkpoints, data_path):
    """
    Paper Logic: Select the intermediate checkpoint with highest Diversity (Pass@K)
    NOT the final checkpoint.
    """
    best_ckpt = None
    best_score = -1.0
    
    # Load validation data
    with open(data_path, 'r') as f:
        probing_data = [json.loads(line) for line in f][:64] 

    for ckpt in checkpoints:
        # Load checkpoint (Probing)
        prober = DiversityProber(ckpt) 
        metrics = prober.probe_domain(probing_data, k=8, num_generations=16)
        
        # SSP Metric: Diversity Score
        score = metrics['diversity_score']
        print(f"  {ckpt}: Pass@8 (Diversity) = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_ckpt = ckpt
            
        del prober # Free VRAM
        
    return best_ckpt

def main():
    selected_experts = []
    
    for domain in DOMAINS:
        print(f"=== Training Domain: {domain} ===")
        # 1. Train Specialist
        # Note: Increase LR for 0.6B model (try 5e-5 instead of 5e-6)
        os.system(f"python scripts/train_sft_specialist.py --domain {domain} --data data/{domain}.jsonl --base-model {BASE_MODEL} --max-steps 1000")
        
        # 2. Select Best Checkpoint (The Spectrum Logic)
        ckpt_dir = f"checkpoints/{domain}_specialist"
        checkpoints = sorted(glob.glob(f"{ckpt_dir}/checkpoint-*"))
        best_ckpt = get_best_checkpoint(domain, checkpoints, f"data/{domain}.jsonl")
        selected_experts.append(best_ckpt)

    # 3. Fuse
    print("=== Fusing Selected Experts ===")
    experts = load_expert_models(selected_experts)
    fused_model = fusion_weighted_average(experts)
    fused_model.save_pretrained("checkpoints/vibethinker_spectrum_fused")

if __name__ == "__main__":
    main()
```

### Final Verdict
With the correction regarding `Qwen3`, the implementation plan is now **correct and complete**. You have the valid model, the fixes for the missing RL algorithms, and the missing orchestration logic required to reproduce the paper's specific methodology.
````