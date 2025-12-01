# VibeThinker Fixes Implementation Plan

## Goal Description
Address 5 critical gaps identified in the VibeThinker codebase relative to the VibeThinker-1.5B paper claims:
1.  **RL Stability**: Add Reference Model and KL penalty.
2.  **RL Efficiency**: Optimize `old_log_probs` calculation.
3.  **Base Model**: Correct `Qwen3` hallucination to `Qwen2.5-Math-1.5B`.
4.  **Safety**: Improve code execution safety.
5.  **SFT Pipeline**: Implement multi-domain training and fusion.

## User Review Required
> [!IMPORTANT]
> **Security**: The current environment may not support full Docker/gVisor sandboxing. I will implement a best-effort safer execution using restricted globals and multiprocessing isolation, but for true production safety, an external sandbox service is recommended.

> [!WARNING]
> **Compute Requirements**: Switching to `Qwen2.5-Math-1.5B` and adding a Reference Model will increase VRAM usage. Ensure the target hardware supports this (likely requires >24GB VRAM for training, or aggressive quantization).

## Proposed Changes

### RL Stability & Efficiency (`src/vibethinker/grpo_custom.py`)
#### [MODIFY] [grpo_custom.py](file:///media/limcheekin/My%20Passport/ws/py/VibeThinker/src/vibethinker/grpo_custom.py)
-   **Add Reference Model**:
    -   Initialize a frozen copy of the model as `ref_model`.
    -   Use `peft` to disable adapters for the reference model if sharing weights, or load a separate instance if needed (sharing base weights with disabled adapters is more memory efficient).
-   **KL Penalty**:
    -   Update `MGPOLoss` to accept `ref_log_probs`.
    -   Calculate KL divergence: `kl = exp(log_probs - ref_log_probs) - (log_probs - ref_log_probs) - 1` (approx) or standard `log_probs - ref_log_probs`.
    -   Add `beta * KL` to the loss or reward. Standard PPO/GRPO usually subtracts KL from reward: `R = R_outcome - beta * KL`.
-   **Optimize Log Probs**:
    -   Modify `training_step` to accept `old_log_probs` computed during generation (rollout) instead of re-computing them.
    -   Update `train_complete.py` to capture these during generation.

-   **File:** `src/vibethinker/grpo_custom.py`
    -  Even with the correct model, the code **lacks the Reference Model** required for the KL penalty in GRPO/RLHF. Without this, the model will collapse (reward hack).
        -   **Action:** Implement the `ref_model` logic.
        -   **Optimization:** Since `Qwen3-0.6B` is tiny (~0.5GB VRAM), you can safely load a separate copy as the reference model without running out of memory, or use the adapter disabling trick.

        ```python
        # In MGPOTrainer.__init__
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
            # ... args ...
        )
        ```

### Fix Safety 
-   **File:** `src/vibethinker/monitor.py`
    -  The use of `exec()` is still unsafe.
    -   **Action:** Wrap code execution in `RestrictedPython` or strict globals/builtins sanitization.

### Implement Spectrum Orchestrator (Critical - Still Required)
-   **File:** `scripts/train_spectrum_phase.py`
    The "Spectrum-to-Signal" principle (training 4 specialists -> probing -> selecting -> fusing) is described in the paper but **missing from the codebase**. You still need to create this script.

    **Orchestrator Script:**
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

### Base Model & Context (`scripts/train_sft_specialist.py`)
#### [MODIFY] [train_sft_specialist.py](file:///media/limcheekin/My%20Passport/ws/py/VibeThinker/scripts/train_sft_specialist.py)
-   Change default `base-model` to `unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit`.
-   Verify and update `max_seq_length` to support larger contexts if hardware allows (paper mentions 16k->32k, code has 4k).

### Safety (`src/vibethinker/monitor.py`)
#### [MODIFY] [monitor.py](file:///media/limcheekin/My%20Passport/ws/py/VibeThinker/src/vibethinker/monitor.py)
-   Enhance `_unsafe_code_execution_worker`:
    -   Restrict `builtins` in `exec` globals (disable `open`, `import`, etc. where possible, though imports are needed for math).
    -   Since we need some imports (math, numpy), we can't block everything.
    -   Add strict timeout (already present but ensure it works).
    -   *Alternative*: If possible, use a library like `RestrictedPython` if available, otherwise rely on `multiprocessing` isolation and warn user.

### Subdomain Logic (`scripts/train_sft_specialist.py` & New Script)
#### [MODIFY] [train_sft_specialist.py](file:///media/limcheekin/My%20Passport/ws/py/VibeThinker/scripts/train_sft_specialist.py)
-   Ensure it can be called with different data files/domains easily.

#### [NEW] [scripts/train_spectrum_phase.py](file:///media/limcheekin/My%20Passport/ws/py/VibeThinker/scripts/train_spectrum_phase.py)
-   Orchestrator script to:
    1.  Define the 4 domains: Algebra, Geometry, Calculus, Statistics.
    2.  Call `train_sft_specialist.py` for each (sequentially or parallel if resources allow).
    3.  Call `src/vibethinker/model_fusion.py` to merge them.

## Verification Plan

### Automated Tests
-   Run `pytest src/vibethinker/tests/test_grpo_custom.py` (if exists, or create one) to verify KL calculation.
-   Run `pytest src/vibethinker/tests/test_monitor.py` to verify code execution safety (try to run `os.system` and ensure it fails or is blocked).
-   Run the new `scripts/train_spectrum_phase.py` with a tiny dummy dataset to verify the pipeline flow.
-   Run `scripts/train_spectrum_phase.py` (dry-run or short steps) to verify the pipeline.
-   Run `pytest` to ensure no regressions.

### Manual Verification
-   Inspect logs to see "KL penalty" term in training metrics.
-   Check VRAM usage to ensure Reference Model fits.
