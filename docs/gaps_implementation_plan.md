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

### Manual Verification
-   Inspect logs to see "KL penalty" term in training metrics.
-   Check VRAM usage to ensure Reference Model fits.
