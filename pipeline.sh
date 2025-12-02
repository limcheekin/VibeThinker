#!/bin/bash
set -e

# --- Configuration ---
BASE_MODEL="unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit" 
DATA_DIR="data"
OUTPUT_ROOT="outputs/vibethinker_pipeline"
mkdir -p "$OUTPUT_ROOT"

# ==============================================================================
# PHASE 1: SPECTRUM PHASE (SFT + Selection + Fusion)
# ==============================================================================
# We use the Python orchestrator because it handles:
# 1. Training specialists (calls train_sft_specialist.py internally)
# 2. Diversity Probing (CRITICAL: The bash script in the commentary skipped this!)
# 3. Model Fusion (calls fusion_weighted_average internally)
echo ">>> [Phase 1] Executing Spectrum Phase Orchestrator"

python scripts/train_spectrum_phase.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_ROOT/phase1" \
    --base-model "$BASE_MODEL" \
    --max-steps 1000 \
    --k 8 \
    --num-generations 16

SPECTRUM_FUSED_PATH="$OUTPUT_ROOT/phase1/vibethinker_spectrum_fused"

if [ ! -d "$SPECTRUM_FUSED_PATH" ]; then
    echo "Error: Fused model output not found!"
    exit 1
fi

# ==============================================================================
# PHASE 2: SIGNAL PHASE (RL Curriculum)
# ==============================================================================
# Incorporating the excellent "sed" patch logic from the commentary
# to achieve the 4k -> 16k -> Code curriculum.

TRAIN_SCRIPT="src/vibethinker/train_complete.py"

# --- Stage 1: Math RL (Standard 4k Context) ---
echo ">>> [Phase 2.1] Signal RL - Standard Context (4k)"
STAGE1_OUT="$OUTPUT_ROOT/phase2_stage1_4k"

python "$TRAIN_SCRIPT" \
    --spectrum-path "$SPECTRUM_FUSED_PATH" \
    --output-dir "$STAGE1_OUT" \
    --gpu-type "H100" \
    --max-steps 500

STAGE1_MODEL="$STAGE1_OUT/final"

# --- Stage 2: Math RL (Long Context 16k) ---
echo ">>> [Phase 2.2] Signal RL - Expanding Context to 16k"
STAGE2_OUT="$OUTPUT_ROOT/phase2_stage2_16k"

# 1. Backup script
cp "$TRAIN_SCRIPT" "${TRAIN_SCRIPT}.bak"

# 2. Patch max_seq_length from 4096 to 16384 using sed
# Note: Matches the specific line in train_complete.py provided in text
sed -i 's/max_seq_length=4096/max_seq_length=16384/g' "$TRAIN_SCRIPT"

# 3. Run Training with Stage 1 model as base
python "$TRAIN_SCRIPT" \
    --spectrum-path "$STAGE1_MODEL" \
    --output-dir "$STAGE2_OUT" \
    --gpu-type "H100" \
    --max-steps 500

# 4. Restore script immediately
mv "${TRAIN_SCRIPT}.bak" "$TRAIN_SCRIPT"

STAGE2_MODEL="$STAGE2_OUT/final"

# --- Stage 3: Code Generalization ---
echo ">>> [Phase 2.3] Signal RL - Code Domain"
FINAL_OUT="$OUTPUT_ROOT/phase2_stage3_code"

# Ideally, we would switch the dataset here. train_complete.py hardcodes 
# 'data/algebra_train.jsonl'.
# To be truly complete, we patch the dataset path too:
cp "$TRAIN_SCRIPT" "${TRAIN_SCRIPT}.bak"
sed -i 's|data/algebra_train.jsonl|data/livecodebench.jsonl|g' "$TRAIN_SCRIPT"

python "$TRAIN_SCRIPT" \
    --spectrum-path "$STAGE2_MODEL" \
    --output-dir "$FINAL_OUT" \
    --gpu-type "H100" \
    --max-steps 500

mv "${TRAIN_SCRIPT}.bak" "$TRAIN_SCRIPT"

echo "====================================================================="
echo "Pipeline Complete."
echo "Final Model: $FINAL_OUT/final"
echo "====================================================================="