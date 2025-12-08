#!/bin/bash
set -e

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# All parameters can be overridden via environment variables

BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/vibethinker_pipeline}"

# Training hyperparameters
SPECTRUM_STEPS="${SPECTRUM_STEPS:-1000}"
SIGNAL_STEPS="${SIGNAL_STEPS:-500}"
DIVERSITY_K="${DIVERSITY_K:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
MAX_SEQ_LENGTH_STAGE1="${MAX_SEQ_LENGTH_STAGE1:-4096}"
MAX_SEQ_LENGTH_STAGE2="${MAX_SEQ_LENGTH_STAGE2:-16384}"
MAX_SEQ_LENGTH_STAGE3="${MAX_SEQ_LENGTH_STAGE3:-32768}"

# Data preparation settings
DATA_PREP_MAX_PROBLEMS="${DATA_PREP_MAX_PROBLEMS:-500}"
DATA_PREP_N_SOLUTIONS="${DATA_PREP_N_SOLUTIONS:-3}"
DATA_PREP_TEACHER_MODEL="${DATA_PREP_TEACHER_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"

# Decontamination settings (10-gram matching to prevent eval contamination)
DECONTAM_ENABLED="${DECONTAM_ENABLED:-true}"
DECONTAM_EVAL_DATASETS="${DECONTAM_EVAL_DATASETS:-GSM8K MATH}"

mkdir -p "$OUTPUT_ROOT"

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

validate_checkpoint() {
    local path=$1
    local name=$2
    
    echo "Validating checkpoint: $name"
    
    if [ ! -d "$path" ]; then
        echo "Error: $name checkpoint not found at $path"
        exit 1
    fi
    
    if [ ! -f "$path/config.json" ]; then
        echo "Error: Invalid checkpoint (missing config.json) at $path"
        exit 1
    fi
    
    echo "✓ Validated $name: $path"
}

# ==============================================================================
# PHASE 0: DATA PREPARATION
# ==============================================================================
echo ">>> [Phase 0] Preparing Domain Data"

# Check if all required data files exist
REQUIRED_FILES=(
    "$DATA_DIR/algebra_train.jsonl"
    "$DATA_DIR/geometry_train.jsonl"
    "$DATA_DIR/calculus_train.jsonl"
    "$DATA_DIR/statistics_train.jsonl"
    "$DATA_DIR/apps_train.jsonl"
)

MISSING_FILES=false
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Missing: $file"
        MISSING_FILES=true
    fi
done

if [ "$MISSING_FILES" = true ]; then
    echo "Missing domain data files. Running data preparation..."
    
    # Build decontamination arguments if enabled
    if [ "$DECONTAM_ENABLED" = true ]; then
        echo "Decontamination ENABLED (10-gram matching against: $DECONTAM_EVAL_DATASETS)"
        DECONTAM_ARGS="--decontam-eval $DECONTAM_EVAL_DATASETS"
    else
        echo "Decontamination DISABLED"
        DECONTAM_ARGS=""
    fi
    
    python scripts/prepare_spectrum_data.py \
        --hf-id "lighteval/MATH" \
        --output-dir "$DATA_DIR" \
        --teacher-backend "hf" \
        --teacher-model "$DATA_PREP_TEACHER_MODEL" \
        --n-solutions "$DATA_PREP_N_SOLUTIONS" \
        --verify \
        --max-problems "$DATA_PREP_MAX_PROBLEMS" \
        $DECONTAM_ARGS
    
    if [ $? -ne 0 ]; then
        echo "Error: Data preparation failed!"
        exit 1
    fi
    echo "✓ Data preparation complete"
else
    echo "✓ All domain data files found. Skipping data preparation."
fi

# ==============================================================================
# PHASE 1: SPECTRUM PHASE (SFT + Selection + Fusion)
# ==============================================================================
# We use the Python orchestrator because it handles:
# 1. Training specialists (calls train_sft_specialist.py internally)
# 2. Diversity Probing (CRITICAL: The bash script in the commentary skipped this!)
# 3. Model Fusion (calls fusion_weighted_average internally)
echo ""
echo ">>> [Phase 1] Executing Spectrum Phase Orchestrator"

python scripts/train_spectrum_phase.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_ROOT/phase1" \
    --base-model "$BASE_MODEL" \
    --max-steps "$SPECTRUM_STEPS" \
    --k "$DIVERSITY_K" \
    --num-generations "$NUM_GENERATIONS"

SPECTRUM_FUSED_PATH="$OUTPUT_ROOT/phase1/vibethinker_spectrum_fused"
validate_checkpoint "$SPECTRUM_FUSED_PATH" "Fused Spectrum Model"

# ==============================================================================
# PHASE 2: SIGNAL PHASE (RL Curriculum)
# ==============================================================================
TRAIN_SCRIPT="src/vibethinker/train_complete.py"

# --- Stage 1: Math RL (Standard 4k Context) ---
echo ""
echo ">>> [Phase 2.1] Signal RL - Standard Context (4k)"
STAGE1_OUT="$OUTPUT_ROOT/phase2_stage1_4k"

python "$TRAIN_SCRIPT" \
    --spectrum-path "$SPECTRUM_FUSED_PATH" \
    --output-dir "$STAGE1_OUT" \
    --gpu-type "H100" \
    --max-steps "$SIGNAL_STEPS" \
    --max-seq-length "$MAX_SEQ_LENGTH_STAGE1" \
    --train-data "$DATA_DIR/algebra_train.jsonl" \
    --val-data "$DATA_DIR/algebra_val.jsonl"

STAGE1_MODEL="$STAGE1_OUT/final"
validate_checkpoint "$STAGE1_MODEL" "Stage 1 Model"

# --- Stage 2: Math RL (Long Context 16k) ---
echo ""
echo ">>> [Phase 2.2] Signal RL - Expanding Context to 16k"
STAGE2_OUT="$OUTPUT_ROOT/phase2_stage2_16k"

python "$TRAIN_SCRIPT" \
    --spectrum-path "$STAGE1_MODEL" \
    --output-dir "$STAGE2_OUT" \
    --gpu-type "H100" \
    --max-steps "$SIGNAL_STEPS" \
    --max-seq-length "$MAX_SEQ_LENGTH_STAGE2" \
    --train-data "$DATA_DIR/algebra_train.jsonl" \
    --val-data "$DATA_DIR/algebra_val.jsonl"

STAGE2_MODEL="$STAGE2_OUT/final"
validate_checkpoint "$STAGE2_MODEL" "Stage 2 Model"

# --- Stage 3: Math RL (Long Context 32k) ---
echo ""
echo ">>> [Phase 2.3] Signal RL - Expanding Context to 32k"
STAGE3_OUT="$OUTPUT_ROOT/phase2_stage3_32k"

python "$TRAIN_SCRIPT" \
    --spectrum-path "$STAGE2_MODEL" \
    --output-dir "$STAGE3_OUT" \
    --gpu-type "H100" \
    --max-steps "$SIGNAL_STEPS" \
    --max-seq-length "$MAX_SEQ_LENGTH_STAGE3" \
    --train-data "$DATA_DIR/algebra_train.jsonl" \
    --val-data "$DATA_DIR/algebra_val.jsonl"

STAGE3_MODEL="$STAGE3_OUT/final"
validate_checkpoint "$STAGE3_MODEL" "Stage 3 Model"

# --- Stage 4: Code Generalization ---
echo ""
echo ">>> [Phase 2.4] Signal RL - Code Domain"
FINAL_OUT="$OUTPUT_ROOT/phase2_stage4_code"

# Code RL with APPS dataset
CODE_TRAIN_DATA="$DATA_DIR/apps_train.jsonl"
CODE_VAL_DATA="$DATA_DIR/apps_val.jsonl"

if [ ! -f "$CODE_TRAIN_DATA" ]; then
    echo "Error: $CODE_TRAIN_DATA not found. Please run scripts/prepare_apps_data.py first."
    exit 1
fi

python "$TRAIN_SCRIPT" \
    --spectrum-path "$STAGE3_MODEL" \
    --output-dir "$FINAL_OUT" \
    --gpu-type "H100" \
    --max-steps "$SIGNAL_STEPS" \
    --train-data "$CODE_TRAIN_DATA" \
    --val-data "$CODE_VAL_DATA" \
    --reward-type "code" \
    --test-field "test_cases"

validate_checkpoint "$FINAL_OUT/final" "Final Model"

echo ""
echo "====================================================================="
echo "Pipeline Complete."
echo "Final Model: $FINAL_OUT/final"
echo "====================================================================="
echo ""
echo "Configuration Summary:"
echo "  BASE_MODEL: $BASE_MODEL"
echo "  SPECTRUM_STEPS: $SPECTRUM_STEPS"
echo "  SIGNAL_STEPS: $SIGNAL_STEPS"
echo "  MAX_SEQ_LENGTH (Stage 1): $MAX_SEQ_LENGTH_STAGE1"
echo "  MAX_SEQ_LENGTH (Stage 2): $MAX_SEQ_LENGTH_STAGE2"
echo "  MAX_SEQ_LENGTH (Stage 3): $MAX_SEQ_LENGTH_STAGE3"
echo "====================================================================="