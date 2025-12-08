#!/bin/bash
# VibeThinker Training Pipeline - Optimized for T4 GPU (16GB) in ~5 hours
# This is a reduced-scale configuration for experimentation

set -e

# ==============================================================================
# T4-OPTIMIZED CONFIGURATION
# ==============================================================================

# Base model (same as default, already optimized for 4-bit)
export BASE_MODEL="unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit"

# Output directories
export DATA_DIR="data"
export OUTPUT_ROOT="outputs/vibethinker_t4_5h"

# ----------------------
# REDUCED TRAINING STEPS
# ----------------------
# T4 throughput: ~500-1000 tokens/sec (vs H100: ~5000+ tokens/sec)
# We reduce steps by ~5-10x to fit in 5 hours

export SPECTRUM_STEPS=200          # Default: 1000 (reduced 5x)
export SIGNAL_STEPS=100            # Default: 500 (reduced 5x)

# ----------------------
# REDUCED DIVERSITY PROBING
# ----------------------
export DIVERSITY_K=4               # Default: 8 (halved for speed)
export NUM_GENERATIONS=8           # Default: 16 (halved for speed)

# ----------------------
# REDUCED CONTEXT LENGTH (CRITICAL FOR T4 VRAM)
# ----------------------
# T4 can't handle 32K context - reduce significantly
export MAX_SEQ_LENGTH_STAGE1=2048  # Default: 4096
export MAX_SEQ_LENGTH_STAGE2=4096  # Default: 16384
export MAX_SEQ_LENGTH_STAGE3=4096  # Default: 32768 (T4 can't do 32K!)

# ----------------------
# DATA PREPARATION (MINIMAL)
# ----------------------
export DATA_PREP_MAX_PROBLEMS=100  # Default: 500 (reduced for speed)
export DATA_PREP_N_SOLUTIONS=1     # Default: 3 (reduced for speed)
export DATA_PREP_TEACHER_MODEL="meta-llama/Llama-3.2-1B-Instruct"  # Smaller teacher

# Skip decontamination for speed (not recommended for final training)
export DECONTAM_ENABLED=false

echo "============================================================"
echo "VibeThinker T4 Quick Training Pipeline"
echo "============================================================"
echo "Estimated time: ~4-5 hours on T4 GPU"
echo "VRAM usage: ~12-14 GB peak"
echo ""
echo "Configuration:"
echo "  SPECTRUM_STEPS: $SPECTRUM_STEPS"
echo "  SIGNAL_STEPS: $SIGNAL_STEPS"
echo "  MAX_SEQ_LENGTH: $MAX_SEQ_LENGTH_STAGE1 -> $MAX_SEQ_LENGTH_STAGE2 -> $MAX_SEQ_LENGTH_STAGE3"
echo "  DIVERSITY_K: $DIVERSITY_K"
echo "  NUM_GENERATIONS: $NUM_GENERATIONS"
echo "============================================================"
echo ""

# Run the main pipeline
./pipeline.sh

echo ""
echo "============================================================"
echo "Training Complete!"
echo "Final model: $OUTPUT_ROOT/phase2_stage4_code/final"
echo "============================================================"
