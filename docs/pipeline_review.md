# Pipeline Review: VibeThinker Workflow Analysis

## Executive Summary

The [pipeline.sh](file:///media/limcheekin/My Passport/ws/py/VibeThinker/pipeline.sh) script implements a two-phase training workflow (Spectrum + Signal) for VibeThinker. While the **core architecture is sound**, there are **critical gaps** that prevent it from being a complete, production-ready workflow.

> [!CAUTION]
> **Critical Issues Found**: Missing data files, hardcoded path modifications using `sed`, no data preparation step, and incomplete error handling.

---

## ✅ What Works Well

### 1. **Correct Phase Architecture**
The pipeline correctly implements the two-phase VibeThinker methodology:
- **Phase 1 (Spectrum)**: Multi-domain specialist training → diversity probing → fusion
- **Phase 2 (Signal)**: Three-stage RL curriculum (4k math → 16k math → code)

### 2. **Proper Use of Orchestrator**
Lines 19-25 correctly delegate Phase 1 to the Python orchestrator ([train_spectrum_phase.py](file:///media/limcheekin/My Passport/ws/py/VibeThinker/scripts/train_spectrum_phase.py)), which handles:
- Training specialists via [train_sft_specialist.py](file:///media/limcheekin/My Passport/ws/py/VibeThinker/scripts/train_sft_specialist.py)
- Diversity probing for checkpoint selection
- Model fusion

### 3. **Staged Context Expansion**
Lines 42-76 implement the curriculum learning strategy:
- Stage 1: Standard 4k context
- Stage 2: Extended 16k context  
- Stage 3: Code domain generalization

### 4. **Error Detection**
Lines 29-32 verify the fused model exists before proceeding to Phase 2.

---

## ❌ Critical Issues

### Issue 1: **Missing Domain Data Files**

> [!WARNING]
> **Blocker**: Only `algebra` data exists; missing `geometry`, `calculus`, `statistics`

**Current State:**
```bash
data/
├── algebra_train.jsonl  ✓
└── algebra_val.jsonl    ✓
# Missing:
# ❌ geometry.jsonl
# ❌ calculus.jsonl  
# ❌ statistics.jsonl
# ❌ livecodebench.jsonl
```

**Impact:**
- Phase 1 will only train 1 specialist instead of 4
- Model fusion will be severely underfitted
- Stage 3 (code generalization) will fail at line 85

**Evidence from orchestrator:**
[train_spectrum_phase.py:L20](file:///media/limcheekin/My Passport/ws/py/VibeThinker/scripts/train_spectrum_phase.py#L20):
```python
DOMAINS = ["algebra", "geometry", "calculus", "statistics"]
```

[train_spectrum_phase.py:L166-169](file:///media/limcheekin/My Passport/ws/py/VibeThinker/scripts/train_spectrum_phase.py#L166-L169):
```python
for domain in DOMAINS:
    data_path = f"{args.data_dir}/{domain}.jsonl"
    if not os.path.exists(data_path):
        print(f"WARNING: Data file not found: {data_path}")
```

---

### Issue 2: **Dangerous `sed` Modifications**

> [!CAUTION]
> **Anti-Pattern**: Lines 59-63 and 84-85 modify source code in-place using `sed`

**Problems:**

1. **Fragile Pattern Matching**
   ```bash
   sed -i 's/max_seq_length=4096/max_seq_length=16384/g' "$TRAIN_SCRIPT"
   ```
   - Breaks if [train_complete.py](file:///media/limcheekin/My Passport/ws/py/VibeThinker/src/vibethinker/train_complete.py) formatting changes
   - No verification that substitution succeeded
   
2. **Race Conditions**
   - Backup/restore pattern is not atomic
   - Concurrent runs would corrupt each other

3. **Dataset Path Hardcoding**
   ```bash
   sed -i 's|data/algebra_train.jsonl|data/livecodebench.jsonl|g' "$TRAIN_SCRIPT"
   ```
   This is **extremely brittle** because [train_complete.py:L292](file:///media/limcheekin/My Passport/ws/py/VibeThinker/src/vibethinker/train_complete.py#L292) hardcodes:
   ```python
   train_dataset = load_dataset("json", data_files="data/algebra_train.jsonl", split="train")
   ```

**Better Approach:**
`train_complete.py` should accept `--max-seq-length` and `--data-file` arguments instead of sed patching.

---

### Issue 3: **No Data Preparation Step**

> [!IMPORTANT]
> **Gap**: Pipeline assumes pre-existing data but provides no preparation step

**Missing Link:**
The repository includes [prepare_spectrum_data.py](file:///media/limcheekin/My Passport/ws/py/VibeThinker/scripts/prepare_spectrum_data.py) (1023 lines) which:
- Downloads and processes MATH dataset
- Performs teacher distillation for multi-solution generation  
- Applies semantic deduplication
- Splits into domain-specific files

**Pipeline should include:**
```bash
# PHASE 0: DATA PREPARATION
echo ">>> [Phase 0] Preparing Spectrum Data"
python scripts/prepare_spectrum_data.py \
    --hf-id "lighteval/MATH" \
    --output-dir "$DATA_DIR" \
    --n-solutions 3 \
    --verify
```

---

### Issue 4: **Incomplete Error Handling**

**Current validation:**
- ✓ Checks fused model exists (line 29)
- ❌ No check for Stage 1/2 model outputs
- ❌ No validation that `sed` substitutions worked
- ❌ No recovery strategy if training fails mid-stage

**Missing:**
```bash
if [ ! -d "$STAGE1_MODEL" ]; then
    echo "Error: Stage 1 model not found!"
    exit 1
fi
```

---

### Issue 5: **Hardcoded Configuration**

**Inflexible settings:**
```bash
--max-steps 1000  # Phase 1
--max-steps 500   # Phase 2 stages
--k 8
--num-generations 16
```

**Better approach:**
```bash
# Configuration section
SPECTRUM_STEPS=${SPECTRUM_STEPS:-1000}
SIGNAL_STEPS=${SIGNAL_STEPS:-500}
DIVERSITY_K=${DIVERSITY_K:-8}
```

---

## 📊 Workflow Completeness Analysis

| Component | Status | Notes |
|-----------|--------|-------|
| **Phase 1: Specialist Training** | ⚠️ Partial | Only works for `algebra` domain currently |
| **Phase 1: Diversity Probing** | ✅ Complete | Orchestrator calls `DiversityProber` correctly |
| **Phase 1: Model Fusion** | ⚠️ Risky | Works but only with 1 expert (needs 4) |
| **Phase 2: Stage 1 (4k Math)** | ✅ Complete | Solid implementation |
| **Phase 2: Stage 2 (16k Math)** | ⚠️ Fragile | `sed` pattern risky |
| **Phase 2: Stage 3 (Code)** | ❌ Broken | Missing `livecodebench.jsonl` |
| **Data Preparation** | ❌ Missing | No integration of `prepare_spectrum_data.py` |
| **Error Recovery** | ⚠️ Minimal | Basic checks only |
| **Configuration Management** | ❌ Hardcoded | No flexibility |

---

## 🔧 Recommended Fixes

### Priority 1: Data Preparation Integration

Insert before Phase 1:
```bash
# ==============================================================================
# PHASE 0: DATA PREPARATION
# ==============================================================================
echo ">>> [Phase 0] Preparing Domain Data"

if [ ! -f "$DATA_DIR/algebra.jsonl" ] || \
   [ ! -f "$DATA_DIR/geometry.jsonl" ] || \
   [ ! -f "$DATA_DIR/calculus.jsonl" ] || \
   [ ! -f "$DATA_DIR/statistics.jsonl" ]; then
    
    python scripts/prepare_spectrum_data.py \
        --hf-id "lighteval/MATH" \
        --output-dir "$DATA_DIR" \
        --teacher-backend "hf" \
        --teacher-model "meta-llama/Llama-3.2-3B-Instruct" \
        --n-solutions 3 \
        --verify \
        --max-problems 500
    
    if [ $? -ne 0 ]; then
        echo "Error: Data preparation failed!"
        exit 1
    fi
fi
```

### Priority 2: Fix `train_complete.py` to Accept Arguments

Modify [train_complete.py:L282-289](file:///media/limcheekin/My Passport/ws/py/VibeThinker/src/vibethinker/train_complete.py#L282-L289):

```python
parser.add_argument("--max-seq-length", type=int, default=4096)
parser.add_argument("--train-data", type=str, default="data/algebra_train.jsonl")
parser.add_argument("--val-data", type=str, default="data/algebra_val.jsonl")

# Then use these in load_dataset calls
train_dataset = load_dataset("json", data_files=args.train_data, split="train")
```

### Priority 3: Replace `sed` with Arguments

**Stage 2 (16k context):**
```bash
python "$TRAIN_SCRIPT" \
    --spectrum-path "$STAGE1_MODEL" \
    --output-dir "$STAGE2_OUT" \
    --gpu-type "H100" \
    --max-steps 500 \
    --max-seq-length 16384  # Instead of sed
```

**Stage 3 (code domain):**
```bash
python "$TRAIN_SCRIPT" \
    --spectrum-path "$STAGE2_MODEL" \
    --output-dir "$FINAL_OUT" \
    --gpu-type "H100" \
    --max-steps 500 \
    --train-data "data/livecodebench.jsonl" \  # Instead of sed
    --val-data "data/livecodebench_val.jsonl"
```

### Priority 4: Add Comprehensive Validation

```bash
validate_checkpoint() {
    local path=$1
    local name=$2
    if [ ! -d "$path" ]; then
        echo "Error: $name checkpoint not found at $path"
        exit 1
    fi
    # Check for required files
    if [ ! -f "$path/config.json" ]; then
        echo "Error: Invalid checkpoint (missing config.json)"
        exit 1
    fi
    echo "✓ Validated $name: $path"
}

# Use after each stage
validate_checkpoint "$SPECTRUM_FUSED_PATH" "Fused Model"
validate_checkpoint "$STAGE1_MODEL" "Stage 1"
validate_checkpoint "$STAGE2_MODEL" "Stage 2"
```

---

## 📋 Implementation Checklist

### Must-Have (Critical)
- [ ] Add Phase 0 (data preparation) with domain file generation
- [ ] Create missing data files: `geometry.jsonl`, `calculus.jsonl`, `statistics.jsonl`, `livecodebench.jsonl`
- [ ] Add `--max-seq-length`, `--train-data`, `--val-data` arguments to `train_complete.py`
- [ ] Remove all `sed` modifications
- [ ] Add checkpoint validation after each stage

### Should-Have (Important)
- [ ] Make configuration parameters environment-variable-driven
- [ ] Add resume capability (detect existing checkpoints)
- [ ] Log time and cost estimates for each phase
- [ ] Add cleanup option for intermediate checkpoints

### Nice-to-Have (Enhancement)
- [ ] Parallel specialist training (Phase 1)
- [ ] Integration with MLflow/Weights & Biases for tracking
- [ ] Dry-run mode showing execution plan without running
- [ ] Email/Slack notifications on completion or failure

---

## 🎯 Conclusion

> [!IMPORTANT]
> **Status**: Pipeline is **60% complete** but **not production-ready**

**Assessment:**
- ✅ **Architecture**: Correct implementation of VibeThinker methodology  
- ❌ **Execution**: Missing critical data and has dangerous code modifications
- ⚠️ **Robustness**: Minimal error handling and validation

**Immediate Actions:**
1. Implement **Priority 1** (data preparation integration)
2. Generate missing data files using `prepare_spectrum_data.py`
3. Refactor `train_complete.py` to accept CLI arguments (**Priority 2-3**)
4. Test end-to-end with small dataset before full runs

**Estimated Effort:**
- Fix critical issues: **4-6 hours**
- Full production hardening: **2-3 days**

**Risk Level**: **HIGH** - Current pipeline will fail in Phase 1 and 2.3 due to missing data
