#!/bin/bash
# ==============================================================================
# Execution Script for Staged CoOp-CSC + LoRA Fine-Tuning
# 
# Mechanism:
#   Stage 1 (0 to CSC_ITERS iters/shot): Class-Specific Context (CSC) prompts
#           are actively optimized alongside LoRA modules.
#   Stage 2 (CSC_ITERS to N_ITERS iters/shot): CSC prompts are frozen, and
#           only the LoRA adapter layers continue training to prevent prompt overfitting.
#
# Usage:
#   ./run_csc_staged.sh [ENCODER_TARGET] [CSC_ITERS] [N_ITERS]
#
# Examples:
#   ./run_csc_staged.sh vision 100 250   # CSC trains 100 iters/shot, then frozen; Vision LoRA trains 250 iters/shot (Default)
#   ./run_csc_staged.sh both 100 250     # CSC trains 100 iters/shot, then frozen; Dual LoRA trains 250 iters/shot
#   ./run_csc_staged.sh text 50 250      # CSC trains 50 iters/shot, then frozen; Text LoRA trains 250 iters/shot
# ==============================================================================
set -e

ENCODER_TARGET="${1:-vision}"
CSC_ITERS="${2:-100}"
N_ITERS="${3:-250}"

echo "========================================================"
echo "🚀 1. Setting up Python Dependencies"
echo "========================================================"
pip install -q ftfy regex tqdm scikit-learn matplotlib seaborn pandas grad-cam || true

# Ensure dataset path is resolved
if [ -d "/kaggle/input/fewshotdata/FewShotData" ]; then
    ROOT_PATH="/kaggle/input/fewshotdata/FewShotData"
elif [ -d "/kaggle/input/few-shot-data/FewShotData" ]; then
    ROOT_PATH="/kaggle/input/few-shot-data/FewShotData"
elif [ -d "FewShotData" ]; then
    ROOT_PATH="FewShotData"
elif [ -d "../FewShotData" ]; then
    ROOT_PATH="../FewShotData"
else
    ROOT_PATH=""
fi

echo "📂 Using Dataset Root Path: ${ROOT_PATH:-'Auto-Detect'}"
echo "🎯 LoRA Target Encoder: ${ENCODER_TARGET}"
echo "⏱️ CSC Active Iterations: ${CSC_ITERS} iters/shot (Prompt will freeze after this)"
echo "🏁 Total Training Budget: ${N_ITERS} iters/shot (LoRA continues training)"

echo "========================================================"
echo "🚀 2. Running Staged CoOp-CSC Benchmark (Shots: 1, 2, 4, 8, 16, 32)"
echo "========================================================"

EXTRA_ARGS=""
if [ -n "$ROOT_PATH" ]; then
    EXTRA_ARGS="--root_path $ROOT_PATH"
fi

OUTPUT_DIR="experiments_output/staged_csc_${CSC_ITERS}iters_${ENCODER_TARGET}_lora_run"
CHECKPOINTS_DIR="checkpoints/staged_csc_${CSC_ITERS}iters_${ENCODER_TARGET}_lora_run"

if [ "$ENCODER_TARGET" = "none" ]; then
    METHOD="coop_only"
else
    METHOD="coop_lora"
fi

python3 my_impl/run_experiments.py \
    --dataset walnut \
    --shots 1 2 4 8 16 32 \
    --method "$METHOD" \
    --encoder "$ENCODER_TARGET" \
    --csc \
    --n_ctx 4 \
    --lr 2e-4 \
    --n_iters "$N_ITERS" \
    --csc_iters "$CSC_ITERS" \
    --batch_size 32 \
    --seed 1 \
    --output_dir "$OUTPUT_DIR" \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    $EXTRA_ARGS

echo "========================================================"
echo "✅ Staged CoOp-CSC Benchmark Finished Successfully!"
echo "📊 Results and visual plots saved to: ${OUTPUT_DIR}"
echo "========================================================"
