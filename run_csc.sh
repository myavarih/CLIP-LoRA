#!/bin/bash
# ==============================================================================
# Execution Script for CoOp Class-Specific Context (CSC) + Configurable LoRA
# Usage:
#   ./run_csc.sh [text|vision|both|none] [csc_iters]
# Example:
#   ./run_csc.sh vision 100   # CSC Prompt Tuning (100 iters/shot) + Vision LoRA (250 iters/shot) [Default]
#   ./run_csc.sh both 100     # CSC Prompt Tuning (100 iters/shot) + Dual LoRA (250 iters/shot)
#   ./run_csc.sh vision 250   # Unfrozen CSC + Vision LoRA (trains CSC for all 250 iters)
# ==============================================================================
set -e

ENCODER_TARGET="${1:-vision}"
CSC_ITERS="${2:-100}"

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
echo "⏱️ CSC Active Iterations per shot: ${CSC_ITERS} (freezes after this, LoRA continues to 250)"

echo "========================================================"
echo "🚀 2. Running CoOp-CSC Multi-Shot Benchmark (Shots: 1, 2, 4, 8, 16, 32)"
echo "========================================================"

EXTRA_ARGS=""
if [ -n "$ROOT_PATH" ]; then
    EXTRA_ARGS="--root_path $ROOT_PATH"
fi

OUTPUT_DIR="experiments_output/csc_${ENCODER_TARGET}_lora_run"
CHECKPOINTS_DIR="checkpoints/csc_${ENCODER_TARGET}_lora_run"

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
    --n_iters 250 \
    --csc_iters "$CSC_ITERS" \
    --batch_size 32 \
    --seed 1 \
    --output_dir "$OUTPUT_DIR" \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    $EXTRA_ARGS

echo "========================================================"
echo "✅ CoOp-CSC (${ENCODER_TARGET}, csc_iters=${CSC_ITERS}) Multi-Shot Run Finished Successfully!"
echo "📊 Results and visual plots are saved in: ${OUTPUT_DIR}"
echo "========================================================"
