#!/bin/bash
# ==============================================================================
# Execution Script for UniFGVC MLLM Ensemble + Feature-Space Residual + Vision LoRA + kgCoOp
# (Variant 2 with integrated Variant 1 Zero-Shot baseline logging)
#
# Usage:
#   ./run_mllm_variant2.sh [walnut|piarom_shape] [seed] [gpu_id]
# Example:
#   ./run_mllm_variant2.sh walnut 1 0
#   ./run_mllm_variant2.sh piarom_shape 1 0
# ==============================================================================
set -e

DATASET="${1:-walnut}"
SEEDS="${2:-21 22 23}"
GPU_ID="${3:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
# Unset invalid proxy if present
unset ALL_PROXY

echo "========================================================"
echo "🚀 UniFGVC MLLM Feature Residual + Vision LoRA + kgCoOp"
echo "🎯 Dataset: ${DATASET} | Seeds: ${SEEDS} | GPU: ${GPU_ID}"
echo "========================================================"

# Auto-detect dataset root path
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

EXTRA_ARGS=""
if [ -n "$ROOT_PATH" ]; then
    EXTRA_ARGS="--root_path $ROOT_PATH"
fi

OUTPUT_DIR="experiments_output/mllm_variant2_${DATASET}"
CHECKPOINTS_DIR="checkpoints/mllm_variant2_${DATASET}"

python3 my_impl/run_experiments.py \
    --dataset "${DATASET}" \
    --shots 1 2 4 8 16 32 \
    --method mllm_feat_lora \
    --encoder vision \
    --use_mllm_prompts \
    --use_kgcoop \
    --lambda_kg 2.0 \
    --lr 2e-4 \
    --lr_prompt 1e-4 \
    --n_iters 250 \
    --batch_size 32 \
    --seed ${SEEDS} \
    --output_dir "${OUTPUT_DIR}" \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    $EXTRA_ARGS

echo "========================================================"
echo "✅ Experiment completed successfully!"
echo "📊 Results and plots saved in: ${OUTPUT_DIR}"
echo "========================================================"
