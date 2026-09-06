#!/bin/bash
# ==============================================================================
# Kaggle Execution Script for Residual-Template Dual-LoRA (RT-LoRA)
# ==============================================================================
set -e

echo "========================================================"
echo "🚀 1. Setting up Python Dependencies in Kaggle Environment"
echo "========================================================"
pip install -q ftfy regex tqdm scikit-learn matplotlib seaborn pandas grad-cam

# Ensure dataset path is resolved
if [ -d "/kaggle/input/fewshotdata/FewShotData" ]; then
    ROOT_PATH="/kaggle/input/fewshotdata/FewShotData"
elif [ -d "/kaggle/input/few-shot-data/FewShotData" ]; then
    ROOT_PATH="/kaggle/input/few-shot-data/FewShotData"
elif [ -d "FewShotData" ]; then
    ROOT_PATH="FewShotData"
else
    ROOT_PATH=""
fi

echo "📂 Using Dataset Root Path: ${ROOT_PATH:-'Auto-Detect'}"

echo "========================================================"
echo "🚀 2. Running RT-LoRA Multi-Shot Benchmark (Shots: 1, 2, 4, 8, 16, 32)"
echo "========================================================"

EXTRA_ARGS=""
if [ -n "$ROOT_PATH" ]; then
    EXTRA_ARGS="--root_path $ROOT_PATH"
fi

python3 my_impl/run_experiments.py \
    --dataset walnut \
    --shots 1 2 4 8 16 32 \
    --method rt_lora \
    --encoder both \
    --learn_class_tokens \
    --learn_template_tokens \
    --lr 2e-4 \
    --lr_class 1e-3 \
    --lr_template 1e-4 \
    --use_ordinal \
    --lambda_ord 1.0 \
    --n_iters 250 \
    --batch_size 32 \
    --seed 1 \
    --output_dir experiments_output/rt_lora_run \
    --checkpoints_dir checkpoints/rt_lora_run \
    $EXTRA_ARGS

echo "========================================================"
echo "✅ RT-LoRA Multi-Shot Run Finished Successfully!"
echo "📊 Results and visual plots are saved in: experiments_output/rt_lora_run"
echo "========================================================"
