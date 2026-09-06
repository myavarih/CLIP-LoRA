#!/bin/bash
# ==============================================================================
# Multi-Dataset Benchmark Execution Suite (4 Paradigms across Seeds 11, 12, 13)
# 
# Usage:
#   ./run_experiments_suite.sh [1|2|3|4|all] [DATASET] [SEEDS] [N_ITERS] [ROOT_PATH]
#
# Datasets:
#   - walnut         : Walnut Color & Quality Grading (6 classes)
#   - piarom_shape   : Piarom Date Morphological & Defect Grading (5 classes)
#   - pistachio_afat : Pistachio Pest & Biological Defect Diagnosis (4 classes)
#   - stanford_cars  : Stanford Cars Fine-Grained Model Classification (196 classes)
#   - all            : Run across all benchmark datasets
#
# Paradigms:
#   1 -> Infix CoOp-LoRA (M=4: "photo of a [CLASS] <item>") + ResCls + Ordinal (λ=1.0)
#   2 -> Plain LoRA Baseline (Fixed Natural Sentence, Dual LoRA)
#   3 -> CoOp-CSC Dual LoRA (Class-Specific Context KxM + Dual LoRA)
#   4 -> Plain LoRA + Residual Class Token Learning (Static Template, Dual LoRA)
#   all -> Run all 4 paradigms sequentially
#
# Examples:
#   ./run_experiments_suite.sh all walnut "11 12 13" 250 "/kaggle/input/datasets/emmwhy/few-shot-data/FewShotData"
#   ./run_experiments_suite.sh 1 piarom_shape "11 12 13" 250
#   ./run_experiments_suite.sh all all "11 12 13" 250
# ==============================================================================
set -e

RUN_ID="${1:-all}"
DATASET_ARG="${2:-all}"
SEEDS="${3:-11 12 13}"
N_ITERS="${4:-250}"
CUSTOM_ROOT="${5:-}"

echo "========================================================"
echo "🚀 1. Setting up Python Dependencies"
echo "========================================================"
pip install -q ftfy regex tqdm scikit-learn matplotlib seaborn pandas grad-cam scipy || true

# Auto-resolve dataset directory path (Priority: CLI Arg -> Env Var -> Known Kaggle Paths -> Local)
if [ -n "$CUSTOM_ROOT" ] && [ -d "$CUSTOM_ROOT" ]; then
    ROOT_PATH="$CUSTOM_ROOT"
elif [ -n "$DATASET_ROOT" ] && [ -d "$DATASET_ROOT" ]; then
    ROOT_PATH="$DATASET_ROOT"
elif [ -d "/kaggle/input/datasets/emmwhy1/few-shot-data/FewShotData" ]; then
    ROOT_PATH="/kaggle/input/datasets/emmwhy1/few-shot-data/FewShotData"
elif [ -d "/kaggle/input/datasets/emmwhy/few-shot-data/FewShotData" ]; then
    ROOT_PATH="/kaggle/input/datasets/emmwhy/few-shot-data/FewShotData"
elif [ -d "/kaggle/input/fewshotdata/FewShotData" ]; then
    ROOT_PATH="/kaggle/input/fewshotdata/FewShotData"
elif [ -d "/kaggle/input/few-shot-data/FewShotData" ]; then
    ROOT_PATH="/kaggle/input/few-shot-data/FewShotData"
elif [ -d "FewShotData" ]; then
    ROOT_PATH="FewShotData"
elif [ -d "../FewShotData" ]; then
    ROOT_PATH="../FewShotData"
else
    ROOT_PATH="${CUSTOM_ROOT:-}"
fi

EXTRA_ARGS=""
if [ -n "$ROOT_PATH" ]; then
    EXTRA_ARGS="--root_path $ROOT_PATH"
fi

get_ctx_init() {
    local d="$1"
    case "$d" in
        walnut)
            echo "photo_of_a_{}_walnut"
            ;;
        piarom_shape)
            echo "photo_of_a_{}_Piarom_date"
            ;;
        pistachio_afat)
            echo "photo_of_a_{}_pistachio"
            ;;
        stanford_cars|cars)
            echo "photo_of_a_{}_car"
            ;;
        *)
            echo "photo_of_a_{}_item"
            ;;
    esac
}

run_single_paradigm_on_dataset() {
    local paradigm="$1"
    local ds="$2"
    local ctx_template=$(get_ctx_init "$ds")

    case "$paradigm" in
        1)
            echo "========================================================"
            echo "▶️ [RUN 1] Infix CoOp-LoRA (M=4: '${ctx_template}') + ResCls + Ordinal on ${ds} (Seeds: ${SEEDS})"
            echo "========================================================"
            python3 my_impl/run_experiments.py \
                --dataset "$ds" \
                --shots 1 2 4 8 16 32 \
                --seed $SEEDS \
                --method coop_lora \
                --ctx_init "$ctx_template" \
                --learn_class_tokens \
                --use_ordinal \
                --lambda_ord 1.0 \
                --encoder vision \
                --lr 2e-4 \
                --lr_class 1e-3 \
                --n_iters "$N_ITERS" \
                --batch_size 32 \
                --output_dir "experiments_output/run1_infix_coop_ordinal_${ds}" \
                --checkpoints_dir "checkpoints/run1_infix_coop_ordinal_${ds}" \
                $EXTRA_ARGS
            ;;
        2)
            echo "========================================================"
            echo "▶️ [RUN 2] Plain LoRA Baseline (Dual LoRA) on ${ds} (Seeds: ${SEEDS})"
            echo "========================================================"
            python3 my_impl/run_experiments.py \
                --dataset "$ds" \
                --shots 1 2 4 8 16 32 \
                --seed $SEEDS \
                --method lora \
                --encoder both \
                --r 2 \
                --alpha 1.0 \
                --lr 2e-4 \
                --n_iters "$N_ITERS" \
                --batch_size 32 \
                --output_dir "experiments_output/run2_plain_lora_${ds}" \
                --checkpoints_dir "checkpoints/run2_plain_lora_${ds}" \
                $EXTRA_ARGS
            ;;
        3)
            echo "========================================================"
            echo "▶️ [RUN 3] CoOp-CSC Dual LoRA (KxM Context + Dual LoRA) on ${ds} (Seeds: ${SEEDS})"
            echo "========================================================"
            python3 my_impl/run_experiments.py \
                --dataset "$ds" \
                --shots 1 2 4 8 16 32 \
                --seed $SEEDS \
                --method csc_lora \
                --csc \
                --n_ctx 4 \
                --encoder both \
                --r 2 \
                --alpha 1.0 \
                --lr 2e-4 \
                --n_iters "$N_ITERS" \
                --batch_size 32 \
                --output_dir "experiments_output/run3_coop_csc_dual_lora_${ds}" \
                --checkpoints_dir "checkpoints/run3_coop_csc_dual_lora_${ds}" \
                $EXTRA_ARGS
            ;;
        4)
            echo "========================================================"
            echo "▶️ [RUN 4] Plain LoRA + Residual Class Tokens on ${ds} (Seeds: ${SEEDS})"
            echo "========================================================"
            python3 my_impl/run_experiments.py \
                --dataset "$ds" \
                --shots 1 2 4 8 16 32 \
                --seed $SEEDS \
                --method res_cls_lora \
                --encoder both \
                --learn_class_tokens \
                --lr 2e-4 \
                --lr_class 1e-3 \
                --n_iters "$N_ITERS" \
                --batch_size 32 \
                --output_dir "experiments_output/run4_plain_lora_rescls_${ds}" \
                --checkpoints_dir "checkpoints/run4_plain_lora_rescls_${ds}" \
                $EXTRA_ARGS
            ;;
    esac
}

execute_suite() {
    local target_ds="$1"
    if [ "$target_ds" = "all" ]; then
        DATASETS=("walnut" "piarom_shape" "pistachio_afat" "stanford_cars")
    else
        DATASETS=("$target_ds")
    fi

    for ds in "${DATASETS[@]}"; do
        echo "========================================================"
        echo "🌾 BENCHMARKING DATASET: ${ds}"
        echo "========================================================"
        if [ "$RUN_ID" = "all" ]; then
            run_single_paradigm_on_dataset 1 "$ds"
            run_single_paradigm_on_dataset 2 "$ds"
            run_single_paradigm_on_dataset 3 "$ds"
            run_single_paradigm_on_dataset 4 "$ds"
        else
            run_single_paradigm_on_dataset "$RUN_ID" "$ds"
        fi
    done
}

execute_suite "$DATASET_ARG"

echo "========================================================"
echo "✅ All Benchmark Experiments Finished Successfully!"
echo "========================================================"
