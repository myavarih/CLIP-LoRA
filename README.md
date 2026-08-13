# CLIP-LoRA: Efficient Few-Shot Fine-Tuning with Low-Rank Adaptation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![LoRA](https://img.shields.io/badge/PEFT-LoRA-orange.svg)](https://github.com/microsoft/LoRA)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides an optimized PyTorch implementation of **CLIP-LoRA** for parameter-efficient few-shot adaptation of OpenAI's CLIP (`ViT-B/16`). It includes automated multi-shot training pipelines, hyperparameter ablation suites, visualization tools, and catastrophic forgetting evaluation.

🔗 **Repository:** [https://github.com/myavarih/CLIP-LoRA](https://github.com/myavarih/CLIP-LoRA)

---

## 📌 Overview

Parameter-Efficient Fine-Tuning (PEFT) using **Low-Rank Adaptation (LoRA)** injects trainable rank-decomposition matrices into CLIP's attention projections ($q, k, v$) while keeping original backbone parameters frozen.

### Key Highlights
- **Parameter Efficiency:** Trains only $\sim 0.15\%$ of model parameters ($r=2$).
- **Multi-Shot Progression:** Evaluates 1-shot, 2-shot, 4-shot, 8-shot, 16-shot, and 32-shot fine-grained classification.
- **Visual Diagnostics:** Automated generation of t-SNE, PCA, Confusion Matrices, Prediction Grids, and EigenCAM attention heatmaps.
- **Catastrophic Forgetting Assessment:** Evaluated on ImageNet validation accuracy before and after adaptation.

---

## 📊 Performance Summary

| Configuration | 0-Shot | 1-Shot | 2-Shot | 4-Shot | 8-Shot | 16-Shot | 32-Shot |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot CLIP Baseline** | 22.26% | — | — | — | — | — | — |
| **CLIP-LoRA ($r=2, \alpha=1.0, q,k,v$)** | — | **39.70%** | **45.30%** | **47.70%** | **52.20%** | **57.40%** | **62.82%** |
| **CLIP-LoRA ($\alpha=1.5, q,k,v$)** | — | 38.20% | 44.10% | **50.60%** | 53.10% | 58.00% | — |

### ImageNet Validation (Catastrophic Forgetting)
- **Zero-Shot CLIP:** `66.71%`
- **+ 4-Shot Walnut LoRA:** `66.66%` ($\Delta = -0.05\%$)
- **+ 32-Shot Walnut LoRA:** `64.78%` ($\Delta = -1.93\%$)

---

## 📁 Repository Structure

```
CLIP-LoRA/
├── docs/                        # LaTeX report source and build files
│   ├── main.tex                 # Comprehensive paper & lab report
│   ├── main.pdf                 # Compiled report
│   └── generate_plots.py        # Figure generation script
├── my_impl/                     # Core implementation & experiment runners
│   ├── clip/                    # Fast CLIP backbone interface
│   ├── datasets/                # Dataset loaders (Walnut, Caltech101, etc.)
│   ├── loralib/                 # Custom LoRA layer implementations
│   ├── lora.py                  # CLIP-LoRA model wrapper
│   ├── main.py                  # Single-run training entrypoint
│   ├── run_experiments.py       # Automated multi-shot runner with resumption
│   ├── vis_utils.py             # Diagnostic plotting & EigenCAM tools
│   └── utils.py                 # Seeds, evaluation, and helper functions
├── original_implementation/     # Baseline original reference implementation
├── .gitignore                   # Workspace ignore configurations
├── imagenet_32shot_eval.json    # ImageNet evaluation output artifact
└── README.md                    # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/myavarih/CLIP-LoRA.git
   cd CLIP-LoRA
   ```

2. **Install dependencies:**
   ```bash
   pip install -r my_impl/requirements.txt
   ```

---

## 🚀 Running Experiments

### 1. Automated Multi-Shot Training Pipeline
Run the full multi-shot training pipeline across 1, 2, 4, 8, 16, and 32 shots:

```bash
python my_impl/run_experiments.py \
    --dataset walnut \
    --shots 1 2 4 8 16 32 \
    --r 2 \
    --alpha 1.0 \
    --params q k v \
    --encoder both \
    --batch_size 32 \
    --lr 2e-4 \
    --n_iters 500
```

### 2. Single Run Entrypoint
To train a single configuration manually:

```bash
python my_impl/main.py \
    --dataset walnut \
    --shots 4 \
    --r 2 \
    --alpha 1.5 \
    --position all \
    --encoder both \
    --params q k v
```

---

## 📄 Documentation & Report

The detailed lab report and empirical findings are available in `docs/`:
- **LaTeX Source:** [`docs/main.tex`](docs/main.tex)
- **PDF Report:** [`docs/main.pdf`](docs/main.pdf)

To recompile the PDF document locally:
```bash
cd docs
pdflatex -interaction=nonstopmode main.tex
```

---

## 👤 Author

**Mohammad Yavari**  
GitHub: [@myavarih](https://github.com/myavarih)  
Repository: [CLIP-LoRA](https://github.com/myavarih/CLIP-LoRA)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
