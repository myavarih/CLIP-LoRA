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

### Benchmark Suite (4 Key Paradigms & 4 Datasets)

All runs leverage semantic English class name mappings to provide rich textual priors to CLIP's text encoder:

| Dataset Identifier | Domain / Crop | Classes | Task Nature & Class Mapping |
|---|---|:---:|---|
| **`walnut`** | Walnut | **6** | Color & Maturity: `dark meaty`, `high-quality brown`, `premium brown`, `standard white`, `premium white`, `luxury extra light` |
| **`piarom_shape`** | Piarom Date | **5** | Morphological & Defect: `grade 1 premium`, `grade 2 standard`, `grade 3`, `small stunted`, `crushed bruised` |
| **`pistachio_afat`** | Pistachio | **4** | Pest & Biological Defect: `healthy sound`, `empty puffy shell`, `insect pest damaged`, `black stained defect` |
| **`stanford_cars`** | Cars | **196** | Fine-Grained Model Recognition: 196 makes/models (e.g., `2012 Tesla Model S`, `1997 BMW 3 Series`) |

---

### Paradigm Architectural Breakdown

| Run | Method / Paradigm | Context Parameterization | Class Token Adaptation | Encoders with LoRA | Ordinal Cost Loss |
|:---:|---|---|---|:---:|:---:|
| **1** | **Infix CoOp-LoRA** | $M=4$ (`photo of a [CLASS] <item>`) | Residual ($\Delta w_c$, `1e-3`) | Vision only | ✅ $\lambda=1.0$ |
| **2** | **Plain LoRA (Baseline)** | Fixed Natural Template | ❌ None (Fixed) | Vision + Text | ❌ None |
| **3** | **CoOp-CSC Dual LoRA** | $K \times M$ Class-Specific Context | ❌ None (Base token) | Vision + Text | ❌ None |
| **4** | **Plain LoRA + ResCls** | Fixed Natural Template | Residual ($\Delta w_c$, `1e-3`) | Vision + Text | ❌ None |

#### Execute with Unified Multi-Seed Runner (Seeds 11, 12, 13):
```bash
# Run all 4 paradigms on all benchmark datasets across seeds 11, 12, 13:
./run_experiments_suite.sh all all "11 12 13" 250

# Run all 4 paradigms on a single dataset:
./run_experiments_suite.sh all walnut "11 12 13" 250
./run_experiments_suite.sh all piarom_shape "11 12 13" 250
./run_experiments_suite.sh all pistachio_afat "11 12 13" 250
./run_experiments_suite.sh all stanford_cars "11 12 13" 250

# Or execute an individual paradigm on a specific dataset:
./run_experiments_suite.sh 1 piarom_shape "11 12 13" 250   # Infix CoOp-LoRA + ResCls + Ordinal
./run_experiments_suite.sh 2 stanford_cars "11 12 13" 250  # Plain LoRA Baseline
./run_experiments_suite.sh 3 pistachio_afat "11 12 13" 250 # CoOp-CSC Dual LoRA
./run_experiments_suite.sh 4 walnut "11 12 13" 250         # Plain LoRA + Residual Class Tokens
```

---

### Individual Command Execution

#### 1. Infix CoOp-LoRA ($M=4$) + Residual Class Tokens + Ordinal Loss
```bash
python my_impl/run_experiments.py \
    --dataset walnut \
    --shots 1 2 4 8 16 32 \
    --method coop_lora \
    --ctx_init "photo_of_a_{}_walnut" \
    --learn_class_tokens \
    --use_ordinal \
    --lambda_ord 1.0 \
    --encoder vision \
    --n_iters 250 \
    --output_dir experiments_output/run1_infix_coop_ordinal_rescls \
    --checkpoints_dir checkpoints/run1_infix_coop_ordinal_rescls
```

#### 2. Plain LoRA Baseline
```bash
python my_impl/run_experiments.py \
    --dataset walnut \
    --shots 1 2 4 8 16 32 \
    --method lora \
    --encoder both \
    --n_iters 250 \
    --output_dir experiments_output/run2_plain_lora \
    --checkpoints_dir checkpoints/run2_plain_lora
```

#### 3. CoOp-CSC Dual LoRA
```bash
python my_impl/run_experiments.py \
    --dataset walnut \
    --shots 1 2 4 8 16 32 \
    --method csc_lora \
    --csc \
    --encoder both \
    --n_iters 250 \
    --output_dir experiments_output/run3_coop_csc_dual_lora \
    --checkpoints_dir checkpoints/run3_coop_csc_dual_lora
```

#### 4. Plain LoRA with Residual Class Token Learning
```bash
python my_impl/run_experiments.py \
    --dataset walnut \
    --method res_cls_lora \
    --encoder both \
    --learn_class_tokens \
    --n_iters 250 \
    --output_dir experiments_output/run4_plain_lora_rescls \
    --checkpoints_dir checkpoints/run4_plain_lora_rescls
```

---

### Single Configuration Manual Entrypoint
To train a single configuration manually (e.g., 4-shot with custom $\alpha$):
```bash
python my_impl/main.py \
    --dataset walnut \
    --shots 4 \
    --r 2 \
    --alpha 1.5 \
    --position all \
    --encoder both \
    --params q k v \
    --n_iters 250
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
