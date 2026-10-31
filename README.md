# 🌊 Flood Segmentation Research Framework (Paper-Ready Suite)

A research-grade training and benchmarking framework for flood segmentation with **strict determinism**, **comprehensive evaluation metrics**, and **cross-dataset domain transfer** using PyTorch.

---

## 📋 Table of Contents
- [Datasets](#-datasets)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Paper Evaluation Metrics](#-paper-evaluation-metrics)
- [Cross-Dataset Generalization (Item 5)](#-cross-dataset-generalization)
- [Baseline Models & Pretrained Options (Item 1)](#-baseline-models--pretrained-options)
- [Multi-Seed Experiments](#-multi-seed-experiments)
- [Project Structure](#-project-structure)

---

## 🗂️ Datasets

| Dataset | Type | Classes | Loss | Structure / Notes |
| :--- | :--- | :---: | :--- | :--- |
| **FloodKaggle** | Binary | 1 | `bce` | Images (`images/`) & Masks (`masks/`) |
| **FloodScene** | Multi-class / Binary | 4 or 1 | `ce_dice` / `bce` | Classes: background (0), sky (85), building (170), flood (255) |
| **FloodVN** | Binary | 1 | `bce` | Supports both image masks and LabelMe polygon JSON (`labels/*.json`) |
| **FloodNet** | Multi-class / Binary | 10 or 1 | `ce_dice` / `bce` | High-resolution UAV flood disaster imagery |

---

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/tuannh1905/MambaFloodLite.git
cd MambaFloodLite

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Training Proposed Model (`fsenet`)
```bash
# Single-seed training
python main.py \
  --dataset floodkaggle \
  --model fsenet \
  --loss bce \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.001 \
  --seed 42
```

### 2. Multi-Class Training on FloodScene (4 classes)
```bash
python main.py \
  --dataset floodscene \
  --model fsenet \
  --num_classes 4 \
  --loss ce_dice \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.001
```

---

## 📊 Paper Evaluation Metrics (Item 2)

The framework now computes a complete suite of standard remote sensing and computer vision metrics:

1. **mIoU (Mean Intersection over Union)**
2. **Dice Score / F1-Score**
3. **Pixel Accuracy**
4. **Precision**: $\frac{TP}{TP + FP}$ (Quantifies false flood alarms)
5. **Recall**: $\frac{TP}{TP + FN}$ (Quantifies missed flooded areas - critical in disaster rescue)
6. **FWIoU (Frequency Weighted IoU)**: Weights IoU by ground-truth class pixel abundance
7. **Boundary IoU**: Measures contour/edge alignment between prediction and ground-truth boundary (crucial for validating the Auxiliary Boundary Head)
8. **Mathematical FLOPs / GMACs**: Accurately accounts for Depthwise Separable Convolution groups (`in_channels // groups`)

---

## 🌐 Cross-Dataset Generalization (Item 5)

Evaluate zero-shot transferability across flood datasets:

```bash
# Standalone evaluation across domains:
python cross_dataset_eval.py \
  --model fsenet \
  --weights outputs/fsenet_bce_floodkaggle_s42.pth \
  --source_dataset floodkaggle \
  --target_datasets floodkaggle floodscene floodvn
```

Or via `main.py`:
```bash
python main.py \
  --mode cross_eval \
  --model fsenet \
  --weights outputs/fsenet_bce_floodkaggle_s42.pth \
  --source_dataset floodkaggle \
  --target_datasets floodkaggle floodscene
```

Output includes a consolidated summary table and formatted **LaTeX code** ready for pasting into the paper!

---

## 🔬 Baseline Models & Pretrained Options (Item 1)

All major real-time and heavy segmentation baselines are supported:

| Category | Model Name | Description |
| :--- | :--- | :--- |
| **Proposed** | `fsenet` | Ultra-lightweight edge flood segmentation network (~0.14M deployable params) |
| **Lightweight Baselines** | `fast-scnn`, `espnetv2`, `stdc1`, `dabnet`, `edanet`, `enet`, `ulite` | Real-time edge architectures |
| **Classic Baselines** | `unet`, `segnet` | Canonical segmentation backbones |
| **Heavy Baselines** | `deeplabv3`, `segformer` | Deep baselines (supports `--pretrained` for ImageNet backbone initialization) |

To train a baseline with pretrained weights:
```bash
python main.py --model deeplabv3 --dataset floodkaggle --pretrained --epochs 50
```

---

## 📈 Multi-Seed Experiments

Run experiments with 5 random seeds to produce statistically rigorous `Mean ± Std` results and LaTeX tables:

```bash
python main.py \
  --dataset floodkaggle \
  --model fsenet \
  --multiseed \
  --seeds 42 123 456 789 2024 \
  --epochs 50
```

**Generated LaTeX Output:**
```latex
mIoU: $0.8765 \pm 0.0123$
Dice: $0.9234 \pm 0.0089$
Prec: $0.9120 \pm 0.0105$
Rec:  $0.9351 \pm 0.0078$
B-IoU:$0.7812 \pm 0.0142$
FPS:  $124.50 \pm 2.10$
```

---

## 📁 Project Structure

```
MambaFloodLite/
├── main.py                     # Unified CLI entry point (train, multiseed, cross-eval)
├── cross_dataset_eval.py       # Cross-dataset generalization benchmark tool
├── benchmark.py                # Binary benchmark engine
├── benchmark_floodscene.py     # Multi-class benchmark engine
├── analyze.py                  # Grad-CAM and Best/Worst prediction analyzer
├── models/
│   ├── fsenet.py               # Proposed FSENet architecture (clean reference)
│   ├── deeplabv3.py            # DeepLabV3 (supports --pretrained)
│   ├── segformer.py            # SegFormer MiT backbone
│   ├── unet.py, segnet.py      # Classic baselines
│   ├── fast-scnn.py, stdc1.py  # Lightweight real-time baselines
│   └── ablation2_*.py ...      # 24 architectural ablation models (Ablation 2 to 24)
├── losses/
│   ├── bce.py, dice.py         # Binary loss functions
│   ├── ce.py                   # Multi-class CrossEntropy & CE-Dice loss
│   ├── focal.py, tversky.py    # Class imbalance losses
│   └── lovasz.py, lovasz_focal.py # Boundary & IoU surrogate losses
├── utils/
│   ├── dataloader.py           # Robust dataset loader (PNG, JPG, JSON polygons)
│   ├── trainer.py              # Reproducible training loop
│   └── metrics.py              # Mathematical FLOPs, mIoU, Dice, Precision, Recall, B-IoU, FWIoU
├── requirements.txt            # Cleaned Python dependencies
└── README.md
```

---

## ✅ Reproducibility Features

- Fixed random seed for Python, NumPy, PyTorch, and CUDA.
- Deterministic CUDA operations (`torch.backends.cudnn.deterministic = True`).
- Deterministic DataLoader workers with per-sample seed hashing.
- Complete state dictionary saving (weights, optimizer, RNG states).
