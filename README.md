# 🧠 Brain Tumor Detection & Explainability Framework
### MRI-Based Classification using ViT · PDSCNN · GradCAM · RELM

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![XAI](https://img.shields.io/badge/XAI-GradCAM%20%7C%20RELM-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Model Components](#model-components)
5. [Explainability (XAI)](#explainability-xai)
6. [Project Structure](#project-structure)
7. [Setup & Installation](#setup--installation)
8. [Usage](#usage)
9. [Results & Metrics](#results--metrics)
10. [Progress Tracker](#progress-tracker)
11. [References](#references)

---

## 🎯 Project Overview

This project presents a **hybrid, explainable deep learning framework** for brain tumor classification from MRI images. The system combines:

- **ViT (Vision Transformer)** — for global context and long-range feature extraction via self-attention
- **PDSCNN (Parallel Depthwise Separable CNN)** — for efficient local spatial feature extraction with reduced parameters
- **GradCAM (Gradient-weighted Class Activation Mapping)** — for visual explanations highlighting discriminative regions
- **RELM (Relevance-based Explanation via Layer-wise Mapping)** — for fine-grained, attribution-level explainability

The model classifies MRI scans into **4 classes**:

| Label | Description |
|-------|-------------|
| `glioma` | Malignant brain tumor originating from glial cells |
| `meningioma` | Usually benign tumor arising from meninges |
| `pituitary` | Tumor located in the pituitary gland |
| `no_tumor` | Healthy brain — no tumor detected |

> **Clinical Motivation**: Accurate and explainable AI is critical in medical imaging. Radiologists need to trust and understand model decisions. This framework ensures both high accuracy AND interpretability.

---

## 🏗️ Architecture

```
MRI Input (224x224 RGB)
        |
        +-----------------------------------+
        |                                   |
  +-----v------+                  +---------v-----------+
  |  PDSCNN    |                  |  Vision Transformer  |
  |  Branch    |                  |       (ViT)          |
  |            |                  |                      |
  | Depthwise  |                  |  Patch Embedding     |
  | Separable  |                  |  + Position Encoding |
  | Conv Layers|                  |  + Multi-Head Attn   |
  | (parallel) |                  |  Transformer Blocks  |
  +-----+------+                  +---------+-----------+
        |                                   |
        +---------------+-------------------+
                        |
               +--------v--------+
               |  Feature Fusion  |
               |  (Concat + MLP)  |
               +--------+--------+
                        |
               +--------v--------+
               |  Classifier Head |
               |  (FC + Softmax)  |
               +--------+--------+
                        |
                  4-class output
                        |
        +---------------+---------------+
        |                               |
  +-----v------+               +--------v-----+
  |  GradCAM   |               |    RELM      |
  |  Heatmap   |               |  Attribution |
  |  Overlay   |               |   Maps       |
  +------------+               +--------------+
```

---

## 📦 Dataset

### Primary Dataset

| Source | Images | Classes | Format |
|--------|--------|---------|--------|
| **Masoud Nickparvar (Kaggle)** | 7,023 | 4 | JPEG/PNG |

```
Kaggle: masoudnickparvar/brain-tumor-mri-dataset
```
- Combined from: Figshare + SARTAJ + Br35H datasets
- Pre-cleaned and merged — **recommended starting point**
- Split: `Training/` and `Testing/` folders already provided

### Supplementary Datasets

| Source | Images | Notes |
|--------|--------|-------|
| Figshare (Jun Cheng) | 3,064 T1-weighted | `.mat` format, per-slice labels, clinical provenance |
| SARTAJ (Kaggle) | ~2,800 | 4-class, older source dataset |
| BraTS (Synapse) | Multi-modal | Pixel-level segmentation masks (for future work) |
| MRI w/ Bounding Boxes | 5,249 | YOLO-format boxes (for detection extension) |

### Data Preprocessing Pipeline
- [ ] Resize to **224x224** (ViT patch compatibility)
- [ ] Normalize with ImageNet mean/std `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`
- [ ] Augmentations: Random Horizontal Flip, Rotation (±15°), Color Jitter, Random Crop
- [ ] Train/Val/Test split: **70% / 15% / 15%**
- [ ] Class imbalance handling: Weighted sampler / oversampling

---

## 🔬 Model Components

### 1. ViT (Vision Transformer)

| Parameter | Value |
|-----------|-------|
| Patch Size | 16x16 |
| Image Size | 224x224 |
| Num Patches | 196 |
| Embedding Dim | 768 |
| Transformer Layers | 12 |
| Attention Heads | 12 |
| MLP Ratio | 4.0 |
| Dropout | 0.1 |
| Pretrained | ImageNet-21k (ViT-B/16) |

**Key Idea**: Divides the MRI image into fixed-size patches, linearly embeds each patch, and processes them as a sequence using Transformer encoder blocks. The [CLS] token serves as the global representation for classification.

---

### 2. PDSCNN (Parallel Depthwise Separable CNN)

**Architecture Design**:
```
Input (224x224x3)
     |
+----v------------------------------------+
|         Parallel Branches               |
|  +------------+  +------------------+  |
|  | Branch A   |  |   Branch B       |  |
|  | 3x3 DW     |  |  5x5 DW Conv     |  |
|  | Conv       |  |  (Large RF)      |  |
|  +------+-----+  +--------+---------+  |
|         +----------+------+            |
|               Concat / Add             |
+---------------------+------------------+
                      |
               Pointwise Conv (1x1)
               BN + ReLU
                      |
                MaxPool / Stride
                      |
               Repeat x N stages
                      |
               Global Avg Pool
                      |
               Flatten → Feature Vector
```

**Why PDSCNN?**
- Depthwise Separable Convolutions reduce parameters by ~8-9x
- Parallel branches capture multi-scale local features simultaneously
- Complementary to ViT's global attention mechanism
- Lightweight yet effective for medical image feature extraction

---

### 3. Hybrid Fusion Strategy

| Strategy | Description |
|----------|-------------|
| **Concatenation** | Concat ViT [CLS] token + PDSCNN Global Avg Pool vector |
| **Attention Fusion** | Cross-attention between CNN and ViT features |
| **Weighted Sum** | Learnable alpha * ViT_feat + (1-alpha) * CNN_feat |

> **Current Plan**: Start with Concatenation → benchmark → try Attention Fusion

---

## 🔍 Explainability (XAI)

### GradCAM (Gradient-weighted Class Activation Mapping)
- Computes gradients of the class score w.r.t. the **last convolutional feature map**
- Produces a coarse heatmap highlighting regions most important for the predicted class
- Applied to the **PDSCNN branch** (Conv layers are amenable to GradCAM)
- Overlaid on original MRI slice for visual inspection

**Output**: Colour-coded heatmap (blue → red) overlaid on MRI

---

### RELM (Relevance-based Explanation via Layer-wise Mapping)
- Layer-wise relevance propagation adapted for Transformer architectures
- Assigns **pixel-level relevance scores** to each input pixel
- Applied to the **ViT branch** — traces attention back to input patches
- Produces finer-grained attribution maps compared to GradCAM

**Output**: Per-pixel relevance score map, normalized and overlaid on MRI

---

### Combined Explainability View
```
Original MRI  |  GradCAM Heatmap  |  RELM Attribution  |  Fused View
```

---

## 📁 Project Structure

```
brain_tumor/
|
+-- data/
|   +-- raw/                   # Raw downloaded dataset
|   +-- processed/             # Preprocessed images (224x224, normalized)
|   +-- splits/                # train.csv, val.csv, test.csv
|
+-- models/
|   +-- vit/
|   |   +-- vit_model.py       # ViT implementation / pretrained wrapper
|   |   +-- vit_config.py      # Hyperparameters
|   +-- pdscnn/
|   |   +-- pdscnn_model.py    # PDSCNN architecture
|   |   +-- blocks.py          # DS Conv blocks, Parallel branch modules
|   +-- fusion/
|   |   +-- hybrid_model.py    # Combined ViT + PDSCNN hybrid
|   +-- classifier.py          # Final classification head
|
+-- xai/
|   +-- gradcam.py             # GradCAM implementation
|   +-- relm.py                # RELM implementation
|   +-- visualize.py           # Overlay heatmaps on MRI
|
+-- training/
|   +-- train.py               # Main training loop
|   +-- validate.py            # Validation loop
|   +-- loss.py                # Loss functions (CE + optional Focal Loss)
|   +-- scheduler.py           # LR scheduler configs
|
+-- evaluation/
|   +-- metrics.py             # Accuracy, F1, AUC-ROC, Confusion Matrix
|   +-- evaluate.py            # Full evaluation on test set
|
+-- notebooks/
|   +-- 01_data_exploration.ipynb
|   +-- 02_model_training.ipynb
|   +-- 03_explainability.ipynb
|   +-- 04_results_analysis.ipynb
|
+-- configs/
|   +-- config.yaml            # Master config (paths, hyperparams, seeds)
|
+-- checkpoints/               # Saved model weights (.pth)
|
+-- results/
|   +-- plots/                 # Training curves, confusion matrix
|   +-- xai_outputs/           # GradCAM / RELM visualizations
|
+-- requirements.txt
+-- braintumor.txt             # Dataset source notes
+-- README.md                  # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 8GB+ VRAM recommended

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd brain_tumor

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt (planned)
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0                 # ViT pretrained models
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
opencv-python>=4.8.0
grad-cam>=1.4.8             # pytorch-grad-cam
Pillow>=10.0.0
tqdm>=4.66.0
pyyaml>=6.0
albumentations>=1.3.0
```

---

## 🚀 Usage

### Running the Web Interface

We have provided a modern web interface for you to easily test the standalone PDSCNN model.

#### 1. Setup the Checkpoint
Ensure your trained model is placed in the correct path:
`models/checkpoints/pdscnn.pth`

#### 2. Start the Backend Server
```bash
cd backend
npm install
npm start
```
*Runs on http://localhost:3001*

#### 3. Start the Frontend Application
```bash
cd frontend
npm install
npm run dev
```
*Runs on http://localhost:5173*

### CLI Training & Evaluation
```bash
# Training
python training/train.py --config configs/config.yaml --model hybrid

# Evaluation
python evaluation/evaluate.py --checkpoint checkpoints/best_model.pth

# Generate XAI Explanations
python xai/visualize.py --image path/to/mri.jpg --checkpoint checkpoints/best_model.pth
```

---

## 📊 Results & Metrics

> Results will be populated as training experiments complete.

### Classification Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| PDSCNN (baseline) | — | — | — | — | — |
| ViT-B/16 (baseline) | — | — | — | — | — |
| **Hybrid ViT + PDSCNN** | — | — | — | — | — |

### Per-Class Performance (Hybrid Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Glioma | — | — | — | — |
| Meningioma | — | — | — | — |
| Pituitary | — | — | — | — |
| No Tumor | — | — | — | — |

### Explainability Quality

| Metric | GradCAM | RELM |
|--------|---------|------|
| Faithfulness Score | — | — |
| Localization Accuracy | — | — |
| Human Evaluation (Radiologist) | — | — |

---

## ✅ Progress Tracker

> **HOW TO USE**: This section is updated after every task completion.
> When resuming work, read this section FIRST to understand current state and next steps.

### Phase 1: Project Setup & Data
- [x] Project initialized and README created (2026-07-26)
- [x] Dataset sources identified and documented (braintumor.txt)
- [ ] Dataset downloaded (Masoud Nickparvar — Kaggle)
- [ ] Directory structure created (data/, models/, xai/, etc.)
- [ ] requirements.txt created and environment set up
- [ ] configs/config.yaml created with all hyperparameters
- [ ] 01_data_exploration.ipynb — EDA & class distribution analysis
- [ ] Data preprocessing pipeline implemented

### Phase 2: Model Development
- [ ] PDSCNN architecture implemented (models/pdscnn/)
- [ ] ViT wrapper/fine-tuning setup (models/vit/)
- [ ] Hybrid fusion module implemented (models/fusion/)
- [ ] Unit tests for all model components pass

### Phase 3: Training
- [ ] Training loop implemented (training/train.py)
- [ ] PDSCNN baseline training complete
- [ ] ViT baseline training complete
- [ ] Hybrid model training complete
- [ ] Best model checkpoints saved

### Phase 4: Explainability (XAI)
- [ ] GradCAM implementation (xai/gradcam.py)
- [ ] RELM implementation (xai/relm.py)
- [ ] Visualization pipeline (xai/visualize.py)
- [ ] 03_explainability.ipynb — XAI demos complete

### Phase 5: Evaluation & Analysis
- [ ] Metrics module implemented (evaluation/metrics.py)
- [ ] Confusion matrix plotted
- [ ] AUC-ROC curves plotted
- [ ] 04_results_analysis.ipynb — full analysis complete

### Phase 6: Documentation & Finalization
- [ ] All notebooks cleaned and documented
- [ ] Paper/report draft started
- [x] Web Demo (React Frontend + Node Backend) created

### Last Completed Task
> **2026-08-25**: Created a stunning React+Vite frontend and a Node.js Express backend to serve model inferences.

### Next Immediate Task
> **TODO**: Download dataset from Kaggle (`masoudnickparvar/brain-tumor-mri-dataset`) and create the directory structure + requirements.txt.

---

## 🔖 References

| Paper / Resource | Link |
|-----------------|------|
| An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT) | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) |
| Grad-CAM: Visual Explanations from Deep Networks | [arXiv:1610.02391](https://arxiv.org/abs/1610.02391) |
| MobileNets: Efficient CNNs (Depthwise Separable Conv) | [arXiv:1704.04861](https://arxiv.org/abs/1704.04861) |
| Brain Tumor MRI Dataset (Masoud Nickparvar) | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Figshare Brain Tumor Dataset (Jun Cheng) | [Figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) |
| timm — PyTorch Image Models | [GitHub](https://github.com/huggingface/pytorch-image-models) |
| pytorch-grad-cam | [GitHub](https://github.com/jacobgil/pytorch-grad-cam) |

---

## 👥 Authors

> Add your name and institution here.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

> **Last Updated**: 2026-07-26 | **Status**: Phase 1 — Project Initialized
