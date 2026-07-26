# GST-UDA-Medical-Segmentation
Geometric Self-Training for Unsupervised Domain Adaptation in Multi-Modality Medical Image Segmentation
Here is a complete, professional, publication-grade `README.md` tailored specifically for your UDA project.

### Suggested Repository Name

* **Repository Name:** `GST-UDA-Medical-Segmentation`
* **Full Title:** Geometric Self-Training for Unsupervised Domain Adaptation in Multi-Modality Medical Image Segmentation

---

```markdown
# GST-UDA-Medical-Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for the paper: **"Cross-Modality Medical Image Segmentation via Fourier Domain Adaptation and Geometric Self-Training"**.

---

## 📖 Overview
Unsupervised Domain Adaptation (UDA) for multi-modal medical image segmentation often suffers from severe **structural bleeding** and boundary collapse due to cross-modality shifts (e.g., CT to MRI). While global texture alignment techniques (like Fourier Domain Adaptation) match macro-level distributions, they fail to constrain local boundary topology. 

This repository provides the complete implementation of our unified UDA framework featuring:
1. **Fourier Domain Adaptation (FDA):** Non-adversarial, spectrum-swapping global style alignment ($\beta = 0.05$).
2. **Class-Balanced Self-Training (CBST):** Entropy-minimized pseudo-labeling with class-weight penalties to handle severe volumetric imbalance (e.g., AMOS22 abdominal organs).
3. **Geometric Self-Training (GST):** A novel dual-branch boundary constraint that decouples the semantic body from the high-frequency topological edge manifold to eliminate boundary ballooning.
4. **Largest Connected Component (LCC) Post-Processing:** A deterministic 3D topological filter that removes isolated false-positive hallucinations.

---

## 🗂️ Repository Structure

```text
GST-UDA-Medical-Segmentation/
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── models/                    # Core neural network architectures
│   ├── dbsnet.py              # Unified DBS-Net (Dual-head body/edge extraction)
│   └── networks.py            # ResNet-9 Generator and PatchGAN Discriminator
├── datasets/                  # Dataset loaders & preprocessing pipelines
│   ├── amos22_dataset.py      # AMOS22 4-organ subset loader & LCC filter
│   └── mmwhs_dataset.py       # MMWHS cardiovascular dataset loader
├── utils/                     # Mathematical utilities & loss functions
│   ├── fda.py                 # Fast Fourier Transform & Amplitude Spectrum Swap
│   ├── losses.py              # SoftmaxWeightedLoss, JointLoss, StructureLoss
│   └── metrics.py             # 3D Dice Similarity Coefficient, HD95, ASD
├── experiments/               # Training scripts for AMOS22 & MMWHS
│   ├── amos22/                # Phase 1 and Phase 2 scripts for AMOS22
│   └── mmwhs/                 # Phase 1 and Phase 2 scripts for MMWHS
└── evaluation/                # Standalone inference and vector chart generation
    └── evaluate_inference.py  # Generates publication-ready PDFs (Radar, Bland-Altman, Error Maps)

```

---

## ⚙️ Installation & Requirements

1. **Clone the repository:**
```bash
git clone [https://github.com/YourUsername/GST-UDA-Medical-Segmentation.git](https://github.com/YourUsername/GST-UDA-Medical-Segmentation.git)
cd GST-UDA-Medical-Segmentation

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


*Main dependencies include:* `torch >= 2.0`, `torchvision`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `opencv-python`, `Pillow`, and `tqdm`.

---

## 📊 Data Preparation

Ensure your 2D preprocessed `.npy` image arrays and `.png` label masks are structured correctly in your local directory:

* **AMOS22:** 5-class mapping (Background, Liver, Right Kidney, Left Kidney, Spleen).
* **MMWHS:** 4-class mapping (Ascending Aorta, Left Atrium Cavity, Left Ventricle Cavity, Myocardium).

Update the root paths in the `CONFIG` dictionary of your respective experiment or configuration scripts.

---

## 🚀 Training Pipeline

Training is executed in a sequential, multi-phase progression to ensure stable optimization:

### Step 1: Supervised Baseline Training

Train the DBS-Net baseline segmentor on the source domain:

```bash
python experiments/amos22/train_supervised_baseline.py

```

### Step 2: Phase 1 - Global Alignment (FDA + CBST)

Initialize the generator, feature discriminator, and segmentor for domain adaptation:

```bash
python experiments/amos22/train_phase1_cbst.py

```

### Step 3: Phase 2 - Boundary Refinement (Geometric Self-Training)

Load your Phase 1 weights and execute the 50-epoch GST boundary-sharpening sprint:

```bash
python experiments/amos22/train_phase2_gst.py

```

---

## 📈 Evaluation & Inference

To evaluate a trained model checkpoint, compute volumetric 3D metrics (DSC, HD95, ASD), apply LCC post-processing, and export publication-grade vector charts (Radar plots, Error maps, and Bland-Altman plots), run:

```bash
python evaluation/evaluate_inference.py --task CT2MR --weights path/to/best_segmentor.pth

```

---

## 📝 Citation

If you find this code useful for your research or academic publication, please cite our work:

```bibtex
@article{motwani2026gstuda,
  title={Cross-Modality Medical Image Segmentation via Fourier Domain Adaptation and Geometric Self-Training},
  author={Motwani, Nilesh and Collaborators},
  journal={IEEE Transactions on Medical Imaging / Under Review},
  year={2026}
}

```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

```
