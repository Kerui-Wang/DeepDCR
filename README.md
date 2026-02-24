# DeepDCR: Multimodal Deep Learning for En-DCR Surgical Difficulty and Prognosis Stratification
Code for DeepDCR, a CT-DCG–based multimodal AI framework for:

   Automated segmentation of surgically relevant anatomy for endoscopic dacryocystorhinostomy (En-DCR)
   
   Preoperative prediction of surgical difficulty (Normal vs Difficult)
   
   Preoperative prediction of 6-month postoperative outcome (Success vs Failure)

# Overview
DeepDCR is a clinically oriented pipeline for primary acquired nasolacrimal duct obstruction (PANDO) based on preoperative CT dacryocystography (CT-DCG). The framework combines:

   1.Two-stage cascade nnU-Net segmentation (coarse localization + fine segmentation)
   
   2.Segmentation-derived handcrafted morphology and thickness features
   
   3.Deep learning embeddings extracted from a frozen nnU-Net encoder bottleneck
   
   4.Clinical variables
   
   5.A probability-calibrated Linear SVM (selected as final classifier in the study)
   
The design goal is to provide interpretable, leakage-aware, and surgery-relevant preoperative risk stratification for En-DCR case planning and triage.

# What is included in this code release
This de-identified code release focuses on the imaging feature engineering and representation learning components, including:

   Morphological feature extraction from segmentation masks (lacrimal / maxilla / nasal space)
   
   Surgery-relevant maxilla thickness visualization and measurement
   
   nnU-Net encoder embedding extraction with leakage-safe PCA fitting (fit on internal OOF only)
   
   Helper scripts for downstream integration into hybrid tabular models (clinical + morphology + DL features)

# Example scripts (de-identified)

   dcr_surgery_morph_features_extraction.py
   
   extract_nnunet_embeddings.py
   
   maxilla_thickness_axial_half_surgery.py

# System Requirements

1. Hardware Requirements:
   
   GPU: NVIDIA GPU with ≥ 8GB VRAM (Recommended: RTX 3080 or A100);
   
   Memory: ≥ 16GB RAM;
   
   Storage: ≥ 10GB available space
   
2. Software Requirements:
   
   Operating System: Ubuntu 18.04+, Windows 10+, or macOS 12+;
   
   Python: 3.7-3.10

# Installation & Dependencies
1. Clone Repository:
   ```
   git clone https://github.com/Kerui-Wang/DeepDCR.git
   cd DeepDCR
   ```
2. Install Dependencies:
   ```pip install -r requirements.txt```
3. Download Model Weights:
   
   Pre-trained model weights are **not publicly available** in this repository.
   
   Upon reasonable request and **with permission from the authors**, model weights may be provided for academic research purposes only.
   
   Please contact the corresponding author (or the authors listed in the manuscript) to request access.
   
# Data Preparation
Ensure the following directory structure:
```
data/
├── train/
│   ├── images/ (Raw CT images)
│   └── labels/ (Segmentation annotations)
├── test/
│   ├── images/
│   └── labels/
└── clinical_data.csv (Clinical features)
```

# Performance Evaluation
DeepDCR was evaluated on two binary tasks:

   1.Surgical difficulty prediction
   
      Difficult vs normal (based on surgeon-specific operative-time percentile threshold)
      
   2.6-month outcome prediction
   
      Failure vs success (based on follow-up patency/recurrence criteria)
      
The manuscript describes a retrospective multi-center study design and external validation strategy in detail.

# Contributing
We welcome Issues and Pull Requests to improve this project.

#  License
This project is licensed under the MIT License. See LICENSE file for details.

# Ethics Statement
This research adheres to medical research ethics guidelines. All data were anonymized and approved by the institutional review board.

# Disclaimer 
This model is intended to assist clinical decision-making and should not replace professional medical judgment. Users should make final decisions based on clinical experience and individual patient circumstances.

# Citation
If you use this code or adapt parts of the feature engineering / embedding extraction pipeline, please cite the associated DeepDCR study/manuscript (and the nnU-Net framework) as appropriate.

# Code availability statement
The de-identified code for feature extraction, embedding extraction, and visualization used in the DeepDCR workflow is publicly available in this repository.

Pretrained model weights are not publicly released.
