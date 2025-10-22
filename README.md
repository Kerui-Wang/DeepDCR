# DeepDCR: Deep Learning for DCR Surgery Difficulty Prediction
Code for "A CT Dacryocystography-Based Deep Learning Approach for Lacrimal Duct Segmentation and Surgical Difficulty Assessment to Guide Management of Primary acquired Nasolacrimal Duct Obstruction"

# Introduction
DeepDCR is a deep learning model designed to predict surgical difficulty in endoscopic dacryocystorhinostomy (En-DCR) based on preoperative CT-DCG imaging and clinical features.

# Model Architecture
DeepDCR employs a dual-path architecture:
Imaging Segmentation Path: 3D CNN based on nnUNet for precise extraction of lacrimal sac and nasal structure features from preoperative CT. We utilized the nnU-Net framework for automatic segmentation of the target regions of interest (ROIs). The nnU-Net is a self-configuring deep learning-based segmentation method that automatically adapts to any given dataset, determining key parameters such as patch size, batch size, and data augmentation strategies (https://github.com/MIC-DKFZ/nnUNet). The network architecture was a 3D full-resolution U-Net (3d_fullres) with six encoding and decoding stages. The model consisted of convolutional layers with kernel sizes of 3×3×3 and instance normalization. The number of feature channels started at 32 in the first stage and doubled after each downsampling step, up to a maximum of 320. The input patch size was set to 56×224×192 voxels based on the dataset characteristics.
Classification Prediction Path: Multimodal network integrating imaging features with clinical features, outputting surgical difficulty classification (Normal=0, Difficult=1)

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
   git clone https://github.com/Kerui-Wang/DeepDCR.git
   cd DeepDCR
2. Install Dependencies:
   pip install -r requirements.txt
3. Download Model Weights:
   Model weights are available via Hugging Face Hub:from huggingface_hub import snapshot_download
                                                    snapshot_download(repo_id="kerui7/DeepDCR_model_weight", local_dir="./weights")
# Data Preparation
Ensure the following directory structure:
data/
├── train/
│   ├── images/ (Raw CT images)
│   └── labels/ (Segmentation annotations)
├── test/
│   ├── images/
│   └── labels/
└── clinical_data.csv (Clinical features)

# Performance Evaluation
Performance metrics on independent test set(30 patients):
DeepDCR achieved Dice scores of 0.810 (lacrimal sac), 0.834 (maxilla), and 0.825 (nasal cavity). On external testing (n=30), DeepDCR demonstrated an AUC of 0.871 (95% CI: 0.720-0.978) with perfect sensitivity of 100.0% (95% CI: 74.1%-100.0%) and specificity of 73.7% (95% CI: 52.4%-93.3%). 

# Contributing
We welcome Issues and Pull Requests to improve this project.

#  License
This project is licensed under the MIT License. See LICENSE file for details.

# Ethics Statement
This research adheres to medical research ethics guidelines. All data were anonymized and approved by the institutional review board.

# Disclaimer: 
This model is intended to assist clinical decision-making and should not replace professional medical judgment. Users should make final decisions based on clinical experience and individual patient circumstances.
