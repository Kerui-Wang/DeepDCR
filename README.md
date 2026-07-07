# DeepDCR: AI-assisted En-DCR surgical difficulty stratification from CT dacryocystography

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Research Use](https://img.shields.io/badge/Use-research--only-orange)
![DeepDCR](https://img.shields.io/badge/DeepDCR-v1.0.0-teal)
![Medical AI](https://img.shields.io/badge/Medical%20AI-CT--DCG-lightgrey)

DeepDCR is a research-use artificial intelligence framework for preoperative endonasal endoscopic dacryocystorhinostomy (En-DCR) planning from CT dacryocystography (CT-DCG). It combines two-stage 3D nnU-Net segmentation, segmentation-derived morphometric and bone-window thickness features, PCA-reduced nnU-Net encoder embeddings, clinical variables and a calibrated Linear SVM classifier to estimate case-level surgical difficulty.

## Key functions

- Automated segmentation of the lacrimal sac, maxilla at the En-DCR site and ipsilateral nasal cavity / endonasal surgical corridor.
- Morphometric and bone-window thickness feature extraction from predicted masks.
- Deep imaging embedding extraction from the nnU-Net encoder bottleneck.
- Calibrated surgical difficulty prediction.
- Research-use DeepDCR Workstation for CT overlay, 3D segmentation, bone-window thickness mapping, structured feature review and modular report export.

## Repository contents

| Folder or file | Description |
|---|---|
| `DeepDCR_segmentation/` | Segmentation workflow based on a two-stage 3D nnU-Net cascade. |
| `DeepDCR_classification/` | Feature integration, model training, validation and inference scripts. |
| `docs/` | Installation guide, feature definitions, model card, app guide and deployment notes. |
| `configs/` | Public example configuration files and feature schema. |
| `examples/` | Synthetic placeholder examples only. |
| `assets/` | Public figures and software screenshots. |
| `requirements.txt` | Python dependency list. |

## Model overview

DeepDCR follows a hybrid design:

1. CT-DCG preprocessing.
2. Coarse full-field segmentation.
3. Cropped high-resolution segmentation.
4. Paste-back to original image space.
5. Automated feature extraction.
6. nnU-Net encoder embedding extraction and PCA reduction.
7. Calibrated surgical difficulty prediction.
8. Case-level visualization and report export.

## Input

- De-identified preoperative CT-DCG data.
- Basic clinical variables required by the prediction model.

## Output

- Segmentation masks of surgery-relevant structures.
- Morphometric and bone-window thickness measurements.
- PCA-reduced deep imaging embeddings.
- Calibrated surgical difficulty probability.
- Structured case-level report.

## Feature groups

| Group | Description |
|---|---|
| Clinical variables | Age, sex, surgical eye, symptom duration, Munk score, previous treatment history and systemic history. |
| Lacrimal sac morphology | Volume, surface area, sphericity, equivalent diameter and elongation. |
| Ipsilateral nasal corridor | Surgery-side nasal cavity / endonasal corridor volume, surface area and spatial relationship to the lacrimal sac. |
| Maxilla and bone-window features | Ipsilateral maxilla morphology, peri-lacrimal bone burden and local bone-window thickness. |
| Deep imaging embeddings | PCA-reduced features derived from the frozen nnU-Net encoder bottleneck. |

The nasal-cavity features used for prediction are ipsilateral, surgery-side features rather than bilateral nasal-cavity totals.

## Research-use workstation

DeepDCR includes a secure research-use web workstation for case-level review. The workstation supports de-identified CT-DCG upload, CT-overlay visualization, interactive 3D segmentation, bone-window thickness mapping, structured feature tables, difficulty prediction and modular report export.

The workstation includes login authentication and access logging and is intended for institutional research review rather than autonomous clinical decision-making.

## What is not included

This repository does not contain:

- Patient CT-DCG images.
- Manual segmentation masks.
- Identifiable clinical metadata.
- Trained model weights.
- Internal out-of-fold prediction tables.
- Institutional file paths.
- Access logs or uploaded cases.

Model weights and de-identified data may be requested from the corresponding author for academic research, subject to institutional data-governance approval and data-transfer agreements.

## Installation

See [`docs/installation.md`](docs/installation.md).

## Quick start

See [`docs/quick_start.md`](docs/quick_start.md).

## Feature definitions

See [`docs/feature_definitions.md`](docs/feature_definitions.md).

## Model card

See [`docs/model_card.md`](docs/model_card.md).

## Disclaimer

DeepDCR is provided for research use only. It is not a certified medical device and is not intended for autonomous diagnosis, treatment selection or clinical decision-making. Any use of model outputs must be reviewed by qualified clinicians and validated under local institutional governance.

## Citation

If you use this repository, please cite the associated DeepDCR manuscript.
