# Quick start

This repository does not include patient CT-DCG data or trained model weights.

A typical DeepDCR workflow includes:

1. Prepare de-identified CT-DCG data.
2. Run two-stage segmentation.
3. Extract morphology and bone-window thickness features.
4. Extract PCA-reduced deep imaging embeddings.
5. Apply the calibrated surgical difficulty classifier.
6. Review the case-level output in the research-use workstation.
7. Export a structured report.

Synthetic placeholder files are provided in `examples/synthetic_case/` for schema inspection only.
