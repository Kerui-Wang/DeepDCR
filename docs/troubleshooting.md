# Troubleshooting

## nnU-Net not found

Please ensure nnU-Net v2 has been installed and is available in the current Python environment.

## CUDA unavailable

Please check:

- NVIDIA driver installation.
- CUDA-compatible PyTorch installation.
- Whether `torch.cuda.is_available()` returns `True`.

## Missing model weights

The public repository does not contain pretrained model weights.

Please contact the corresponding author for academic access, subject to institutional data-governance approval.

## Missing patient data

The public repository does not include patient CT-DCG images, manual masks or identifiable clinical metadata.

Synthetic placeholder files are provided only for schema inspection.
