# DeepDCR model card

## Model name

DeepDCR surgical difficulty prediction system.

## Tasks

1. CT-DCG anatomical segmentation.
2. Segmentation-derived feature extraction.
3. Surgical difficulty prediction.
4. Exploratory 6-month outcome prediction.

## Input

De-identified preoperative CT dacryocystography and required clinical variables.

## Output

- Segmentation masks.
- Morphometric and thickness measurements.
- PCA-reduced imaging embeddings.
- Calibrated surgical difficulty probability.
- Structured case-level report.

## Intended use

DeepDCR is intended for research-use preoperative En-DCR planning support, case-level review and prospective evaluation.

## Non-intended use

DeepDCR is not intended for:

- Autonomous clinical decision-making.
- Emergency decision-making.
- Diagnosis or treatment selection without clinician review.
- Use on non-CT-DCG imaging modalities without validation.
- Use outside institutional research governance.

## Limitations

- Retrospective development.
- Single tertiary surgical centre.
- Limited number of operating surgeons.
- Requires prospective multicentre validation.
- Model weights are not publicly released because of institutional data-governance restrictions.

## Safety statement

All outputs must be interpreted by qualified clinicians. The software is provided for research use only and is not a certified medical device.
