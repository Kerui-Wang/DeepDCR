# Feature definitions

This document defines the major feature groups used by DeepDCR for surgical difficulty prediction.

## Clinical variables

| Feature | Unit | Description |
|---|---:|---|
| surgical_eye | categorical | Operated eye / surgical side. |
| sex | categorical | Biological sex. |
| age | years | Age at surgery. |
| symptoms | categorical | Symptom category code. |
| duration_of_symptoms | months | Symptom duration before surgery. |
| severity_of_symptoms(MUNK) | score | Munk epiphora severity score. |
| previous_treatment_history | binary | Previous lacrimal intervention. |
| systemic_medical_history | binary | Relevant systemic medical history. |

## Segmentation-derived morphology and thickness features

| Feature | Unit | Description |
|---|---:|---|
| lac_volume_mm3 | mm³ | Lacrimal sac volume. |
| lac_surface_area_mm2 | mm² | Lacrimal sac surface area. |
| lac_sphericity | unitless | Lacrimal sac sphericity. |
| lac_equiv_diameter_mm | mm | Equivalent lacrimal sac diameter. |
| lac_elongation_pca | unitless | PCA-based lacrimal sac elongation. |
| nas_ipsi_volume_mm3 | mm³ | Ipsilateral nasal cavity / endonasal surgical corridor volume. |
| nas_ipsi_surface_area_mm2 | mm² | Ipsilateral nasal cavity surface area. |
| dist_lac_cent_to_nas_cent_mm | mm | Distance from lacrimal sac centroid to ipsilateral nasal cavity centroid. |
| bone_path_total_len_mm | mm | Estimated bony path length along the surgical corridor. |
| max_ipsi_volume_mm3 | mm³ | Ipsilateral maxilla volume at the En-DCR site. |
| max_ipsi_surface_area_mm2 | mm² | Ipsilateral maxilla surface area. |
| max_bone_burden_within_lac_dilate_mm3 | mm³ | Peri-lacrimal bone burden within a lacrimal-sac dilation ROI. |
| angle_maxilla_majoraxis_to_sagittal_deg | degrees | Long-axis angle of the maxillary region relative to the sagittal plane. |
| maxilla_thickness_max_mm | mm | Maximum local bone-window thickness. |
| maxilla_thickness_mean_mm | mm | Mean local bone-window thickness. |
| maxilla_thickness_p95_mm | mm | 95th percentile local bone-window thickness. |

## Deep imaging embeddings

`dl_00` to `dl_29` are PCA-reduced features derived from the frozen nnU-Net encoder bottleneck.

## Important note on nasal-cavity features

The nasal-cavity features are ipsilateral, surgery-side measurements. They are not bilateral nasal-cavity totals.
