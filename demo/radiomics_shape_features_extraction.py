import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure
from scipy.ndimage import distance_transform_edt
import os
import warnings

# Set font for English text
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


# Morphological feature extraction function
def extract_morphological_features(seg_path, image_path, clinical_data=None):
    """
    Extract surgery-related morphological features from segmentation results
    
    Parameters:
    seg_path: Segmentation file path
    image_path: Original CT image path
    clinical_data: Clinical data dictionary (optional)
    
    Returns:
    feature_dict: Dictionary of extracted features
    """
    try:
        # Read data
        seg_img = sitk.ReadImage(seg_path)
        ct_img = sitk.ReadImage(image_path)
        seg_array = sitk.GetArrayFromImage(seg_img)
        ct_array = sitk.GetArrayFromImage(ct_img)
        spacing = ct_img.GetSpacing()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]  # Voxel volume (mm³)

        # Extract case ID from file path
        case_id = os.path.basename(seg_path).split('.')[0]

        features = {}

        # 1.1 Lacrimal Features -----------------------------------------------------------
        lacrimal_mask = (seg_array == 1).astype(np.uint8)
        # Volume features
        lacrimal_voxels = np.sum(lacrimal_mask)
        features['lacrimal_volume'] = lacrimal_voxels * voxel_volume
        # Morphological features
        props = measure.regionprops(lacrimal_mask)
        if props:
            lacrimal_props = props[0]
            features['lacrimal_equiv_diameter'] = lacrimal_props.equivalent_diameter * spacing[0]
            features['lacrimal_solidity'] = lacrimal_props.solidity
            # Alternative eccentricity feature: use elongation (3D compatible)
            if lacrimal_props.axis_major_length > 0:
                elongation = lacrimal_props.axis_minor_length / lacrimal_props.axis_major_length
                features['lacrimal_elongation'] = elongation
            else:
                features['lacrimal_elongation'] = 0
        else:
            # Set default values if no region detected
            features['lacrimal_equiv_diameter'] = 0
            features['lacrimal_solidity'] = 0
            features['lacrimal_elongation'] = 0

        # 1.2 Maxilla Thickness Features -----------------------------------------------------
        maxilla_mask = (seg_array == 2).astype(np.uint8)

        # Calculate bone thickness
        def calculate_thickness(mask, spacing):
            mask_uint8 = mask.astype(np.uint8)
            inner_dist = distance_transform_edt(mask_uint8, sampling=spacing)
            outer_dist = distance_transform_edt(1 - mask_uint8, sampling=spacing)
            thickness_map = inner_dist + outer_dist
            thickness_map[mask_uint8 == 0] = 0
            return thickness_map

        thickness_map = calculate_thickness(maxilla_mask, spacing[::-1])  # Note spacing order

        # Thickness statistics
        maxilla_thickness = thickness_map[maxilla_mask > 0]
        features['maxilla_mean_thickness'] = np.mean(maxilla_thickness) if maxilla_thickness.size else 0
        features['maxilla_min_thickness'] = np.min(maxilla_thickness) if maxilla_thickness.size else 0
        features['maxilla_max_thickness'] = np.max(maxilla_thickness) if maxilla_thickness.size else 0
        features['maxilla_thickness_std'] = np.std(maxilla_thickness) if maxilla_thickness.size else 0

        # 1.3 Nasal Space Features -------------------------------------------------------
        nasal_mask = (seg_array == 3).astype(np.uint8)
        # Volume features
        nasal_voxels = np.sum(nasal_mask)
        features['nasal_volume'] = nasal_voxels * voxel_volume
        # Spatial features
        nasal_props = measure.regionprops(nasal_mask)
        if nasal_props:
            features['nasal_centroid_z'] = nasal_props[0].centroid[0] * spacing[2]
            # Spatial curvature features (simplified version)
            try:
                verts, faces, _, _ = measure.marching_cubes(nasal_mask, level=0.5, spacing=spacing[::-1])
                features['nasal_surface_area'] = measure.mesh_surface_area(verts, faces)
            except:
                features['nasal_surface_area'] = 0
        else:
            features['nasal_centroid_z'] = 0
            features['nasal_surface_area'] = 0

        # 1.4 Relative Position Features -------------------------------------------------------
        if props and nasal_props:
            # Distance between lacrimal and nasal centers
            lacrimal_center = np.array(lacrimal_props.centroid)
            nasal_center = np.array(nasal_props[0].centroid)
            distance = np.linalg.norm((lacrimal_center - nasal_center) * spacing)
            features['lacrimal_nasal_distance'] = distance
        else:
            features['lacrimal_nasal_distance'] = 0

        # 1.5 Merge Clinical Data ------------------------------------------------------
        if clinical_data:
            features.update(clinical_data)

        return features
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return {}


# Feature extraction pipeline
def extract_features_for_dataset(seg_dir, image_dir, clinical_data_csv=None,
                                 output_file='shape_feature.csv'):
    """
    Extract features for entire dataset
    
    Parameters:
    seg_dir: Segmentation results directory
    image_dir: Original images directory
    clinical_data_csv: Clinical data CSV file path
    output_file: Output feature file name
    
    Returns:
    features_df: DataFrame containing all features
    """
    # Get case list
    seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.nii.gz')]
    case_ids = [f.split('.')[0] for f in seg_files]  # Assume filename format: patient_001.nii.gz

    # Load clinical data
    clinical_data = {}
    if clinical_data_csv and os.path.exists(clinical_data_csv):
        clinical_df = pd.read_csv(clinical_data_csv)
        # Ensure correct column names
        clinical_df.rename(columns=clinical_name_map, inplace=True)
        clinical_df.set_index('patient_id', inplace=True)
        clinical_data = clinical_df.to_dict('index')

    # Extract features for each case
    all_features = []
    for i, case_id in enumerate(case_ids):
        print(f"\nProcessing case {i + 1}/{len(case_ids)}: {case_id}")

        # Build file paths
        seg_path = os.path.join(seg_dir, f"{case_id}.nii.gz")

        # Find corresponding CT image (may have different modalities, take first)
        matching_files = [f for f in os.listdir(image_dir)
                          if f.startswith(case_id) and f.endswith('.nii.gz')]
        if not matching_files:
            print(f"Warning: No CT image found for {case_id}")
            continue
        ct_file = matching_files[0]
        ct_path = os.path.join(image_dir, ct_file)

        # Get clinical data
        case_clinical = clinical_data.get(case_id, {}) if clinical_data else {}

        # Extract morphological features
        try:
            morph_features = extract_morphological_features(
                seg_path, ct_path, case_clinical
            )
        except Exception as e:
            print(f"Morphological feature extraction error {case_id}: {str(e)}")
            morph_features = {}

        # Merge features
        features = {**morph_features}
        features['case_id'] = case_id

        # Add target variable (if exists)
        if 'difficulty' in case_clinical:
            features['difficulty'] = case_clinical['difficulty']

        all_features.append(features)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)

    # Handle missing values - fill numerical features with column mean
    for col in features_df.columns:
        if col == 'case_id':
            continue
        if features_df[col].dtype in [np.float64, np.int64]:
            features_df[col].fillna(features_df[col].mean(), inplace=True)
        else:
            features_df[col].fillna(0, inplace=True)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save features
    features_df.to_csv(output_file, index=False)
    print(f"\nFeature extraction completed, saved to {output_file}")

    return features_df


# Clinical feature mapping
clinical_name_map = {
    'clinical_0': 'Eye',
    'clinical_1': 'Sex',
    'clinical_2': 'Age',
    'clinical_3': 'Duration_of_symptoms',
    'clinical_4': 'Severity_of_symptoms',
    'clinical_5': 'Previous_treatment_history',
    'clinical_6': 'Systemic_medical_history'
}


# Main function - feature extraction pipeline
def extract_all_features():
    # Configure paths
    train_image_dir = "path/to/train/images"
    train_seg_dir = "path/to/train/segmentations"
    train_clinical_csv = "path/to/train/clinical_data.csv"

    test_image_dir = "path/to/test/images"
    test_seg_dir = "path/to/test/segmentations"
    test_clinical_csv = "path/to/test/clinical_data.csv"

    # Output directory
    output_base_dir = "path/to/output/ML_features"

    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Extract training set features
    train_features_file = os.path.join(output_base_dir, "shape_feature_train.csv")
    print("\n" + "=" * 50)
    print("Starting training set feature extraction...")
    print("=" * 50)
    train_features_df = extract_features_for_dataset(
        seg_dir=train_seg_dir,
        image_dir=train_image_dir,
        clinical_data_csv=train_clinical_csv,
        output_file=train_features_file
    )

    # Extract test set features
    test_features_file = os.path.join(output_base_dir, "shape_feature_test.csv")
    print("\n" + "=" * 50)
    print("Starting test set feature extraction...")
    print("=" * 50)
    test_features_df = extract_features_for_dataset(
        seg_dir=test_seg_dir,
        image_dir=test_image_dir,
        clinical_data_csv=test_clinical_csv,
        output_file=test_features_file
    )

    print("\n" + "=" * 50)
    print("Feature extraction pipeline completed!")
    print(f"Training set features saved to: {train_features_file}")
    print(f"Test set features saved to: {test_features_file}")
    print("=" * 50)


if __name__ == "__main__":
    extract_all_features()