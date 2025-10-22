import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.ndimage import distance_transform_edt
import os

# Set font for English text
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


def calculate_thickness(bone_mask, voxel_spacing):
    """Calculate thickness distribution of bone segmentation"""
    bone_mask_uint8 = bone_mask.astype(np.uint8)
    inner_distance = distance_transform_edt(bone_mask_uint8, sampling=voxel_spacing)
    outer_mask = 1 - bone_mask_uint8
    outer_distance = distance_transform_edt(outer_mask, sampling=voxel_spacing)
    thickness_map = inner_distance + outer_distance
    thickness_map[outer_mask == 1] = 0
    bone_thickness = thickness_map[bone_mask_uint8 == 1]
    mean_thickness = np.mean(bone_thickness) if bone_thickness.size else 0
    min_thickness = np.min(bone_thickness) if bone_thickness.size else 0
    max_thickness = np.max(bone_thickness) if bone_thickness.size else 0
    return thickness_map, mean_thickness, min_thickness, max_thickness


def visualize_bone_thickness_analysis(ct_path, label_path, output_dir, bone_label=2, slice_direction="z", slice_idx=None):
    """
    Visualize bone thickness analysis from CT and segmentation data
    
    Parameters:
    ct_path: Path to CT image file
    label_path: Path to segmentation label file
    output_dir: Directory to save output images
    bone_label: Label value for bone in segmentation (default: 2)
    slice_direction: Direction of slice to display ("z", "y", or "x")
    slice_idx: Index of slice to display (if None, uses middle slice)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Read image data
        ct = sitk.ReadImage(ct_path)
        label = sitk.ReadImage(label_path)
        ct_array = sitk.GetArrayFromImage(ct)
        label_array = sitk.GetArrayFromImage(label)
        spacing = ct.GetSpacing()

        # Adjust voxel spacing order to match array dimensions (Z, Y, X)
        voxel_spacing = (spacing[2], spacing[1], spacing[0])

        bone_mask = (label_array == bone_label)

        # Automatically calculate valid slice index (if not specified)
        if slice_idx is None:
            if slice_direction == "z":
                slice_idx = ct_array.shape[0] // 2  # Take middle slice along Z-axis
            elif slice_direction == "y":
                slice_idx = ct_array.shape[1] // 2  # Take middle slice along Y-axis
            elif slice_direction == "x":
                slice_idx = ct_array.shape[2] // 2  # Take middle slice along X-axis
            else:
                raise ValueError("slice_direction must be 'z', 'y', or 'x'")

        # Extract slice based on direction
        if slice_direction == "z":
            ct_slice = ct_array[slice_idx, :, :]
            bone_mask_slice = bone_mask[slice_idx, :, :]
        elif slice_direction == "y":
            ct_slice = ct_array[:, slice_idx, :]
            bone_mask_slice = bone_mask[:, slice_idx, :]
        elif slice_direction == "x":
            ct_slice = ct_array[:, :, slice_idx]
            bone_mask_slice = bone_mask[:, :, slice_idx]

        # Calculate thickness
        thickness_map, mean_thick, min_thick, max_thick = calculate_thickness(bone_mask, voxel_spacing)

        # Extract thickness map slice for corresponding direction
        if slice_direction == "z":
            thickness_slice = thickness_map[slice_idx, :, :]
        elif slice_direction == "y":
            thickness_slice = thickness_map[:, slice_idx, :]
        elif slice_direction == "x":
            thickness_slice = thickness_map[:, :, slice_idx]

        # Extract patient ID from file path for naming
        patient_id = os.path.basename(ct_path).split('_')[0]

        # 1. Generate thickness distribution histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(thickness_map[bone_mask], bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel('Thickness (mm)', fontsize=12)
        ax.set_title(f'Thickness Distribution - {patient_id}', fontsize=14)
        hist_path = os.path.join(output_dir, f'{patient_id}_bone_hist.png')
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 2. Generate thickness overlay slice plot (with colorbar and statistics)
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))

        # Display CT background - remove origin='lower' parameter to maintain consistency with segmentation visualization
        ax.imshow(ct_slice, cmap='gray')

        # Create mask for thickness map (only show bone regions)
        masked_thickness = np.ma.masked_where(thickness_slice == 0, thickness_slice)

        # Display thickness using jet colormap
        im = ax.imshow(masked_thickness, cmap='jet', alpha=0.7,
                       vmin=min_thick, vmax=max_thick)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Thickness (mm)', fontsize=12)

        # Add statistics box in upper right corner
        stats_text = (f"Max Thickness: {max_thick:.2f} mm\n"
                      f"Mean Thickness: {mean_thick:.2f} mm\n"
                      f"Min Thickness: {min_thick:.2f} mm")

        # Use text box for better readability
        bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8)
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=bbox_props)

        ax.set_title(f'Bone Thickness Analysis - {patient_id}', fontsize=16)
        slice_path = os.path.join(output_dir, f'{patient_id}_bone_slice.png')
        plt.savefig(slice_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Generated images: {hist_path} and {slice_path}")

    except Exception as e:
        print(f"Error: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Update these paths to your actual file locations
    visualize_bone_thickness_analysis(
        ct_path="path/to/ct/image.nii.gz",
        label_path="path/to/segmentation/labels.nii.gz",
        output_dir="path/to/output/directory",
        bone_label=2,
        slice_direction="z",  # Consistent with segmentation visualization
        slice_idx=30  # Consistent with segmentation visualization
    )