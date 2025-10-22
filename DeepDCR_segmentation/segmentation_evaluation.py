import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # For creating legend color patches
from matplotlib import colormaps  # Using new colormaps API
import os


def plot_segmentation_comparison(ct_path, label_path, pred_path, output_dir, slice_direction="z", slice_idx=None):
    """
    Plot segmentation comparison between CT image, ground truth, and prediction
    
    Parameters:
    ct_path: Path to CT image file
    label_path: Path to ground truth label file
    pred_path: Path to predicted segmentation file
    output_dir: Directory to save output images
    slice_direction: Direction of slice to display ("z", "y", or "x")
    slice_idx: Index of slice to display (if None, uses middle slice)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract patient ID from file path for naming
    patient_id = os.path.basename(ct_path).split('_')[0]

    # Read images and labels
    ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
    label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))

    # Print image dimensions to confirm orientation (e.g., (Z, Y, X))
    print(f"CT Image Shape: {ct.shape}")

    # Automatically calculate valid slice index (if not specified)
    if slice_idx is None:
        if slice_direction == "z":
            slice_idx = ct.shape[0] // 2  # Take middle slice along Z-axis
        elif slice_direction == "y":
            slice_idx = ct.shape[1] // 2  # Take middle slice along Y-axis
        elif slice_direction == "x":
            slice_idx = ct.shape[2] // 2  # Take middle slice along X-axis
        else:
            raise ValueError("slice_direction must be 'z', 'y', or 'x'")

    # Ensure index is valid
    if slice_direction == "z":
        if slice_idx >= ct.shape[0] or slice_idx < 0:
            slice_idx = ct.shape[0] - 1  # Automatically adjust to maximum valid index
        ct_slice = ct[slice_idx, :, :]
        label_slice = label[slice_idx, :, :]
        pred_slice = pred[slice_idx, :, :]
    elif slice_direction == "y":
        if slice_idx >= ct.shape[1] or slice_idx < 0:
            slice_idx = ct.shape[1] - 1
        ct_slice = ct[:, slice_idx, :]
        label_slice = label[:, slice_idx, :]
        pred_slice = pred[:, slice_idx, :]
    elif slice_direction == "x":
        if slice_idx >= ct.shape[2] or slice_idx < 0:
            slice_idx = ct.shape[2] - 1
        ct_slice = ct[:, :, slice_idx]
        label_slice = label[:, :, slice_idx]
        pred_slice = pred[:, :, slice_idx]

    # Visualization settings
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define class names and color mapping (adjust according to actual data)
    class_names = {
        0: "Background",
        1: "Lacrimal Sac",
        2: "Maxilla (En-DCR Site)",
        3: "Nasal Cavity (En-DCR Approach)"
    }
    num_classes = len(class_names)  # 4 classes (including background)

    # Use new colormaps API to replace deprecated get_cmap
    cmap = colormaps["jet"].resampled(num_classes)  # Discrete color mapping

    # Generate color-class correspondence legend
    legend_elements = [
        Patch(facecolor=cmap(i), edgecolor="k", label=class_names[i])
        for i in range(num_classes)
    ]

    # Display CT image
    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title(f"CT Image (Slice {slice_idx})")

    # Overlay ground truth labels
    axes[1].imshow(ct_slice, cmap="gray", alpha=0.8)
    label_img = axes[1].imshow(label_slice, alpha=0.5, cmap=cmap)
    axes[1].set_title("Ground Truth Label")

    # Overlay prediction results
    axes[2].imshow(ct_slice, cmap="gray", alpha=0.8)
    axes[2].imshow(pred_slice, alpha=0.5, cmap=cmap)
    axes[2].set_title("Predicted Segmentation")

    # Turn off axes
    for ax in axes:
        ax.axis("on")

    # Place legend in the bottom right corner of the entire figure
    fig.legend(
        handles=legend_elements,
        loc="lower right",  # Position: bottom right of entire figure
        bbox_to_anchor=(0.95, 0.05),  # Fine-tune position
        fontsize=10,
        title="Class",  # Legend title
        title_fontsize=11
    )

    # Adjust layout to make space for legend
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # Reserve bottom right space for legend

    # Save the figure
    output_path = os.path.join(output_dir, f"{patient_id}_segmentation_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Segmentation visualization result saved to: {output_path}")

    plt.show()


# Example usage
if __name__ == "__main__":
    # Update these paths to your actual file locations
    plot_segmentation_comparison(
        ct_path="path/to/ct/image.nii.gz",
        label_path="path/to/ground/truth/labels.nii.gz",
        pred_path="path/to/predicted/segmentation.nii.gz",
        output_dir="path/to/output/directory",
        slice_direction="z",
        slice_idx=30
    )