import sys
import os
from os.path import join
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc, \
    f1_score, roc_curve, accuracy_score, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
from imblearn.over_sampling import SMOTE
import monai
from skimage.transform import resize
from monai.transforms import (
    Compose, EnsureChannelFirstd, ScaleIntensityRanged,
    RandRotated, RandZoomd, ToTensord, EnsureTyped,
    RandFlipd, RandGaussianNoised, RandAdjustContrastd,
    RandGaussianSmoothd
)
import cv2
import gc
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
import scipy.stats as stats


def set_seed(seed=36):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# Fix nnUNet import
sys.path.append("path/to/nnUNet")
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# 1. Configuration Parameters
class Config:
    # Data paths
    train_image_dir = "path/to/train/images"
    train_seg_dir = "path/to/train/segmentations"
    train_clinical_csv = "path/to/train/clinical_features.csv"

    test_image_dir = "path/to/test/images"
    test_seg_dir = "path/to/test/segmentations"
    test_clinical_csv = "path/to/test/clinical_features.csv"

    # nnU-Net pretrained model path
    nnunet_best_model_path = "path/to/nnunet/best/model.pth"

    # Training parameters
    batch_size = 2
    num_workers = 2
    learning_rate = 1e-4
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = "path/to/save/model.pth"

    # Visualization parameters
    viz_save_dir = "path/to/visualizations"
    os.makedirs(viz_save_dir, exist_ok=True)

    # Clinical feature configuration
    categorical_features = ['Eye', 'Sex', 'Previous_treatment_history', 'Systemic_medical_history']
    numerical_features = ['Age', 'Duration_of_symptoms','Severity_of_symptoms']
    all_clinical_features = categorical_features + numerical_features


# Clinical feature mapping and type definitions
clinical_name_map = {
    'clinical_0': 'Eye',
    'clinical_1': 'Sex',
    'clinical_2': 'Age',
    'clinical_3': 'Duration_of_symptoms',
    'clinical_4': 'Severity_of_symptoms',
    'clinical_5': 'Previous_treatment_history',
    'clinical_6': 'Systemic_medical_history'
}

# Define clinical feature types
clinical_feature_types = {
    'Eye': 'categorical',
    'Sex': 'categorical',
    'Age': 'numerical',
    'Duration_of_symptoms': 'numerical',
    'Severity_of_symptoms': 'numerical',
    'Previous_treatment_history': 'categorical',
    'Systemic_medical_history': 'categorical'
}


# 2. Implement Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevent nans
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# 3. Data Loader
class CTDataset(Dataset):
    def __init__(self, image_dir, seg_dir, clinical_csv, transform=None, is_train=True):
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.clinical_csv = clinical_csv
        self.transform = transform
        self.is_train = is_train

        # Case list
        seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.nii.gz')]
        self.case_ids = [f.split('.')[0] for f in seg_files]

        # Clinical data
        self.clinical_data, self.labels = self._load_clinical_data(clinical_csv)

        # Ensure valid_case_ids only contains cases present in clinical data
        self.valid_case_ids = [cid for cid in self.case_ids if cid in self.clinical_data]

        # Ensure labels correspond to valid cases
        self.labels = [self.clinical_data[cid].get('difficulty', 0) for cid in self.valid_case_ids]

        # Clinical data preprocessing
        self.scaler, self.encoder = self._fit_preprocessors()

        # Add debug information
        print(f"\n{'=' * 50}")
        print(f"Dataset: {'Train' if is_train else 'Test'}")
        print(f"Valid cases: {len(self.valid_case_ids)}")
        print(f"Class distribution: {np.bincount(self.labels) if len(self.labels) > 0 else 'N/A'}")

    def _load_clinical_data(self, path):
        if not path or not os.path.exists(path):
            print(f"Clinical CSV not found at {path}")
            return {}, []

        print(f"Loading clinical data from {path}")
        df = pd.read_csv(path)

        if 'difficulty' not in df.columns:
            print("Warning: 'difficulty' column not found in clinical data!")
            return {}, []

        if 'patient_id' not in df.columns:
            if 'case_id' in df.columns:
                print("Renaming 'case_id' to 'patient_id'")
                df.rename(columns={'case_id': 'patient_id'}, inplace=True)
            else:
                print("Error: No patient/case ID column found!")
                return {}, []

        # Extract labels
        labels = df['difficulty'].values

        # Extract features
        clinical_data = {}
        for _, row in df.iterrows():
            case_id = row['patient_id']
            features = {}
            for feat in Config.all_clinical_features:
                if feat in row:
                    features[feat] = row[feat]
                else:
                    print(f"Warning: Feature {feat} not found for case {case_id}")
                    features[feat] = 0  # default value

            # Ensure label is also stored in clinical data
            features['difficulty'] = row['difficulty']

            clinical_data[case_id] = features

        print(f"Loaded clinical data for {len(clinical_data)} cases")

        # Only print distribution if there is label data
        if len(labels) > 0:
            print(f"Label distribution in clinical data: {np.bincount(labels)}")
        else:
            print("No labels found in clinical data")

        return clinical_data, labels

    def _fit_preprocessors(self):
        # Collect feature data for all cases
        categorical_data = []
        numerical_data = []

        for cid in self.valid_case_ids:
            clin_data = self.clinical_data[cid]
            cat_feats = [clin_data[feat] for feat in Config.categorical_features]
            num_feats = [clin_data[feat] for feat in Config.numerical_features]

            categorical_data.append(cat_feats)
            numerical_data.append(num_feats)

        # Train scaler and encoder
        scaler = StandardScaler()
        if numerical_data and len(numerical_data) > 0:
            scaler.fit(numerical_data)

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        if categorical_data and len(categorical_data) > 0:
            encoder.fit(categorical_data)

        return scaler, encoder

    def __len__(self):
        return len(self.valid_case_ids)

    def __getitem__(self, idx):
        case_id = self.valid_case_ids[idx]

        # Load CT
        ct_path = os.path.join(self.image_dir, f"{case_id}_0000.nii.gz")
        if not os.path.exists(ct_path):
            ct_path = os.path.join(self.image_dir, f"{case_id}.nii.gz")
            if not os.path.exists(ct_path):
                raise FileNotFoundError(f"CT file not found for case {case_id} in {self.image_dir}")

        # Load segmentation mask
        seg_path = os.path.join(self.seg_dir, f"{case_id}.nii.gz")
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Segmentation file not found for case {case_id} in {self.seg_dir}")

        ct_img = sitk.ReadImage(ct_path)
        ct_array = sitk.GetArrayFromImage(ct_img)

        # Ensure CT is 3D or 4D (with channel)
        if ct_array.ndim == 3:  # No channel dimension [D, H, W]
            ct_array = ct_array[np.newaxis, ...]  # Add channel dimension [C, D, H, W]
        elif ct_array.ndim == 4:  # With channel dimension [C, D, H, W]
            if ct_array.shape[0] > 1:
                ct_array = ct_array[0:1]  # If multi-channel, take first channel

        seg_img = sitk.ReadImage(seg_path)
        seg_array = sitk.GetArrayFromImage(seg_img)

        # Ensure segmentation mask has channel dimension [C, D, H, W]
        if seg_array.ndim == 3:  # No channel dimension [D, H, W]
            seg_array = seg_array[np.newaxis, ...]  # Add channel dimension [C, D, H, W]
        elif seg_array.ndim == 4:  # With channel dimension [C, D, H, W]
            if seg_array.shape[0] > 1:
                seg_array = seg_array[0:1]  # If multi-channel, take first channel

        # Clinical data
        clin_data = self.clinical_data[case_id]
        label = int(clin_data.get('difficulty', 0))  # Get label from clinical data

        # Separate categorical and numerical features
        cat_feats = [clin_data[feat] for feat in Config.categorical_features]
        num_feats = [clin_data[feat] for feat in Config.numerical_features]

        # Preprocess features
        if self.scaler and num_feats and len(num_feats) > 0:
            num_feats = self.scaler.transform([num_feats])[0]
        else:
            num_feats = np.array(num_feats, dtype=np.float32)

        if self.encoder and cat_feats and len(cat_feats) > 0:
            cat_feats = self.encoder.transform([cat_feats])[0]
        else:
            cat_feats = np.array(cat_feats, dtype=np.float32)

        # Merge features
        clin_features = np.concatenate([cat_feats, num_feats])

        # Create data dictionary
        data_dict = {
            'ct': ct_array,
            'seg': seg_array,
            'clinical': clin_features,
            'label': label,
            'case_id': case_id
        }

        # Apply transformations
        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict


# 4. Model Definition
class AttentionWeightedPooling(nn.Module):
    def forward(self, features, mask):
        # Ensure mask has correct dimensions [N, C, D, H, W]
        if mask.dim() == 4:
            mask = mask.unsqueeze(1)  # Add channel dimension [N, 1, D, H, W]

        # Check if spatial dimensions match
        if mask.shape[2:] != features.shape[2:]:
            target_size = tuple(features.shape[2:])
            mask = F.interpolate(
                mask,
                size=target_size,
                mode='trilinear',
                align_corners=False
            )

        # Calculate weighted features
        weighted_features = features * mask

        # Sum along spatial dimensions (D, H, W)
        sum_features = torch.sum(weighted_features, dim=(2, 3, 4))
        sum_mask = torch.sum(mask, dim=(2, 3, 4)) + 1e-8

        # Calculate weighted average
        pooled = sum_features / sum_mask
        return pooled


class SurgicalDifficultyClassifier(nn.Module):
    def __init__(self, nnunet_ckpt_path, clin_feat_dim):
        super().__init__()
        self.encoder = self._load_nnunet_encoder(nnunet_ckpt_path)
        self.encoder = self.encoder.to(Config.device)
        self.attn_pooling = AttentionWeightedPooling()

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Dynamically determine feature dimension
        self.encoder_feat_dim = self._get_encoder_feature_dim()
        print(f"Encoder feature dimension: {self.encoder_feat_dim}")
        print(f"Clinical feature dimension: {clin_feat_dim}")

        # Add feature interaction layer
        self.feature_interaction = nn.Sequential(
            nn.Linear(self.encoder_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, clin_feat_dim)
        )

        # Calculate final dimension of fused features
        self.fused_feat_dim = self.encoder_feat_dim + clin_feat_dim + clin_feat_dim
        print(f"Fused feature dimension: {self.fused_feat_dim}")

        # Improved classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_feat_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _load_nnunet_encoder(self, nnunet_ckpt_path):
        # Initialize predictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=Config.device,
            verbose=False
        )

        # Get model directory and checkpoint name
        model_dir = os.path.dirname(nnunet_ckpt_path)
        checkpoint_name = os.path.basename(nnunet_ckpt_path)

        print(f"Loading model from directory: {model_dir}")
        print(f"Using checkpoint: {checkpoint_name}")

        # Check model directory structure
        if os.path.basename(model_dir) == "fold_all":
            actual_model_dir = os.path.dirname(model_dir)
            fold_name = "all"
            print(f"Adjusting model directory to parent: {actual_model_dir}")
        else:
            actual_model_dir = model_dir
            fold_name = "all"

        # Ensure checkpoint file exists
        ckpt_path = os.path.join(model_dir, checkpoint_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        print(f"Actual checkpoint path: {ckpt_path}")

        try:
            # Initialize model
            predictor.initialize_from_trained_model_folder(
                actual_model_dir,
                use_folds=(fold_name,),
                checkpoint_name=checkpoint_name
            )
        except FileNotFoundError as e:
            print(f"Error loading model: {str(e)}")
            print(f"Model directory: {actual_model_dir}")
            print(f"Fold name: {fold_name}")
            print(f"Checkpoint name: {checkpoint_name}")
            raise

        if predictor.network is None:
            raise RuntimeError("Failed to initialize nnUNet network")

        # Return encoder part
        return predictor.network.encoder

    def _get_encoder_feature_dim(self):
        # Use dummy input to determine encoder output feature dimension
        test_input = torch.randn(1, 1, 64, 128, 128).to(Config.device)
        with torch.no_grad():
            test_output = self.encoder(test_input)
            # Handle possible list output
            if isinstance(test_output, (list, tuple)):
                test_output = test_output[-1]  # Take last feature map
            return test_output.shape[1]  # Return channel number

    def forward(self, ct, seg, clinical):
        # Ensure input data is on the correct device
        ct = ct.to(Config.device)
        seg = seg.to(Config.device)
        clinical = clinical.to(Config.device)

        # Process encoder output (could be list or tensor)
        encoder_output = self.encoder(ct)
        if isinstance(encoder_output, (list, tuple)):
            # If list, take the last feature map (usually highest level features)
            features = encoder_output[-1]
        else:
            features = encoder_output

        pooled = self.attn_pooling(features, seg)

        # Feature interaction
        img_features = self.feature_interaction(pooled)
        interacted = img_features * clinical

        fused = torch.cat([pooled, clinical, interacted], dim=1)
        logits = self.classifier(fused)
        return logits.squeeze(-1)


# 5. Training and Evaluation
def validate_model(model, loader, criterion, threshold=0.49):
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    total_loss = 0.0
    all_case_ids = []

    with torch.no_grad():
        for batch in loader:
            ct = batch['ct'].to(Config.device).float()
            seg = batch['seg'].to(Config.device).float()
            clinical = batch['clinical'].to(Config.device).float()
            labels = batch['label'].to(Config.device).float().view(-1)

            logits = model(ct, seg, clinical)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)

            all_probs.extend(probs)
            all_labels.extend(batch['label'].cpu().numpy())
            all_preds.extend(preds)
            all_case_ids.extend(batch['case_id'])

    # Calculate metrics
    auc_score = roc_auc_score(all_labels, all_probs)
    avg_loss = total_loss / len(loader)

    # Add more evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)

    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    report = classification_report(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}, Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"MCC: {mcc:.4f}, Cohen's Kappa: {kappa:.4f}")
    print(f"ROC AUC: {auc_score:.4f}, PR AUC: {pr_auc:.4f}, F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(report)

    return (auc_score, avg_loss, report, all_probs, all_labels, all_case_ids,
            accuracy, balanced_accuracy, mcc, kappa, pr_auc, f1, cm)


# 6. Visualization Functions
def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve to {save_path}")


def plot_pr_curve(y_true, y_probs, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved PR curve to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_calibration_curve(y_true, y_probs, save_path):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_probs, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title('Calibration Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved calibration curve to {save_path}")


def plot_waterfall(y_true, y_probs, case_ids, save_path):
    # Sort by prediction probability
    sorted_indices = np.argsort(y_probs)
    sorted_probs = np.array(y_probs)[sorted_indices]
    sorted_labels = np.array(y_true)[sorted_indices]
    sorted_case_ids = np.array(case_ids)[sorted_indices]

    # Create color mapping
    colors = []
    for i in range(len(sorted_probs)):
        if sorted_labels[i] == 0 and sorted_probs[i] < 0.49:
            colors.append('green')  # Correctly predicted easy cases
        elif sorted_labels[i] == 1 and sorted_probs[i] >= 0.49:
            colors.append('blue')  # Correctly predicted hard cases
        else:
            colors.append('red')  # Incorrectly predicted cases

    # Create waterfall plot
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(sorted_probs)), sorted_probs, color=colors)

    # Add baseline
    plt.axhline(y=0.49, color='gray', linestyle='--', alpha=0.7)

    # Set labels and title
    plt.xlabel('Patient Index (Sorted by Prediction Probability)')
    plt.ylabel('Prediction Probability')
    plt.title('Waterfall Plot of Surgical Difficulty Predictions')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Easy - Correct Prediction'),
        Patch(facecolor='blue', label='Hard - Correct Prediction'),
        Patch(facecolor='red', label='Incorrect Prediction')
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.xticks(range(len(sorted_probs)), sorted_case_ids, rotation=90)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved waterfall plot to {save_path}")


# 7. Attention Visualization
def visualize_attn(model, loader, num_samples=None):
    model.eval()
    os.makedirs(Config.viz_save_dir, exist_ok=True)

    # List to store Attention Value data
    attention_value_data = []

    # If no sample count specified, process all samples
    if num_samples is None:
        num_samples = len(loader.dataset)

    # Create batch size 1 data loader for visualization
    viz_loader = DataLoader(
        loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.num_workers
    )

    processed = 0
    for batch in viz_loader:
        if processed >= num_samples:
            break

        ct = batch['ct'].to(Config.device).float()
        seg = batch['seg'].to(Config.device).float()
        clinical = batch['clinical'].to(Config.device).float()
        case_id = batch['case_id'][0]

        with torch.no_grad():
            # Get encoder features
            encoder_output = model.encoder(ct)

            # Handle possible list output
            if isinstance(encoder_output, (list, tuple)):
                # If list, take the last feature map (highest level features)
                features = encoder_output[-1]
            else:
                features = encoder_output

            # Ensure mask has correct dimensions [N, C, D, H, W]
            if seg.dim() == 4:
                seg = seg.unsqueeze(1)  # Add channel dimension [N, 1, D, H, W]

            # Check if spatial dimensions match
            if seg.shape[2:] != features.shape[2:]:
                target_size = tuple(features.shape[2:])
                seg = F.interpolate(
                    seg,
                    size=target_size,
                    mode='trilinear',
                    align_corners=False
                )

            # Calculate weighted features
            weighted_features = features * seg

            # Calculate Attention Value
            attn_map = torch.mean(weighted_features, dim=1)  # Average along channel dimension

            # Upsample Attention Value to original resolution
            attn_map_upsampled = F.interpolate(
                attn_map.unsqueeze(1),
                size=ct.shape[2:],  # Original CT image size
                mode='trilinear',
                align_corners=False
            ).squeeze(1)[0].cpu().numpy()

            # Collect Attention Value statistics
            attn_flat = attn_map_upsampled.flatten()
            attn_stats = {
                'case_id': case_id,
                'mean': np.mean(attn_flat),
                'std': np.std(attn_flat),
                'min': np.min(attn_flat),
                'max': np.max(attn_flat),
                'median': np.median(attn_flat),
                'q1': np.percentile(attn_flat, 25),
                'q3': np.percentile(attn_flat, 75),
                'nonzero_count': np.count_nonzero(attn_flat),
                'total_voxels': len(attn_flat),
                'nonzero_ratio': np.count_nonzero(attn_flat) / len(attn_flat)
            }

            # Add to data list
            attention_value_data.append(attn_stats)

        # Take middle slice for visualization
        slice_idx = ct.shape[2] // 2
        ct_slice = ct[0, 0, slice_idx].cpu().numpy()  # [H, W]
        attn_slice = attn_map_upsampled[slice_idx]  # [H, W]

        # Normalize attention map
        attn_slice_normalized = (attn_slice - np.min(attn_slice)) / (np.max(attn_slice) - np.min(attn_slice) + 1e-8)

        # Create figure with two subplots
        plt.figure(figsize=(12, 6))

        # First subplot: CT slice
        plt.subplot(1, 2, 1)
        plt.imshow(ct_slice, cmap='gray')
        plt.title(f'CT Slice - {case_id}')
        plt.axis('off')

        # Second subplot: CT slice with attention heatmap overlay (using normalized attention map)
        plt.subplot(1, 2, 2)
        plt.imshow(ct_slice, cmap='gray')
        im = plt.imshow(attn_slice_normalized, cmap='hot', alpha=0.5)
        plt.title('CT with Attention Overlay (Normalized)')
        plt.axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Attention Value', rotation=270, labelpad=15)

        plt.tight_layout()
        plt.savefig(os.path.join(Config.viz_save_dir, f'attention_{case_id}.png'), dpi=300)
        plt.close()

        processed += 1
        print(f"Processed {processed}/{min(num_samples, len(loader.dataset))} cases")

    # Save Attention Value statistics to CSV
    if attention_value_data:
        df = pd.DataFrame(attention_value_data)
        csv_path = os.path.join(Config.viz_save_dir, "attention_value_statistics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved Attention Value statistics to {csv_path}")

        # Create detailed Attention Value distribution CSV
        detailed_data = []
        for batch in viz_loader:
            if batch['case_id'][0] in df['case_id'].values:
                case_id = batch['case_id'][0]
                ct = batch['ct'].to(Config.device).float()

                with torch.no_grad():
                    encoder_output = model.encoder(ct)
                    if isinstance(encoder_output, (list, tuple)):
                        features = encoder_output[-1]
                    else:
                        features = encoder_output

                    seg = batch['seg'].to(Config.device).float()
                    if seg.dim() == 4:
                        seg = seg.unsqueeze(1)

                    if seg.shape[2:] != features.shape[2:]:
                        target_size = tuple(features.shape[2:])
                        seg = F.interpolate(
                            seg,
                            size=target_size,
                            mode='trilinear',
                            align_corners=False
                        )

                    weighted_features = features * seg
                    attn_map = torch.mean(weighted_features, dim=1)

                    # Get Attention Value for each region
                    for region_idx in range(attn_map.shape[1]):
                        region_attn = attn_map[0, region_idx].cpu().numpy()
                        detailed_data.append({
                            'case_id': case_id,
                            'region': f'region_{region_idx}',
                            'mean_attention': np.mean(region_attn),
                            'max_attention': np.max(region_attn),
                            'min_attention': np.min(region_attn)
                        })

        detailed_df = pd.DataFrame(detailed_data)
        detailed_csv_path = os.path.join(Config.viz_save_dir, "attention_value_detailed.csv")
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"Saved detailed Attention Value data to {detailed_csv_path}")

    print(f"Saved attention visualizations to {Config.viz_save_dir}")


# 8. Clinical Feature Importance Analysis
def analyze_clinical_feature_importance(model, loader):
    """
    Analyze clinical feature importance
    """
    model.eval()
    all_importances = []
    all_case_ids = []

    for batch in loader:
        ct = batch['ct'].to(Config.device).float()
        seg = batch['seg'].to(Config.device).float()
        clinical = batch['clinical'].to(Config.device).float()
        case_ids = batch['case_id']

        # Ensure inputs require gradients
        clinical.requires_grad = True

        # Forward pass
        output = model(ct, seg, clinical)

        # Zero gradients
        model.zero_grad()

        # For binary classification, output is a scalar, not one-hot encoded
        # Use output value directly as gradient
        one_hot_output = torch.ones_like(output)

        try:
            # Backward pass
            output.backward(gradient=one_hot_output, retain_graph=True)

            # Get gradients for clinical features
            clinical_grad = clinical.grad.detach().cpu().numpy()

            # Calculate clinical feature importance
            clinical_importance = np.abs(clinical_grad)
            all_importances.extend(clinical_importance)
            all_case_ids.extend(case_ids)
        except RuntimeError as e:
            print(f"Error during backward pass: {e}")
            print("Skipping this batch for feature importance analysis")
            continue

    # Calculate average importance
    if all_importances:
        avg_importance = np.mean(all_importances, axis=0)

        # Create feature name list - only use known clinical features
        feature_names = []
        for i in range(len(avg_importance)):
            if i < 7:  # Only process first 7 features (known clinical features)
                if f'clinical_{i}' in clinical_name_map:
                    feature_names.append(clinical_name_map[f'clinical_{i}'])
                else:
                    feature_names.append(f'Feature_{i}')
            # Ignore 7th and later unknown features

        # Only keep importance for first 7 features
        avg_importance = avg_importance[:7]
        feature_names = feature_names[:7]

        # Sort by importance
        sorted_indices = np.argsort(avg_importance)[::-1]
        sorted_importance = avg_importance[sorted_indices]
        sorted_feature_names = [feature_names[i] for i in sorted_indices]

        # Determine feature types
        feature_types = [clinical_feature_types.get(name, 'unknown') for name in sorted_feature_names]

        # Assign different colors for different feature types
        colors = []
        for feat_type in feature_types:
            if feat_type == 'categorical':
                colors.append('skyblue')
            elif feat_type == 'numerical':
                colors.append('lightcoral')
            else:
                colors.append('lightgray')

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(sorted_importance)), sorted_importance, color=colors)
        plt.xlabel('Mean Gradient Value (average impact on surgical difficulty)')
        plt.ylabel('Clinical Features')
        plt.title('Clinical Feature Importance Analysis')
        plt.yticks(range(len(sorted_importance)), sorted_feature_names)

        # Add value labels
        for i, v in enumerate(sorted_importance):
            plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', label='Categorical Features'),
            Patch(facecolor='lightcoral', label='Numerical Features')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(Config.viz_save_dir, "clinical_feature_importance.png"), dpi=300)
        plt.close()

        # Save feature importance data
        importance_df = pd.DataFrame({
            'feature': sorted_feature_names,
            'importance': sorted_importance,
            'feature_type': feature_types
        })
        importance_df.to_csv(os.path.join(Config.viz_save_dir, "clinical_feature_importance.csv"), index=False)

        print("Saved clinical feature importance analysis")

        # Group analysis by feature type
        categorical_importance = importance_df[importance_df['feature_type'] == 'categorical']
        numerical_importance = importance_df[importance_df['feature_type'] == 'numerical']

        print("\nCategorical features importance:")
        for _, row in categorical_importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")

        print("\nNumerical features importance:")
        for _, row in numerical_importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")

        return sorted_importance, sorted_feature_names
    else:
        print("No clinical feature importance data collected")
        return None, None


# 9. Data Transformations
def get_transforms():
    train_transforms = Compose([
        EnsureChannelFirstd(keys=['ct', 'seg'], channel_dim=0),
        ScaleIntensityRanged(keys=['ct'], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        RandRotated(keys=['ct', 'seg'], range_x=0.4, range_y=0.4, range_z=0.4, prob=0.8, keep_size=True),
        RandFlipd(keys=['ct', 'seg'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['ct', 'seg'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['ct', 'seg'], prob=0.5, spatial_axis=2),
        RandZoomd(keys=['ct', 'seg'], prob=0.5, min_zoom=0.8, max_zoom=1.2, keep_size=True),
        RandGaussianNoised(keys=['ct'], prob=0.5, mean=0.0, std=0.15),
        RandAdjustContrastd(keys=['ct'], prob=0.5, gamma=(0.7, 1.5)),
        RandGaussianSmoothd(keys=['ct'], prob=0.3, sigma_x=(0.5, 1.5)),
        ToTensord(keys=['ct', 'seg', 'clinical', 'label']),
        EnsureTyped(keys=['ct', 'seg', 'clinical', 'label'])
    ])

    val_transforms = Compose([
        EnsureChannelFirstd(keys=['ct', 'seg'], channel_dim=0),
        ScaleIntensityRanged(
            keys=['ct'],
            a_min=-1000, a_max=1000,
            b_min=0.0, b_max=1.0, clip=True
        ),
        ToTensord(keys=['ct', 'seg', 'clinical', 'label']),
        EnsureTyped(keys=['ct', 'seg', 'clinical', 'label'])
    ])

    return train_transforms, val_transforms


# 10. Main Pipeline - Directly load pre-trained model for evaluation
def main():
    set_seed(36)

    # Only use validation transforms
    _, val_transforms = get_transforms()

    # Load test dataset
    test_dataset = CTDataset(
        Config.test_image_dir,
        Config.test_seg_dir,
        Config.test_clinical_csv,
        transform=val_transforms,
        is_train=False
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Get clinical feature dimension
    sample = test_dataset[0]
    clin_feat_dim = len(sample['clinical'])
    print(f"Clinical feature dimension: {clin_feat_dim}")

    # Create model
    model = SurgicalDifficultyClassifier(Config.nnunet_best_model_path, clin_feat_dim).to(Config.device)

    # Load pre-trained model
    if os.path.exists(Config.model_save_path):
        print(f"Loading pre-trained model from {Config.model_save_path}")
        model.load_state_dict(torch.load(Config.model_save_path, map_location=Config.device))
        print("Model loaded successfully!")
    else:
        print(f"Error: Pre-trained model not found at {Config.model_save_path}")
        return

    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers
    )

    # Evaluate on test set
    print("Evaluating on test set...")
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    test_results = validate_model(model, test_loader, criterion, threshold=0.487)
    (test_auc, test_loss, test_report, test_probs, test_labels, test_case_ids,
     accuracy, balanced_accuracy, mcc, kappa, pr_auc, f1, cm) = test_results

    print(f"\n{'=' * 50}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(test_report)
    print(f"{'=' * 50}")

    # Save detailed results
    results_df = pd.DataFrame({
        'case_id': test_case_ids,
        'true_label': test_labels,
        'pred_prob': test_probs,
        'pred_label': (np.array(test_probs) > 0.49).astype(int)
    })
    results_df.to_csv(os.path.join(Config.viz_save_dir, "test_predictions.csv"), index=False)

    # Save performance metrics
    metrics_df = pd.DataFrame({
        'Metric': ['AUC', 'Accuracy', 'Balanced Accuracy', 'MCC', 'Cohen\'s Kappa', 'PR AUC', 'F1 Score'],
        'Value': [test_auc, accuracy, balanced_accuracy, mcc, kappa, pr_auc, f1]
    })
    metrics_df.to_csv(os.path.join(Config.viz_save_dir, "performance_metrics.csv"), index=False)

    # Generate visualization charts
    print("Generating performance visualizations...")

    # ROC curve
    plot_roc_curve(test_labels, test_probs, os.path.join(Config.viz_save_dir, "roc_curve.png"))

    # PR curve
    plot_pr_curve(test_labels, test_probs, os.path.join(Config.viz_save_dir, "pr_curve.png"))

    # Confusion matrix
    plot_confusion_matrix(test_labels, (np.array(test_probs) > 0.49).astype(int),
                          os.path.join(Config.viz_save_dir, "confusion_matrix.png"))

    # Calibration curve
    plot_calibration_curve(test_labels, test_probs, os.path.join(Config.viz_save_dir, "calibration_curve.png"))

    # Waterfall plot
    plot_waterfall(test_labels, test_probs, test_case_ids, os.path.join(Config.viz_save_dir, "waterfall_plot.png"))

    # Analyze failure cases
    print("Analyzing failure cases...")
    failure_cases = []
    for i in range(len(test_labels)):
        if (test_probs[i] > 0.49) != test_labels[i]:
            failure_cases.append({
                'case_id': results_df.iloc[i]['case_id'],
                'true_label': test_labels[i],
                'pred_prob': test_probs[i],
                'pred_label': (test_probs[i] > 0.49).astype(int)
            })

    if failure_cases:
        print(f"Found {len(failure_cases)} failure cases")
        failure_df = pd.DataFrame(failure_cases)
        failure_df.to_csv(os.path.join(Config.viz_save_dir, "failure_analysis.csv"), index=False)
    else:
        print("No failure cases found!")

    # Generate attention visualizations - process all 30 cases
    print("Generating attention visualizations for all test cases...")
    visualize_attn(model, test_loader, num_samples=None)  # Set to None to process all cases

    # Analyze clinical feature importance
    print("Analyzing clinical feature importance...")
    analyze_clinical_feature_importance(model, test_loader)

    print("\nAll operations completed successfully!")


if __name__ == "__main__":
    main()