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
    f1_score, roc_curve
from imblearn.over_sampling import SMOTE
import monai
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
    # Data paths - Update these paths according to your setup
    train_image_dir = "path/to/train/images"
    train_seg_dir = "path/to/train/segmentations"
    train_clinical_csv = "path/to/train/clinical_data.csv"

    test_image_dir = "path/to/test/images"
    test_seg_dir = "path/to/test/segmentations"
    test_clinical_csv = "path/to/test/clinical_data.csv"

    # nnU-Net pretrained model path
    nnunet_best_model_path = "path/to/nnunet/best/model.pth"

    # Training parameters
    batch_size = 2
    num_workers = 2
    learning_rate = 1e-4
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = "surgical_difficulty_classifier.pth"

    # Visualization parameters
    viz_save_dir = "path/to/visualizations"
    os.makedirs(viz_save_dir, exist_ok=True)

    # Clinical feature configuration
    categorical_features = ['Eye', 'Sex', 'Previous_treatment_history', 'Systemic_medical_history']
    numerical_features = ['Age', 'Duration_of_symptoms', 'Severity_of_symptoms']
    all_clinical_features = categorical_features + numerical_features


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

        # Register hooks for Grad-CAM
        self.gradients = None
        self.activations = None

    def register_hooks(self):
        # Find the last convolutional layer in the encoder
        target_layer = self.get_target_layer()
        if target_layer:
            target_layer.register_forward_hook(self.forward_hook)
            target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output
        return None

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        return None

    def get_target_layer(self):
        """Find the last convolutional layer in the encoder as target for Grad-CAM"""
        # Recursively find all convolutional layers
        conv_layers = []
        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Conv3d):
                conv_layers.append(module)

        # Return the last convolutional layer
        if conv_layers:
            return conv_layers[-1]
        else:
            # If no convolutional layers found, try alternative methods
            # Based on nnUNet encoder structure, try to access possible last layer
            if hasattr(self.encoder, 'output_block'):
                return self.encoder.output_block.conv
            elif hasattr(self.encoder, 'blocks') and len(self.encoder.blocks) > 0:
                return self.encoder.blocks[-1].conv
            else:
                print("Warning: Could not find suitable target layer for Grad-CAM")
                return None

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

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations


# 5. Grad-CAM Implementation
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        # Store additional inputs
        self.seg = None
        self.clinical = None

    def set_additional_inputs(self, seg, clinical):
        self.seg = seg
        self.clinical = clinical

    def forward(self, x):
        # Use stored additional inputs
        return self.model(x, self.seg, self.clinical)

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        return None

    def forward_hook(self, module, input, output):
        self.activations = output
        return None

    def generate_cam(self, input_image, seg, clinical, target_layer):  # Added seg and clinical parameters
        # Set additional inputs
        self.set_additional_inputs(seg, clinical)

        # Register hooks
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

        # Forward pass
        output = self.forward(input_image)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        output.backward(torch.ones_like(output))

        # Get activations and gradients
        activations = self.activations.detach()
        gradients = self.gradients.detach()

        # Calculate weights
        weights = torch.mean(gradients, dim=(2, 3, 4))

        # Calculate CAM
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i]

        # ReLU operation
        cam = torch.clamp(cam, min=0)

        # Normalize
        cam -= torch.min(cam)
        cam /= torch.max(cam) if torch.max(cam) != 0 else 1

        return cam.cpu().numpy()


# 6. Training and Evaluation
def train_model(train_loader, val_loader, clin_feat_dim):
    # Calculate class weights
    all_labels = []
    for batch in train_loader:
        all_labels.append(batch['label'].long())
    all_labels = torch.cat(all_labels)

    # Check number of classes
    unique_labels = torch.unique(all_labels)
    print(f"Unique labels in training data: {unique_labels.tolist()}")

    # Calculate number of samples per class
    class_counts = torch.bincount(all_labels)
    print(f"Class counts: {class_counts.tolist()}")

    # If there are missing classes, ensure class_counts contains all possible classes
    if len(class_counts) < 2:
        # Extend class_counts to ensure there are two classes
        full_class_counts = torch.zeros(2, dtype=torch.long)
        for i in range(len(class_counts)):
            full_class_counts[i] = class_counts[i]
        class_counts = full_class_counts

    total_samples = class_counts.sum().item()

    # Fix class weight calculation to avoid division by zero
    class_weights = []
    for count in class_counts:
        if count > 0:
            class_weights.append(total_samples / (len(class_counts) * count))
        else:
            class_weights.append(0.0)  # If a class has no samples, set weight to 0

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights: {class_weights.tolist()}")

    # Use Focal Loss
    # Ensure alpha value is valid
    if len(class_weights) > 1 and class_weights[1] > 0:
        alpha = class_weights[1].item()
    else:
        alpha = 0.75  # Default value
        print("Warning: Using default alpha value for Focal Loss")

    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    print(f"Using Focal Loss with alpha={alpha:.4f}")

    model = SurgicalDifficultyClassifier(Config.nnunet_best_model_path, clin_feat_dim).to(Config.device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=1e-5
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.num_epochs,
        eta_min=1e-6
    )

    # Early stopping mechanism
    best_auc = 0.0
    best_loss = float('inf')
    patience = 20
    patience_counter = 0

    train_history = {'loss': [], 'auc': []}
    val_history = {'loss': [], 'auc': []}

    for epoch in range(Config.num_epochs):
        model.train()
        total_loss = 0.0
        all_train_labels = []
        all_train_probs = []

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{Config.num_epochs}', leave=False):
            ct = batch['ct'].to(Config.device).float()
            seg = batch['seg'].to(Config.device).float()
            clinical = batch['clinical'].to(Config.device).float()
            label = batch['label'].to(Config.device).float().view(-1)

            optimizer.zero_grad()
            logits = model(ct, seg, clinical)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Collect training predictions
            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu().numpy()
                all_train_probs.extend(probs)
                all_train_labels.extend(batch['label'].cpu().numpy())

        # Calculate training AUC
        train_auc = roc_auc_score(all_train_labels, all_train_probs)
        train_history['loss'].append(total_loss / len(train_loader))
        train_history['auc'].append(train_auc)

        # Validation
        val_results = validate_model(model, val_loader, criterion)
        val_auc, val_loss = val_results[0], val_results[1]
        val_history['loss'].append(val_loss)
        val_history['auc'].append(val_auc)

        # Early stopping decision
        if val_auc > best_auc or total_loss < best_loss:
            if val_auc > best_auc:
                best_auc = val_auc
            if total_loss < best_loss:
                best_loss = total_loss
            torch.save(model.state_dict(), Config.model_save_path)
            print(f"Saved best model with AUC: {best_auc:.4f}, Loss: {best_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Update learning rate
        scheduler.step()

        print(f"Epoch {epoch + 1}/{Config.num_epochs}: "
              f"Train Loss: {total_loss / len(train_loader):.4f}, Train AUC: {train_auc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Plot training curves
    plot_training_curves(train_history, val_history)

    return model


def validate_model(model, loader, criterion, threshold=0.5):
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    total_loss = 0.0

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

    # Calculate metrics
    auc_score = roc_auc_score(all_labels, all_probs)
    avg_loss = total_loss / len(loader)

    # Add more evaluation metrics
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    report = classification_report(all_labels, all_preds)

    print(f"ROC AUC: {auc_score:.4f}, PR AUC: {pr_auc:.4f}, F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(report)

    return auc_score, avg_loss, report, all_probs, all_labels


# Add training curve visualization function
def plot_training_curves(train_history, val_history):
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_history['loss'], label='Train Loss')
    plt.plot(val_history['loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # AUC curve
    plt.subplot(1, 2, 2)
    plt.plot(train_history['auc'], label='Train AUC')
    plt.plot(val_history['auc'], label='Validation AUC')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(Config.viz_save_dir, 'training_curves.png'))
    plt.close()
    print("Saved training curves to visualizations directory")


# 7. Visualization Functions
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


def plot_waterfall(y_true, y_probs, case_ids, save_path):
    # Sort by prediction probability
    sorted_indices = np.argsort(y_probs)
    sorted_probs = np.array(y_probs)[sorted_indices]
    sorted_labels = np.array(y_true)[sorted_indices]
    sorted_case_ids = np.array(case_ids)[sorted_indices]

    # Create color mapping
    colors = []
    for i in range(len(sorted_probs)):
        if sorted_labels[i] == 0 and sorted_probs[i] < 0.487:
            colors.append('green')  # Correctly predicted easy cases
        elif sorted_labels[i] == 1 and sorted_probs[i] >= 0.487:
            colors.append('blue')  # Correctly predicted hard cases
        else:
            colors.append('red')  # Incorrectly predicted cases

    # Create waterfall plot
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(sorted_probs)), sorted_probs, color=colors)

    # Add baseline
    plt.axhline(y=0.487, color='gray', linestyle='--', alpha=0.7)

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


# 8. Grad-CAM Visualization
def visualize_grad_cam(model, loader, num_samples=3):
    model.eval()

    # Get target layer
    target_layer = model.get_target_layer()
    if target_layer is None:
        print("Warning: No suitable target layer found for Grad-CAM")
        return

    # Create Grad-CAM instance
    grad_cam = GradCAM(model)

    samples = []
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        samples.append(batch)

    for i, sample in enumerate(samples):
        ct = sample['ct'][0:1].to(Config.device).float()
        seg = sample['seg'][0:1].to(Config.device).float()
        clinical = sample['clinical'][0:1].to(Config.device).float()
        case_id = sample['case_id'][0]

        # Generate CAM
        with torch.no_grad():
            output = model(ct, seg, clinical)

        # Get prediction probability
        prob = torch.sigmoid(output).item()
        pred_label = 1 if prob > 0.487 else 0
        true_label = sample['label'][0].item()

        # Generate CAM
        cam = grad_cam.generate_cam(ct, seg, clinical, target_layer)

        # Get middle slice
        ct_slice = ct[0, 0].cpu().numpy()
        slice_idx = ct_slice.shape[0] // 2

        # Resize CAM to match CT slice
        cam_resized = cv2.resize(cam, (ct_slice.shape[2], ct_slice.shape[1]))

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        # Create overlay image
        ct_slice_2d = ct_slice[slice_idx]
        ct_slice_2d = (ct_slice_2d - ct_slice_2d.min()) / (ct_slice_2d.max() - ct_slice_2d.min())
        ct_slice_2d = np.stack([ct_slice_2d] * 3, axis=-1)

        overlayed = 0.5 * ct_slice_2d + 0.5 * heatmap
        overlayed = overlayed / np.max(overlayed)

        # Plot images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(ct_slice_2d)
        plt.title(f'CT Slice - {case_id}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(overlayed)
        plt.title(f'Grad-CAM (Pred: {pred_label}, True: {true_label}, Prob: {prob:.3f})')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(Config.viz_save_dir, f'grad_cam_{case_id}.png'), dpi=300)
        plt.close()

        print(f"Saved Grad-CAM visualization for {case_id}")

    print(f"Saved Grad-CAM visualizations to {Config.viz_save_dir}")


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


# 10. Oversampling Function
def apply_oversampling(dataset, indices):
    """Apply SMOTE oversampling technique"""
    # Get original labels
    labels = [dataset[i]['label'] for i in indices]

    # Check class distribution
    class_counts = np.bincount(labels)
    print(f"Original class distribution: {class_counts}")

    # Apply SMOTE oversampling
    smote = SMOTE(random_state=42)
    # Create feature matrix (using indices as features since we only need to balance indices)
    X = np.array(indices).reshape(-1, 1)
    y = np.array(labels)

    # Apply SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Get oversampled indices
    resampled_indices = X_resampled.flatten().tolist()

    # Check class distribution after oversampling
    resampled_class_counts = np.bincount(y_resampled)
    print(f"Resampled class distribution: {resampled_class_counts}")

    return resampled_indices


# 11. Main Pipeline
def main():
    set_seed(36)

    train_transforms, val_transforms = get_transforms()

    # Load full training dataset
    full_train_dataset = CTDataset(
        Config.train_image_dir,
        Config.train_seg_dir,
        Config.train_clinical_csv,
        transform=train_transforms,
        is_train=True
    )

    # Load test dataset
    test_dataset = CTDataset(
        Config.test_image_dir,
        Config.test_seg_dir,
        Config.test_clinical_csv,
        transform=val_transforms,
        is_train=False
    )

    print(f"Full training dataset size: {len(full_train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Get clinical feature dimension
    sample = full_train_dataset[0]
    clin_feat_dim = len(sample['clinical'])
    print(f"Clinical feature dimension: {clin_feat_dim}")

    # Use stratified sampling to ensure each fold has samples from both classes
    labels = [full_train_dataset[i]['label'] for i in range(len(full_train_dataset))]
    print(f"All labels: {np.bincount(labels)}")

    # Separate positive and negative sample indices
    positive_indices = [i for i, label in enumerate(labels) if label == 1]
    negative_indices = [i for i, label in enumerate(labels) if label == 0]

    print(f"Positive samples: {len(positive_indices)}, Negative samples: {len(negative_indices)}")

    # Manually create 5-fold cross-validation, ensuring each fold has positive and negative samples
    n_folds = 5
    fold_size_pos = len(positive_indices) // n_folds
    fold_size_neg = len(negative_indices) // n_folds

    # Shuffle indices
    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)

    all_fold_results = []

    for fold in range(n_folds):
        print(f"\n{'=' * 40}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'=' * 40}")

        # Select validation set for current fold
        val_start_pos = fold * fold_size_pos
        val_end_pos = (fold + 1) * fold_size_pos if fold < n_folds - 1 else len(positive_indices)
        val_pos_indices = positive_indices[val_start_pos:val_end_pos]

        val_start_neg = fold * fold_size_neg
        val_end_neg = (fold + 1) * fold_size_neg if fold < n_folds - 1 else len(negative_indices)
        val_neg_indices = negative_indices[val_start_neg:val_end_neg]

        val_indices = val_pos_indices + val_neg_indices

        # Training set is all samples except validation set
        train_pos_indices = [i for i in positive_indices if i not in val_pos_indices]
        train_neg_indices = [i for i in negative_indices if i not in val_neg_indices]
        train_indices = train_pos_indices + train_neg_indices

        # Check class distribution
        train_labels = [labels[i] for i in train_indices]
        val_labels = [labels[i] for i in val_indices]
        print(f"Train class distribution: {np.bincount(train_labels)}")
        print(f"Validation class distribution: {np.bincount(val_labels)}")

        # Apply oversampling technique
        print("Applying SMOTE oversampling...")
        resampled_train_idx = apply_oversampling(full_train_dataset, train_indices)

        # Create data loaders
        train_loader = DataLoader(
            torch.utils.data.Subset(full_train_dataset, resampled_train_idx),
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            drop_last=True
        )

        val_loader = DataLoader(
            torch.utils.data.Subset(full_train_dataset, val_indices),
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers
        )

        # Train model
        print("Starting model training...")
        model = train_model(train_loader, val_loader, clin_feat_dim)

        # Load best model
        model.load_state_dict(torch.load(Config.model_save_path, map_location=Config.device))
        print(f"Loaded best model from {Config.model_save_path}")

        # Evaluate on validation set
        print("Evaluating on validation set...")
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        val_results = validate_model(model, val_loader, criterion, threshold=0.5)
        val_auc, val_loss = val_results[0], val_results[1]
        print(f"Fold {fold + 1} Validation AUC: {val_auc:.4f}")

        # Evaluate on test set
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers
        )

        print("Evaluating on test set...")
        test_results = validate_model(model, test_loader, criterion, threshold=0.487)
        test_auc, test_loss = test_results[0], test_results[1]
        print(f"Fold {fold + 1} Test AUC: {test_auc:.4f}")

        # Save results
        all_fold_results.append({
            'fold': fold + 1,
            'val_auc': val_auc,
            'test_auc': test_auc,
            'model': model
        })

    # Calculate average results
    avg_val_auc = np.mean([res['val_auc'] for res in all_fold_results])
    avg_test_auc = np.mean([res['test_auc'] for res in all_fold_results])

    print(f"\n{'=' * 50}")
    print(f"Average Validation AUC: {avg_val_auc:.4f}")
    print(f"Average Test AUC: {avg_test_auc:.4f}")
    print(f"{'=' * 50}")

    # Use all models for ensemble prediction
    print("Performing ensemble prediction on test set...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers
    )

    ensemble_probs = []
    with torch.no_grad():
        for batch in test_loader:
            ct = batch['ct'].to(Config.device).float()
            seg = batch['seg'].to(Config.device).float()
            clinical = batch['clinical'].to(Config.device).float()

            fold_probs = []
            for res in all_fold_results:
                model = res['model']
                logits = model(ct, seg, clinical)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_probs.append(probs)

            # Average probabilities
            avg_probs = np.mean(fold_probs, axis=0)
            ensemble_probs.extend(avg_probs)

    # Calculate ensemble metrics
    test_labels = [test_dataset[i]['label'] for i in range(len(test_dataset))]
    test_case_ids = [test_dataset[i]['case_id'] for i in range(len(test_dataset))]

    if len(np.unique(test_labels)) < 2:
        ensemble_auc = 0.5
        print("Warning: Only one class present in test set, setting AUC to 0.5")
    else:
        ensemble_auc = roc_auc_score(test_labels, ensemble_probs)

    ensemble_preds = (np.array(ensemble_probs) > 0.487).astype(int)

    if len(np.unique(test_labels)) < 2:
        ensemble_report = "Cannot generate classification report with only one class"
    else:
        ensemble_report = classification_report(test_labels, ensemble_preds)

    print(f"\n{'=' * 50}")
    print(f"Ensemble Test AUC: {ensemble_auc:.4f}")
    print(ensemble_report)
    print(f"{'=' * 50}")

    # Save results
    results_df = pd.DataFrame({
        'case_id': test_case_ids,
        'true_label': test_labels,
        'pred_prob': ensemble_probs,
        'pred_label': ensemble_preds
    })
    results_df.to_csv(os.path.join(Config.viz_save_dir, "ensemble_predictions.csv"), index=False)

    # Generate visualization charts
    print("Generating performance visualizations...")

    # ROC curve
    plot_roc_curve(test_labels, ensemble_probs, os.path.join(Config.viz_save_dir, "roc_curve.png"))

    # PR curve
    plot_pr_curve(test_labels, ensemble_probs, os.path.join(Config.viz_save_dir, "pr_curve.png"))

    # Confusion matrix
    plot_confusion_matrix(test_labels, ensemble_preds, os.path.join(Config.viz_save_dir, "confusion_matrix.png"))

    # Waterfall plot
    plot_waterfall(test_labels, ensemble_probs, test_case_ids, os.path.join(Config.viz_save_dir, "waterfall_plot.png"))

    # Analyze failure cases
    print("Analyzing failure cases...")
    failure_cases = []
    for i in range(len(test_labels)):
        if ensemble_preds[i] != test_labels[i]:
            failure_cases.append({
                'case_id': results_df.iloc[i]['case_id'],
                'true_label': test_labels[i],
                'pred_prob': ensemble_probs[i],
                'pred_label': ensemble_preds[i]
            })

    if failure_cases:
        print(f"Found {len(failure_cases)} failure cases")
        failure_df = pd.DataFrame(failure_cases)
        failure_df.to_csv(os.path.join(Config.viz_save_dir, "failure_analysis.csv"), index=False)
    else:
        print("No failure cases found!")

    # Visualize Grad-CAM (using first model)
    print("Generating Grad-CAM visualizations...")
    visualize_grad_cam(all_fold_results[0]['model'], test_loader, num_samples=5)

    print("\nAll operations completed successfully!")


if __name__ == "__main__":
    main()