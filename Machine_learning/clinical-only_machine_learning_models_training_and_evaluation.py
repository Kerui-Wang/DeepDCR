import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             balanced_accuracy_score, matthews_corrcoef, 
                             cohen_kappa_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
import warnings
import os
from scipy import stats
import json
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')

# Set output directory
output_dir = "clinical_model_results_scientific"
os.makedirs(output_dir, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# 1. Load clinical feature data
train_clinical_csv = "path/to/train/clinical_features.csv"
test_clinical_csv = "path/to/test/clinical_features.csv"

train_df = pd.read_csv(train_clinical_csv)
test_df = pd.read_csv(test_clinical_csv)

print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Training set columns: {train_df.columns.tolist()}")

# 2. Data preprocessing - using only clinical features
categorical_features = ['Eye', 'Sex', 'Previous_treatment_history', 'Systemic_medical_history']
numerical_features = ['Age', 'Duration_of_symptoms', 'Severity_of_symptoms']
all_clinical_features = categorical_features + numerical_features

# Check if all features exist
missing_features = [feat for feat in all_clinical_features if feat not in train_df.columns]
if missing_features:
    print(f"Warning: Missing features: {missing_features}")
    all_clinical_features = [feat for feat in all_clinical_features if feat in train_df.columns]
    categorical_features = [feat for feat in categorical_features if feat in train_df.columns]
    numerical_features = [feat for feat in numerical_features if feat in train_df.columns]

print(f"Clinical features used: {all_clinical_features}")

# Extract features and labels
X_train_full = train_df[all_clinical_features].copy()
y_train_full = train_df['difficulty']
X_test = test_df[all_clinical_features].copy()
y_test = test_df['difficulty']

# Save patient IDs
patient_id_column = 'patient_id' if 'patient_id' in test_df.columns else None
if patient_id_column:
    test_patient_ids = test_df[patient_id_column].values

print(f"Training set class distribution: {np.bincount(y_train_full)}")
print(f"Test set class distribution: {np.bincount(y_test)}")

# 3. Setup for nested cross-validation
print("\n" + "=" * 60)
print("NESTED CROSS-VALIDATION SETUP")
print("=" * 60)

# Outer CV for model evaluation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=36)
# Inner CV for hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=36)

# 4. Create preprocessing pipeline
# Separate features by type for correct preprocessing
categorical_features_clean = [f for f in categorical_features if f in X_train_full.columns]
numerical_features_clean = [f for f in numerical_features if f in X_train_full.columns]

print(f"Categorical features to encode: {categorical_features_clean}")
print(f"Numerical features to scale: {numerical_features_clean}")

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features_clean),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features_clean)
    ],
    remainder='passthrough'
)

# Get feature names after preprocessing
# Fit preprocessor to get feature names
preprocessor.fit(X_train_full)
feature_names_after_preprocessing = []

# Get numerical feature names
num_features = numerical_features_clean
feature_names_after_preprocessing.extend(num_features)

# Get categorical feature names
if categorical_features_clean:
    cat_encoder = preprocessor.named_transformers_['cat']
    for i, cat in enumerate(categorical_features_clean):
        categories = cat_encoder.categories_[i][1:]  # Skip first category
        feature_names_after_preprocessing.extend([f"{cat}_{cat_val}" for cat_val in categories])

print(f"Total features after preprocessing: {len(feature_names_after_preprocessing)}")

# 5. Define classifiers with hyperparameter grids
classifiers = {
    "Logistic Regression": {
        'clf': LogisticRegression(random_state=36, max_iter=1000),
        'params': {
            'clf__C': [0.01, 0.1, 1.0, 10.0],
            'clf__penalty': ['l2'],
            'clf__solver': ['liblinear', 'lbfgs']
        }
    },
    "Random Forest": {
        'clf': RandomForestClassifier(random_state=36),
        'params': {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [None, 10, 20],
            'clf__min_samples_split': [2, 5, 10]
        }
    }
}

# 6. Perform nested cross-validation
print("\n" + "=" * 60)
print("PERFORMING NESTED CROSS-VALIDATION")
print("=" * 60)

outer_results = {}
best_models = {}
feature_importances = {}

for name, model_info in classifiers.items():
    print(f"\nTraining {name} with nested cross-validation...")
    
    # Create pipeline with SMOTE inside to prevent data leakage
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=36)),
        ('clf', model_info['clf'])
    ])
    
    # Grid search with inner CV
    grid_search = GridSearchCV(
        pipeline,
        model_info['params'],
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    # Perform outer CV
    outer_fold_results = []
    outer_fold_predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_full, y_train_full)):
        print(f"  Outer Fold {fold + 1}/5")
        
        # Split data for this fold
        X_train_fold = X_train_full.iloc[train_idx]
        y_train_fold = y_train_full.iloc[train_idx]
        X_val_fold = X_train_full.iloc[val_idx]
        y_val_fold = y_train_full.iloc[val_idx]
        
        # Fit pipeline with grid search (SMOTE only applied to training fold)
        grid_search.fit(X_train_fold, y_train_fold)
        
        # Get best model from this fold
        best_model_fold = grid_search.best_estimator_
        
        # Predict on validation fold
        y_pred = best_model_fold.predict(X_val_fold)
        y_prob = best_model_fold.predict_proba(X_val_fold)[:, 1]
        
        # Calculate comprehensive metrics
        fold_metrics = {
            'accuracy': accuracy_score(y_val_fold, y_pred),
            'roc_auc': roc_auc_score(y_val_fold, y_prob),
            'precision': precision_score(y_val_fold, y_pred, zero_division=0),
            'recall': recall_score(y_val_fold, y_pred, zero_division=0),
            'f1': f1_score(y_val_fold, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_val_fold, y_pred),
            'mcc': matthews_corrcoef(y_val_fold, y_pred),
            'kappa': cohen_kappa_score(y_val_fold, y_pred)
        }
        
        # Calculate specificity
        cm = confusion_matrix(y_val_fold, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fold_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        outer_fold_results.append(fold_metrics)
        outer_fold_predictions.append({
            'true': y_val_fold.values,
            'pred': y_pred,
            'prob': y_prob
        })
        
        print(f"    AUC: {fold_metrics['roc_auc']:.3f}, "
              f"F1: {fold_metrics['f1']:.3f}, "
              f"Balanced Accuracy: {fold_metrics['balanced_accuracy']:.3f}")
    
    # Aggregate results across folds
    metrics_df = pd.DataFrame(outer_fold_results)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()
    
    outer_results[name] = {
        'mean': mean_metrics,
        'std': std_metrics,
        'all_folds': outer_fold_results,
        'predictions': outer_fold_predictions
    }
    
    print(f"\n  {name} Cross-Validation Results:")
    print(f"    Mean AUC: {mean_metrics['roc_auc']:.3f} (±{std_metrics['roc_auc']:.3f})")
    print(f"    Mean F1: {mean_metrics['f1']:.3f} (±{std_metrics['f1']:.3f})")
    print(f"    Mean Balanced Accuracy: {mean_metrics['balanced_accuracy']:.3f} (±{std_metrics['balanced_accuracy']:.3f})")
    
    # Train final model on all training data with best hyperparameters
    print(f"\n  Training final {name} model on all training data...")
    final_grid_search = GridSearchCV(
        pipeline,
        model_info['params'],
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    final_grid_search.fit(X_train_full, y_train_full)
    best_models[name] = final_grid_search.best_estimator_
    
    # Get feature importances
    try:
        if hasattr(best_models[name].named_steps['clf'], 'feature_importances_'):
            importances = best_models[name].named_steps['clf'].feature_importances_
        elif hasattr(best_models[name].named_steps['clf'], 'coef_'):
            importances = np.abs(best_models[name].named_steps['clf'].coef_[0])
        else:
            importances = None
        
        if importances is not None:
            # Match importances with feature names
            feature_importances[name] = pd.DataFrame({
                'feature': feature_names_after_preprocessing,
                'importance': importances
            }).sort_values('importance', ascending=False)
    except Exception as e:
        print(f"  Could not extract feature importances: {e}")
        feature_importances[name] = None

# 7. FINAL EVALUATION ON HELD-OUT TEST SET (ONCE)
print("\n" + "=" * 60)
print("FINAL EVALUATION ON TEST SET")
print("=" * 60)

test_results = pd.DataFrame(index=classifiers.keys(), 
                            columns=['AUC', 'Accuracy', 'Balanced_Accuracy', 
                                     'Precision', 'Recall', 'F1', 
                                     'Specificity', 'MCC', 'Kappa'])

roc_data = {}
pr_data = {}
test_predictions = {}
calibration_data = {}

for name in classifiers.keys():
    print(f"\nEvaluating {name} on test set...")
    
    model = best_models[name]
    
    # Predict on test set (only once)
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    
    # Calculate comprehensive metrics
    test_results.loc[name, 'AUC'] = roc_auc_score(y_test, y_prob_test)
    test_results.loc[name, 'Accuracy'] = accuracy_score(y_test, y_pred_test)
    test_results.loc[name, 'Balanced_Accuracy'] = balanced_accuracy_score(y_test, y_pred_test)
    test_results.loc[name, 'Precision'] = precision_score(y_test, y_pred_test, zero_division=0)
    test_results.loc[name, 'Recall'] = recall_score(y_test, y_pred_test, zero_division=0)
    test_results.loc[name, 'F1'] = f1_score(y_test, y_pred_test, zero_division=0)
    test_results.loc[name, 'MCC'] = matthews_corrcoef(y_test, y_pred_test)
    test_results.loc[name, 'Kappa'] = cohen_kappa_score(y_test, y_pred_test)
    
    # Calculate specificity
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    test_results.loc[name, 'Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Store ROC and PR curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    roc_auc = auc(fpr, tpr)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_test)
    pr_auc = auc(recall_curve, precision_curve)
    pr_data[name] = {'precision': precision_curve, 'recall': recall_curve, 'auc': pr_auc}
    
    # Store calibration data
    prob_true, prob_pred = calibration_curve(y_test, y_prob_test, n_bins=10, strategy='uniform')
    calibration_data[name] = {'prob_true': prob_true, 'prob_pred': prob_pred}
    
    # Store predictions
    test_predictions[name] = {
        'true': y_test.values,
        'pred': y_pred_test,
        'prob': y_prob_test
    }
    
    # Save predictions to CSV
    if patient_id_column:
        pred_df = pd.DataFrame({
            'patient_id': test_patient_ids,
            'true_label': y_test,
            'predicted_label': y_pred_test,
            'prediction_probability': y_prob_test
        })
    else:
        pred_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred_test,
            'prediction_probability': y_prob_test
        })
    pred_df.to_csv(os.path.join(output_dir, f'{name}_test_predictions.csv'), index=False)
    
    print(f"  Test AUC: {test_results.loc[name, 'AUC']:.3f}")
    print(f"  Test F1: {test_results.loc[name, 'F1']:.3f}")
    print(f"  Test Balanced Accuracy: {test_results.loc[name, 'Balanced_Accuracy']:.3f}")

# 8. Select best model based on cross-validation performance
cv_performance = {name: outer_results[name]['mean']['roc_auc'] for name in classifiers.keys()}
best_model_name = max(cv_performance, key=cv_performance.get)
best_model = best_models[best_model_name]

print("\n" + "=" * 60)
print(f"BEST MODEL: {best_model_name}")
print("=" * 60)
print(f"Cross-validation AUC: {cv_performance[best_model_name]:.3f}")
print(f"Test AUC: {test_results.loc[best_model_name, 'AUC']:.3f}")
print(f"Test F1: {test_results.loc[best_model_name, 'F1']:.3f}")
print(f"Test Balanced Accuracy: {test_results.loc[best_model_name, 'Balanced_Accuracy']:.3f}")

# 9. Visualization functions
def plot_roc_curves(roc_data, save_path):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(roc_data)))
    
    for i, (name, data) in enumerate(roc_data.items()):
        plt.plot(data['fpr'], data['tpr'], color=colors[i], lw=2,
                 label=f'{name} (AUC = {data["auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Clinical Features Only')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curves(pr_data, save_path, baseline):
    """Plot Precision-Recall curves for all models"""
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(pr_data)))
    
    for i, (name, data) in enumerate(pr_data.items()):
        plt.plot(data['recall'], data['precision'], color=colors[i], lw=2,
                 label=f'{name} (AUC = {data["auc"]:.3f})')
    
    plt.axhline(y=baseline, color='k', linestyle='--', 
                label=f'Baseline (Precision = {baseline:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Clinical Features Only')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_for_model(y_true, y_pred, model_name, save_path):
    """Plot confusion matrix for a specific model"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Difficult'],
                yticklabels=['Normal', 'Difficult'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_calibration_curves(calibration_data, save_path):
    """Plot calibration curves for all models"""
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(calibration_data)))
    
    for i, (name, data) in enumerate(calibration_data.items()):
        plt.plot(data['prob_pred'], data['prob_true'], 's-', color=colors[i], 
                 label=f'{name}', markersize=8)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curves - Clinical Features Only')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(feature_importances_df, model_name, save_path):
    """Plot feature importance for a model"""
    if feature_importances_df is None or len(feature_importances_df) == 0:
        return
    
    # Take all features (clinical features are few)
    plt.figure(figsize=(12, 8))
    
    # Color coding for feature types
    colors = []
    for feature in feature_importances_df['feature']:
        if any(cat in feature for cat in categorical_features_clean):
            colors.append('skyblue')  # Categorical
        else:
            colors.append('lightcoral')  # Numerical
    
    bars = plt.barh(range(len(feature_importances_df)), 
                    feature_importances_df['importance'], 
                    color=colors)
    
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title(f'Feature Importance - {model_name} (Clinical Features)')
    plt.yticks(range(len(feature_importances_df)), feature_importances_df['feature'])
    
    # Add value labels
    for i, v in enumerate(feature_importances_df['importance']):
        plt.text(v, i, f' {v:.3f}', va='center')
    
    plt.gca().invert_yaxis()
    
    # Add legend for feature types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='Categorical Features'),
        Patch(facecolor='lightcoral', label='Numerical Features')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 10. Generate all visualizations
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# ROC curves
plot_roc_curves(roc_data, os.path.join(output_dir, 'clinical_roc_curves.png'))

# PR curves
baseline = len(y_test[y_test == 1]) / len(y_test)
plot_pr_curves(pr_data, os.path.join(output_dir, 'clinical_pr_curves.png'), baseline)

# Confusion matrix for best model
best_preds = test_predictions[best_model_name]
plot_confusion_matrix_for_model(
    best_preds['true'], best_preds['pred'], 
    best_model_name, 
    os.path.join(output_dir, f'clinical_best_model_confusion_matrix.png')
)

# Calibration curves
plot_calibration_curves(calibration_data, os.path.join(output_dir, 'clinical_calibration_curves.png'))

# Feature importance for each model
for name in classifiers.keys():
    if name in feature_importances and feature_importances[name] is not None:
        plot_feature_importance(
            feature_importances[name], 
            name, 
            os.path.join(output_dir, f'{name}_feature_importance.png')
        )
        # Save feature importance to CSV
        feature_importances[name].to_csv(
            os.path.join(output_dir, f'{name}_feature_importance.csv'), 
            index=False
        )

# 11. SHAP analysis for interpretable models
print("\n" + "=" * 60)
print("SHAP ANALYSIS")
print("=" * 60)

shap_supported_models = ["Logistic Regression", "Random Forest"]

for model_name in shap_supported_models:
    if model_name in best_models:
        try:
            print(f"Computing SHAP values for {model_name}...")
            
            # Get the preprocessed test data
            model = best_models[model_name]
            
            # Get the preprocessor and transform test data
            preprocessor = model.named_steps['preprocessor']
            X_test_transformed = preprocessor.transform(X_test)
            
            # Create SHAP explainer based on model type
            if model_name == "Logistic Regression":
                explainer = shap.LinearExplainer(
                    model.named_steps['clf'], 
                    X_test_transformed,
                    feature_names=feature_names_after_preprocessing
                )
            else:
                explainer = shap.TreeExplainer(
                    model.named_steps['clf'],
                    feature_names=feature_names_after_preprocessing
                )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test_transformed)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test_transformed, 
                             feature_names=feature_names_after_preprocessing,
                             show=False)
            plt.title(f"SHAP Summary - {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_shap_summary.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Bar plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_transformed, 
                             feature_names=feature_names_after_preprocessing,
                             plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance - {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_shap_importance.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  SHAP analysis completed for {model_name}")
            
        except Exception as e:
            print(f"  SHAP analysis failed for {model_name}: {e}")

# 12. Create clinical nomogram (if best model is logistic regression)
def create_clinical_nomogram(coefficients, intercept, feature_names, feature_descriptions, 
                            original_data, output_path):
    """Create a clinically interpretable nomogram using original feature values"""
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot feature contributions
    y_pos = np.arange(len(feature_names))
    colors = ['lightcoral' if 'Age' in f or 'Duration' in f or 'Severity' in f 
              else 'skyblue' for f in feature_names]
    
    ax1.barh(y_pos, coefficients, color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feature_names)
    ax1.set_xlabel('Coefficient Value (Log-odds contribution)')
    ax1.set_title('Clinical Feature Contributions to Surgical Difficulty Prediction')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(coefficients):
        ax1.text(v, i, f' {v:.3f}', va='center')
    
    # Create probability scale
    total_points_range = np.linspace(-3, 3, 13)
    probability_range = 1 / (1 + np.exp(-total_points_range))
    
    ax2.plot(total_points_range, probability_range, 'r-', linewidth=2)
    ax2.set_xlabel('Total Score (sum of feature contributions + intercept)')
    ax2.set_ylabel('Predicted Probability of Difficult Surgery')
    ax2.set_title('Probability Scale')
    ax2.grid(True, alpha=0.3)
    
    # Add probability markers
    for prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
        total_point = -np.log((1/prob) - 1)
        ax2.plot([total_point, total_point], [0, prob], 'k--', alpha=0.5)
        ax2.text(total_point, prob, f'{prob:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Add usage instructions
    instructions = (
        "How to use this nomogram:\n"
        "1. For each clinical feature, multiply the patient's value by the coefficient\n"
        "2. Sum all contributions and add the intercept\n"
        "3. Find the total score on the bottom plot to get probability\n"
        "4. Probability > 0.5 predicts difficult surgery"
    )
    plt.figtext(0.02, 0.02, instructions, fontsize=10, va='bottom')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a simple scoring system for clinical use
    scoring_system = {}
    for i, feature in enumerate(feature_names):
        # Convert to original scale if possible
        if feature in original_data.columns:
            # Calculate typical values
            mean_val = original_data[feature].mean()
            std_val = original_data[feature].std()
            
            # Create bins for scoring
            if std_val > 0:
                low_score = coefficients[i] * (mean_val - std_val)
                high_score = coefficients[i] * (mean_val + std_val)
                scoring_system[feature] = {
                    'coefficient': coefficients[i],
                    'typical_range': (mean_val - std_val, mean_val + std_val),
                    'score_range': (low_score, high_score)
                }
    
    return scoring_system

# If best model is logistic regression, create nomogram
if best_model_name == "Logistic Regression" and hasattr(best_model.named_steps['clf'], 'coef_'):
    coefficients = best_model.named_steps['clf'].coef_[0]
    intercept = best_model.named_steps['clf'].intercept_[0]
    
    print(f"\nLogistic regression coefficients:")
    for i, (name, coef) in enumerate(zip(feature_names_after_preprocessing, coefficients)):
        print(f"  {name}: {coef:.4f}")
    print(f"Intercept: {intercept:.4f}")
    
    # Create clinical nomogram
    scoring_system = create_clinical_nomogram(
        coefficients, intercept, 
        feature_names_after_preprocessing,
        {},  # Empty descriptions for now
        X_train_full,  # Original data for reference ranges
        os.path.join(output_dir, 'clinical_nomogram.png')
    )
    
    # Save scoring system
    with open(os.path.join(output_dir, 'clinical_scoring_system.json'), 'w') as f:
        json.dump(scoring_system, f, indent=2)
    
    print("Nomogram and scoring system saved")
else:
    print(f"\n{best_model_name} is not a logistic regression model, skipping nomogram")

# 13. Statistical comparison of models
print("\n" + "=" * 60)
print("STATISTICAL COMPARISON")
print("=" * 60)

# Bootstrap comparison of models
def bootstrap_comparison(model1_preds, model2_preds, y_true, n_bootstraps=1000):
    """Compare two models using bootstrap"""
    auc_diffs = []
    
    for _ in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        pred1_boot = model1_preds[indices]
        pred2_boot = model2_preds[indices]
        
        auc1 = roc_auc_score(y_true_boot, pred1_boot)
        auc2 = roc_auc_score(y_true_boot, pred2_boot)
        auc_diffs.append(auc1 - auc2)
    
    # Calculate confidence interval
    auc_diffs = np.array(auc_diffs)
    ci_lower = np.percentile(auc_diffs, 2.5)
    ci_upper = np.percentile(auc_diffs, 97.5)
    
    # Calculate p-value (two-sided)
    p_value = 2 * min(
        np.mean(auc_diffs <= 0),
        np.mean(auc_diffs >= 0)
    )
    
    return {
        'mean_difference': np.mean(auc_diffs),
        'ci_95': (ci_lower, ci_upper),
        'p_value': p_value
    }

# Compare all pairs of models
model_comparisons = {}
model_names = list(classifiers.keys())

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        model1 = model_names[i]
        model2 = model_names[j]
        
        preds1 = test_predictions[model1]['prob']
        preds2 = test_predictions[model2]['prob']
        
        comparison = bootstrap_comparison(preds1, preds2, y_test.values)
        model_comparisons[f"{model1}_vs_{model2}"] = comparison
        
        print(f"\n{model1} vs {model2}:")
        print(f"  AUC difference: {comparison['mean_difference']:.3f}")
        print(f"  95% CI: [{comparison['ci_95'][0]:.3f}, {comparison['ci_95'][1]:.3f}]")
        print(f"  p-value: {comparison['p_value']:.3f}")

# 14. Save all results
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save test results
test_results.to_csv(os.path.join(output_dir, 'clinical_model_test_performance.csv'))

# Save cross-validation results
cv_results = {}
for name in classifiers.keys():
    cv_results[name] = {
        'mean_metrics': outer_results[name]['mean'].to_dict(),
        'std_metrics': outer_results[name]['std'].to_dict()
    }

with open(os.path.join(output_dir, 'clinical_cross_validation_results.json'), 'w') as f:
    json.dump(cv_results, f, indent=2)

# Save model comparisons
with open(os.path.join(output_dir, 'clinical_model_comparisons.json'), 'w') as f:
    json.dump(model_comparisons, f, indent=2)

# Save configuration
config = {
    'preprocessing': {
        'categorical_features': categorical_features_clean,
        'numerical_features': numerical_features_clean
    },
    'cross_validation': {
        'outer_folds': 5,
        'inner_folds': 3,
        'random_state': 36
    },
    'models_tested': list(classifiers.keys()),
    'best_model': best_model_name
}

with open(os.path.join(output_dir, 'clinical_experiment_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

# 15. Create summary report
print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)

print(f"\nBest Model: {best_model_name}")
print(f"Cross-validation AUC: {cv_performance[best_model_name]:.3f}")
print(f"Test Performance:")
for metric in ['AUC', 'Accuracy', 'F1', 'Balanced_Accuracy', 'MCC']:
    print(f"  {metric}: {test_results.loc[best_model_name, metric]:.3f}")

print(f"\nAll Models Test Performance:")
print(test_results.round(3).to_string())

print(f"\nResults saved to: {output_dir}")
print("=" * 60)
print("Scientific analysis completed successfully!")
