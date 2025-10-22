import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LassoCV, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
from imblearn.over_sampling import SMOTE
import shap
import warnings
import os

warnings.filterwarnings('ignore')

# Set output directory
output_dir = "path/to/clinical_model_results"
os.makedirs(output_dir, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# 1. Load clinical feature data
train_clinical_csv = "path/to/train/clinical_features.csv"
test_clinical_csv = "path/to/test/clinical_features.csv"

train_df = pd.read_csv(train_clinical_csv)
test_df = pd.read_csv(test_clinical_csv)

print("Training set shape:", train_df.shape)
print("Test set shape:", test_df.shape)
print("Training set columns:", train_df.columns.tolist())

# 2. Data preprocessing - using only clinical features
# Clinical feature configuration
categorical_features = ['Eye', 'Sex', 'Previous_treatment_history', 'Systemic_medical_history']
numerical_features = ['Age', 'Duration_of_symptoms', 'Severity_of_symptoms']
all_clinical_features = categorical_features + numerical_features

# Check if all features exist
missing_features = [feat for feat in all_clinical_features if feat not in train_df.columns]
if missing_features:
    print(f"Warning: The following features are missing from the data: {missing_features}")
    # Use only existing features
    all_clinical_features = [feat for feat in all_clinical_features if feat in train_df.columns]
    categorical_features = [feat for feat in categorical_features if feat in train_df.columns]
    numerical_features = [feat for feat in numerical_features if feat in train_df.columns]

print("Clinical features used:", all_clinical_features)

# Extract features and labels
X_train = train_df[all_clinical_features].copy()
y_train = train_df['difficulty']
X_test = test_df[all_clinical_features].copy()
y_test = test_df['difficulty']

# Save patient IDs (if available)
patient_id_column = 'patient_id' if 'patient_id' in test_df.columns else None
patient_ids = test_df[patient_id_column] if patient_id_column else None

print(f"Training set class distribution: {np.bincount(y_train)}")
print(f"Test set class distribution: {np.bincount(y_test)}")

# 3. Handle categorical variables
label_encoders = {}
for col in categorical_features:
    if col in X_train.columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
        print(f"{col} encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Ensure all columns are numeric
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

# 4. Feature scaling - only scale numerical columns
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

print("Preprocessed training set shape:", X_train_scaled.shape)
print("Preprocessed test set shape:", X_test_scaled.shape)

# 5. SMOTE resampling for class imbalance
print("\nApplying SMOTE resampling...")
smote = SMOTE(random_state=36)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"Resampled training set shape: {X_train_resampled.shape}")
print(f"Resampled class distribution: {np.bincount(y_train_resampled)}")

# 6. LASSO feature selection (optional, clinical features are few, can skip)
use_lasso = len(all_clinical_features) > 5  # Use LASSO if we have many features

if use_lasso:
    print("\nPerforming LASSO feature selection...")
    alphas = np.logspace(-3, 1, 100)
    lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X_train_resampled, y_train_resampled)

    best_alpha = lasso_cv.alpha_
    lasso = Lasso(alpha=best_alpha, max_iter=10000)
    lasso.fit(X_train_resampled, y_train_resampled)

    selected_features = np.where(lasso.coef_ != 0)[0]
    print(f"LASSO selected {len(selected_features)} features")

    # Use selected features
    X_train_selected = X_train_resampled.iloc[:, selected_features]
    X_test_selected = X_test_scaled.iloc[:, selected_features]
    feature_names = X_train.columns[selected_features]
else:
    print("\nSkipping LASSO feature selection, using all clinical features")
    X_train_selected = X_train_resampled
    X_test_selected = X_test_scaled
    feature_names = X_train.columns.tolist()

print("Final features used:", feature_names.tolist())

# 7. Define and train multiple classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "XGBoost": XGBClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
}

# 8. Cross-validation model evaluation
print("\nCross-validation evaluation...")
results = {}
for name, clf in classifiers.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train_selected, y_train_resampled, cv=cv, scoring='roc_auc')
    results[name] = scores
    print(f"{name}: AUC = {np.mean(scores):.3f} (±{np.std(scores):.3f})")

# 9. Evaluate all models on test set
print("\nTest set evaluation...")
metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'Specificity']
test_results = pd.DataFrame(index=classifiers.keys(), columns=metrics)

# Store predictions for each classifier
all_predictions = {}
roc_data = {}  # Store ROC curve data for each classifier
pr_data = {}  # Store PR curve data for each classifier

for name, clf in classifiers.items():
    print(f"Training and evaluating {name}...")
    clf.fit(X_train_selected, y_train_resampled)
    y_pred = clf.predict(X_test_selected)
    y_prob = clf.predict_proba(X_test_selected)[:, 1] if hasattr(clf, "predict_proba") else np.zeros_like(y_pred)

    # Calculate specificity
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    test_results.loc[name, 'AUC'] = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    test_results.loc[name, 'Accuracy'] = accuracy_score(y_test, y_pred)
    test_results.loc[name, 'Precision'] = precision_score(y_test, y_pred, zero_division=0)
    test_results.loc[name, 'Recall'] = recall_score(y_test, y_pred, zero_division=0)
    test_results.loc[name, 'F1'] = f1_score(y_test, y_pred, zero_division=0)
    test_results.loc[name, 'Specificity'] = specificity

    # Save prediction results
    all_predictions[name] = {
        'true': y_test,
        'pred': y_pred,
        'prob': y_prob
    }

    # Calculate ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

    # Calculate PR curve data
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    pr_data[name] = {'precision': precision, 'recall': recall, 'auc': pr_auc}

    # Save prediction results to CSV for each classifier (with patient IDs)
    if patient_id_column:
        pred_df = pd.DataFrame({
            'patient_id': patient_ids,
            'true_label': y_test,
            'predicted_label': y_pred,
            'prediction_probability': y_prob
        })
    else:
        pred_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred,
            'prediction_probability': y_prob
        })
    pred_df.to_csv(os.path.join(output_dir, f'{name}_clinical_predictions.csv'), index=False)


# 10. Plot performance comparison
def plot_radar_chart(results_df, title):
    """Plot radar chart for model performance comparison"""
    categories = list(results_df.columns)
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for idx, row in results_df.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=idx)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_ylim(0, 1)
    plt.title(title, size=16, y=1.05)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(os.path.join(output_dir, 'clinical_model_comparison_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()


# Plot radar chart
plot_radar_chart(test_results.astype(float), "Clinical Features Only - Model Performance Comparison")

# 11. Plot ROC curves for all classifiers
plt.figure(figsize=(10, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(classifiers)))

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
plt.savefig(os.path.join(output_dir, 'clinical_roc_curves.png'), dpi=300)
plt.close()

# 12. Plot PR curves for all classifiers
plt.figure(figsize=(10, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(classifiers)))

for i, (name, data) in enumerate(pr_data.items()):
    plt.plot(data['recall'], data['precision'], color=colors[i], lw=2,
             label=f'{name} (AUC = {data["auc"]:.3f})')

# Calculate baseline (positive class proportion)
baseline = len(y_test[y_test == 1]) / len(y_test)
plt.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline (Precision = {baseline:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves - Clinical Features Only')
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'clinical_pr_curves.png'), dpi=300)
plt.close()

# 13. Select best model
best_model_name = test_results['AUC'].astype(float).idxmax()
best_model = classifiers[best_model_name]
print(f"\nBest model: {best_model_name}")

# 14. Plot detailed evaluation for best model
best_model.fit(X_train_selected, y_train_resampled)
y_pred_best = best_model.predict(X_test_selected)
y_prob_best = best_model.predict_proba(X_test_selected)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Difficult'],
            yticklabels=['Actual Normal', 'Actual Difficult'])
plt.title(f'Confusion Matrix - {best_model_name} (Clinical Features Only)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(output_dir, 'clinical_best_model_confusion_matrix.png'), dpi=300)
plt.close()

# Best model ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_model_name} (Clinical Features Only)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'clinical_best_model_roc_curve.png'), dpi=300)
plt.close()

# 15. Feature importance analysis
for name, clf in classifiers.items():
    try:
        # Get feature importance
        if hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            importance = np.abs(clf.coef_[0])
        else:
            print(f"{name} does not support feature importance analysis")
            continue

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)

        # Assign different colors for different feature types
        colors = []
        for feature in importance_df['feature']:
            if feature in categorical_features:
                colors.append('skyblue')  # Categorical features
            else:
                colors.append('lightcoral')  # Numerical features

        # Plot feature importance bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title(f'Feature Importance - {name} (Clinical Features Only)')
        plt.yticks(range(len(importance_df)), importance_df['feature'])

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor='skyblue', label='Categorical Features'),
            Patch(facecolor='lightcoral', label='Numerical Features')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_clinical_feature_importance.png'), dpi=300)
        plt.close()

        print(f"Saved feature importance plot for {name}")

        # Save feature importance data
        importance_df.to_csv(os.path.join(output_dir, f'{name}_clinical_feature_importance.csv'), index=False)

    except Exception as e:
        print(f"Error generating feature importance plot for {name}: {e}")

# 16. SHAP feature importance analysis
shap_supported_models = ["Logistic Regression", "XGBoost", "Random Forest"]

if best_model_name in shap_supported_models and hasattr(best_model, 'predict_proba'):
    try:
        print(f"\nPerforming SHAP analysis for {best_model_name}...")

        # Create SHAP explainer
        if best_model_name == "Logistic Regression":
            explainer = shap.LinearExplainer(best_model, X_train_selected, feature_names=feature_names)
        else:
            explainer = shap.TreeExplainer(best_model, feature_names=feature_names)

        shap_values = explainer(X_test_selected)

        # Feature importance plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_selected, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {best_model_name}")
        plt.savefig(os.path.join(output_dir, 'clinical_shap_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Beeswarm plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_selected, feature_names=feature_names, show=False)
        plt.title(f"SHAP Beeswarm Plot - {best_model_name}")
        plt.savefig(os.path.join(output_dir, 'clinical_shap_beeswarm_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("SHAP analysis completed")

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
else:
    print(f"\nSkipping SHAP analysis for {best_model_name}")


# 17. Plot logistic regression nomogram (if best model is logistic regression)
def plot_clinical_nomogram(feature_names, coefficients, intercept, output_dir, feature_descriptions=None):
    """Create clinical feature nomogram"""
    plt.figure(figsize=(14, 10))

    # Set feature descriptions
    if feature_descriptions is None:
        feature_descriptions = {
            'Eye': 'Eye (0=Left, 1=Right)',
            'Sex': 'Sex (0=Male, 1=Female)',
            'Age': 'Age (years)',
            'Duration_of_symptoms': 'Symptom Duration (months)',
            'Severity_of_symptoms': 'Symptom Severity',
            'Previous_treatment_history': 'Previous Treatment (0=No, 1=Yes)',
            'Systemic_medical_history': 'Systemic History (0=No, 1=Yes)'
        }

    n_features = len(feature_names)
    y_positions = np.arange(n_features) * 1.5

    # Calculate point ranges for each feature
    total_points = 0
    feature_points = []

    for i, (feature, coef) in enumerate(zip(feature_names, coefficients)):
        # Create point axis for each feature
        feature_min = coef * 0  # Assume minimum value is 0
        feature_max = coef * 1  # Assume maximum value is 1 (for standardized features)

        # Draw feature axis
        plt.plot([feature_min, feature_max], [y_positions[i], y_positions[i]], 'b-', linewidth=2)

        # Add ticks
        for point in [feature_min, (feature_min + feature_max) / 2, feature_max]:
            plt.plot([point, point], [y_positions[i] - 0.1, y_positions[i] + 0.1], 'k-', linewidth=1)
            plt.text(point, y_positions[i] - 0.2, f'{point:.2f}', ha='center', va='top', fontsize=9)

        # Add feature labels
        feature_label = feature_descriptions.get(feature, feature)
        plt.text(feature_min - 0.5, y_positions[i], feature_label, ha='right', va='center',
                 fontsize=11, fontweight='bold')

        feature_points.append((feature_min, feature_max))
        total_points += feature_max

    # Add total points axis and probability axis
    total_min = intercept
    total_max = total_points + intercept

    # Draw total points axis
    plt.plot([total_min, total_max], [-2, -2], 'r-', linewidth=3)
    plt.text((total_min + total_max) / 2, -2.5, 'Total Points', ha='center', va='top',
             fontsize=12, fontweight='bold')

    # Add probability scale
    probability_points = np.linspace(total_min, total_max, 11)
    probabilities = 1 / (1 + np.exp(-probability_points))

    for i, (point, prob) in enumerate(zip(probability_points, probabilities)):
        if i % 2 == 0:  # Show labels for every other point
            plt.plot([point, point], [-2.1, -1.9], 'k-', linewidth=1)
            plt.text(point, -2.3, f'{prob:.2f}', ha='center', va='top', fontsize=9)

    plt.ylim(-3, n_features * 1.5)
    plt.xlim(min(total_min, *[fp[0] for fp in feature_points]) - 1,
             max(total_max, *[fp[1] for fp in feature_points]) + 1)
    plt.axis('off')
    plt.title('Clinical Features Nomogram for Surgical Difficulty Prediction',
              fontsize=14, fontweight='bold', pad=20)

    # Add instructions
    explanation = (
        "How to use:\n"
        "1. For each patient, find their value on each feature axis\n"
        "2. Read the corresponding points\n"
        "3. Sum all points and add the intercept\n"
        "4. Find the total on the bottom axis to get probability\n"
        "5. Probability > 0.5 predicts difficult surgery"
    )
    plt.figtext(0.02, 0.02, explanation, fontsize=10, va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clinical_nomogram.png'), dpi=300, bbox_inches='tight')
    plt.close()


# If best model is logistic regression, plot nomogram
if best_model_name == "Logistic Regression" and hasattr(best_model, 'coef_'):
    coefficients = best_model.coef_[0]
    intercept = best_model.intercept_[0]

    print("\nLogistic regression coefficients:")
    for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
        print(f"  {name}: {coef:.4f}")
    print(f"Intercept: {intercept:.4f}")

    # Feature descriptions
    feature_descriptions = {
        'Eye': 'Eye (0=Left, 1=Right)',
        'Sex': 'Sex (0=Male, 1=Female)',
        'Age': 'Age (years)',
        'Duration_of_symptoms': 'Symptom Duration (months)',
        'Severity_of_symptoms': 'Symptom Severity',
        'Previous_treatment_history': 'Previous Treatment (0=No, 1=Yes)',
        'Systemic_medical_history': 'Systemic Medical History (0=No, 1=Yes)'
    }

    plot_clinical_nomogram(feature_names, coefficients, intercept, output_dir, feature_descriptions)
    print("Nomogram saved")
else:
    print(f"\n{best_model_name} is not a logistic regression model, skipping nomogram")

# 18. Save final results
# Save performance results
test_results.to_csv(os.path.join(output_dir, 'clinical_model_performance.csv'))

# Save model configuration information
config_info = {
    'features_used': feature_names,
    'best_model': best_model_name,
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'class_distribution_train': str(np.bincount(y_train)),
    'class_distribution_test': str(np.bincount(y_test))
}

with open(os.path.join(output_dir, 'clinical_model_config.txt'), 'w') as f:
    for key, value in config_info.items():
        f.write(f"{key}: {value}\n")

print(f"\nAll analysis completed! Results saved to: {output_dir}")
print(f"Best model: {best_model_name}")
print("Test set performance:")
print(test_results.round(3))