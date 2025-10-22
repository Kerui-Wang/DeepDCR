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
                             roc_curve, auc)
from imblearn.over_sampling import SMOTE
import shap
import warnings
import os

warnings.filterwarnings('ignore')

# Set output directory
output_dir = "./ML_results"
os.makedirs(output_dir, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# 1. Load data
train_df = pd.read_csv(os.path.join(output_dir, "ML_shape_feature_train.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "ML_shape_feature_test.csv"))

# Check data
print("Training set columns:", train_df.columns.tolist())
print("Test set columns:", test_df.columns.tolist())

# 2. Data preprocessing
# Separate features and labels
# First identify and remove potential ID columns
id_columns = [col for col in train_df.columns if 'id' in col.lower() or 'patient' in col.lower()]
print("Potential ID columns:", id_columns)

# Extract patient ID column (assuming case_id is patient ID)
patient_id_column = 'case_id' if 'case_id' in test_df.columns else None
patient_ids = test_df[patient_id_column] if patient_id_column else None

# Remove ID columns
X_train = train_df.drop(['difficulty'] + id_columns, axis=1)
y_train = train_df['difficulty']
X_test = test_df.drop(['difficulty'] + id_columns, axis=1)
y_test = test_df['difficulty']

print("Feature columns:", X_train.columns.tolist())

# Handle categorical variables
clinical_categorical = ['Eye', 'Sex', 'Previous_treatment_history', 'Systemic_medical_history', 'Severity_of_symptoms']
clinical_numerical = ['Age', 'Duration_of_symptoms']

# Encode categorical variables
label_encoders = {}
for col in clinical_categorical:
    if col in X_train.columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

# Ensure all columns are numeric
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

# Feature scaling - only for numeric columns
numeric_columns = X_train.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])

# 3. SMOTE resampling for class imbalance
smote = SMOTE(random_state=36)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# 4. LASSO feature selection
# Use higher regularization strength to reduce number of features
alphas = np.logspace(-3, 1, 100)  # Adjust alpha range towards larger values (stronger regularization)
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_resampled, y_train_resampled)

# Calculate LASSO coefficient path
coefs = []
for alpha in lasso_cv.alphas_:
    lasso_temp = Lasso(alpha=alpha, max_iter=10000)
    lasso_temp.fit(X_train_resampled, y_train_resampled)
    coefs.append(lasso_temp.coef_)

# Plot LASSO regularization path
plt.figure(figsize=(12, 8))

# Calculate mean MSE and standard deviation
mean_mse = np.mean(lasso_cv.mse_path_, axis=1)
std_mse = np.std(lasso_cv.mse_path_, axis=1)

# Find alpha corresponding to minimum MSE (lambda.min)
min_mse_idx = np.argmin(mean_mse)
lambda_min = lasso_cv.alphas_[min_mse_idx]
min_mse = mean_mse[min_mse_idx]

# Find largest alpha within one standard error of minimum MSE (lambda.1se)
mse_threshold = min_mse + std_mse[min_mse_idx]
lambda_1se_idx = np.where(mean_mse <= mse_threshold)[0][0]
lambda_1se = lasso_cv.alphas_[lambda_1se_idx]

# Plot MSE curve and error range
plt.plot(np.log(lasso_cv.alphas_), mean_mse, 'k-', label='Mean MSE')
plt.fill_between(np.log(lasso_cv.alphas_),
                 mean_mse - std_mse,
                 mean_mse + std_mse,
                 alpha=0.3, label='MSE ± 1 Std. Dev.')

# Add reference lines for lambda.min and lambda.1se
plt.axvline(np.log(lambda_min), linestyle='--', color='r',
            label=f'lambda.min: {lambda_min:.4f} (log={np.log(lambda_min):.2f})')
plt.axvline(np.log(lambda_1se), linestyle='--', color='b',
            label=f'lambda.1se: {lambda_1se:.4f} (log={np.log(lambda_1se):.2f})')

# Mark minimum MSE point
plt.scatter(np.log(lambda_min), min_mse, color='r', s=100, zorder=5)

plt.xlabel('log(λ)')
plt.ylabel('Mean-Squared Error')
plt.title('LASSO Regularization Path')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'lasso_regularization_path.png'), dpi=300)
plt.show()

# Print lambda values
print(f"lambda.min: {lambda_min:.6f} (log={np.log(lambda_min):.4f})")
print(f"lambda.1se: {lambda_1se:.6f} (log={np.log(lambda_1se):.4f})")

# Use lambda.1se as final regularization parameter
best_alpha = lambda_1se
lasso = Lasso(alpha=best_alpha, max_iter=10000)
lasso.fit(X_train_resampled, y_train_resampled)

# Get non-zero coefficient features
selected_features = np.where(lasso.coef_ != 0)[0]
print(f"Using lambda.1se selected {len(selected_features)} features")

# Get selected feature names
selected_feature_names = X_train.columns[selected_features]
print("Selected features:")
for i, name in enumerate(selected_feature_names):
    print(f"{i + 1}. {name}")

# Plot LASSO coefficient path
plt.figure(figsize=(14, 10))
ax = plt.gca()

# Get all feature names
all_feature_names = X_train.columns

# Ensure alphas and coefs dimensions match
alphas_for_plot = lasso_cv.alphas_
coefs_array = np.array(coefs)

# Plot coefficient paths for all features
lines = ax.plot(alphas_for_plot, coefs_array)
ax.set_xscale('log')
ax.set_xlabel('Alpha (log scale)')
ax.set_ylabel('Coefficients')
ax.set_title('LASSO Coefficient Path')

# Find finally selected features
final_coefs = lasso.coef_
selected_mask = final_coefs != 0
selected_indices = np.where(selected_mask)[0]

# Create legend for selected features
legend_lines = []
legend_labels = []
for idx in selected_indices:
    # Bold lines for selected features
    lines[idx].set_linewidth(2)
    lines[idx].set_alpha(1.0)

    # Collect information for legend
    legend_lines.append(lines[idx])
    legend_labels.append(f"{all_feature_names[idx]} (coef: {final_coefs[idx]:.3f})")

# Add legend
ax.legend(legend_lines, legend_labels, loc='upper right', bbox_to_anchor=(1.0, 1.0),
          fontsize=10, frameon=True, fancybox=True, shadow=True)

plt.axis('tight')
plt.savefig(os.path.join(output_dir, 'lasso_coefficient_path.png'), dpi=300, bbox_inches='tight')
plt.show()

# Use selected features
X_train_selected = X_train_resampled.iloc[:, selected_features]
X_test_selected = X_test_scaled.iloc[:, selected_features]
feature_names = X_train.columns[selected_features]

# 5. Define and train multiple classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "XGBoost": XGBClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# 6. Cross-validation model evaluation
results = {}
for name, clf in classifiers.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train_selected, y_train_resampled, cv=cv, scoring='roc_auc')
    results[name] = scores
    print(f"{name}: AUC = {np.mean(scores):.3f} (±{np.std(scores):.3f})")

# 7. Evaluate all models on test set
metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
test_results = pd.DataFrame(index=classifiers.keys(), columns=metrics)

# Store predictions for each classifier
all_predictions = {}
roc_data = {}  # Store ROC curve data for each classifier

for name, clf in classifiers.items():
    clf.fit(X_train_selected, y_train_resampled)
    y_pred = clf.predict(X_test_selected)
    y_prob = clf.predict_proba(X_test_selected)[:, 1] if hasattr(clf, "predict_proba") else np.zeros_like(y_pred)

    test_results.loc[name, 'AUC'] = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    test_results.loc[name, 'Accuracy'] = accuracy_score(y_test, y_pred)
    test_results.loc[name, 'Precision'] = precision_score(y_test, y_pred)
    test_results.loc[name, 'Recall'] = recall_score(y_test, y_pred)
    test_results.loc[name, 'F1'] = f1_score(y_test, y_pred)

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

    # Save prediction results to CSV for each classifier (including patient ID)
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
    pred_df.to_csv(os.path.join(output_dir, f'{name}_predictions.csv'), index=False)

# 8. Plot radar chart for model performance comparison
def plot_radar_chart(results_df, title):
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
    plt.savefig(os.path.join(output_dir, 'model_comparison_radar.png'), dpi=300, bbox_inches='tight')
    plt.show()

plot_radar_chart(test_results.astype(float), "Model Performance Comparison on Test Set")

# 9. Plot ROC curves for all classifiers on one plot
plt.figure(figsize=(10, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(classifiers)))

for i, (name, data) in enumerate(roc_data.items()):
    plt.plot(data['fpr'], data['tpr'], color=colors[i], lw=2,
             label=f'{name} (AUC = {data["auc"]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Classifiers')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'all_classifiers_roc_curves.png'), dpi=300)
plt.close()

# 10. Select best model
best_model_name = test_results['AUC'].astype(float).idxmax()
best_model = classifiers[best_model_name]
print(f"Best model: {best_model_name}")

# 11. Plot confusion matrix and ROC curve for best model
best_model.fit(X_train_selected, y_train_resampled)
y_pred = best_model.predict(X_test_selected)
y_prob = best_model.predict_proba(X_test_selected)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(output_dir, 'best_model_confusion_matrix.png'), dpi=300)
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic - {best_model_name}')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_dir, 'best_model_roc_curve.png'), dpi=300)
plt.show()

# 12. Generate feature importance plots and beeswarm plots for each classifier
# Define feature types (imaging features and clinical features)
imaging_features = ['lacrimal_volume', 'lacrimal_major_axis', 'lacrimal_minor_axis', 'lacrimal_sphericity',
                    'maxilla_mean_thickness', 'maxilla_min_thickness', 'maxilla_max_thickness',
                    'maxilla_surface_irregularity', 'maxilla_bone_density', 'nasal_volume',
                    'affected_nasal_volume', 'affected_nasal_irregularity']

clinical_features = ['Eye', 'Sex', 'Age', 'Duration_of_symptoms', 'Severity_of_symptoms',
                     'Previous_treatment_history', 'Systemic_medical_history']

# Assign type to each feature
feature_types = []
for feature in feature_names:
    if feature in imaging_features:
        feature_types.append('imaging')
    elif feature in clinical_features:
        feature_types.append('clinical')

# Generate feature importance plots for each classifier
for name, clf in classifiers.items():
    try:
        # Get feature importance
        if hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            importance = np.abs(clf.coef_[0])
        else:
            print(f"{name} does not have feature importance or coefficients")
            continue

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'feature_type': feature_types
        })

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=True)

        # Assign different colors for different feature types
        colors = []
        for feat_type in importance_df['feature_type']:
            if feat_type == 'clinical':
                colors.append('skyblue')
            elif feat_type == 'imaging':
                colors.append('lightcoral')

        # Plot feature importance bar chart
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title(f'Feature Importance - {name}')
        plt.yticks(range(len(importance_df)), importance_df['feature'])

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor='skyblue', label='Clinical Features'),
            Patch(facecolor='lightcoral', label='Imaging Features')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_feature_importance.png'), dpi=300)
        plt.close()

        print(f"Saved feature importance plot for {name}")

    except Exception as e:
        print(f"Error generating feature importance for {name}: {e}")

# 13. SHAP feature importance analysis - only for models supporting SHAP
shap_supported_models = ["Logistic Regression", "XGBoost", "Random Forest"]

if best_model_name in shap_supported_models and hasattr(best_model, 'predict_proba'):
    try:
        explainer = shap.Explainer(best_model, X_train_selected, feature_names=feature_names)
        shap_values = explainer(X_test_selected)

        # Feature importance plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_selected, feature_names=feature_names, plot_type="bar")
        plt.savefig(os.path.join(output_dir, 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Beeswarm plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_selected, feature_names=feature_names)
        plt.savefig(os.path.join(output_dir, 'shap_beeswarm_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"SHAP analysis failed for {best_model_name}: {e}")
else:
    print(f"Skipping SHAP analysis for {best_model_name} (not supported)")

# Plot nomogram based on logistic regression
def plot_logistic_nomogram(feature_names, coefficients, intercept, output_dir, X_train_original):
    """
    Create logistic regression nomogram using original feature values
    """
    # Set figure size
    plt.figure(figsize=(16, 12))

    # Set y-axis positions (starting position for each feature)
    n_features = len(feature_names)

    # Create three subplots: feature axes, total points axis, and probability axis
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax2 = plt.subplot2grid((4, 1), (3, 0))

    # Get original value ranges for each feature
    feature_ranges = {}
    for i, feature in enumerate(feature_names):
        if feature == 'Previous_treatment_history':
            # Categorical feature: 0=No, 1=Yes
            feature_ranges[feature] = [0, 1]
        else:
            # Numerical features: use 5th and 95th percentiles of original data as range
            feature_data = X_train_original[feature]
            feature_ranges[feature] = [
                np.percentile(feature_data, 5),
                np.percentile(feature_data, 95)
            ]

    # Create point axes for each feature
    for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
        # Determine value range based on feature type
        if name == 'Previous_treatment_history':
            # Categorical feature: only two values
            values = np.array([0, 1])
            value_labels = ['No', 'Yes']
            points = coef * values
        else:
            # Numerical features: use 5 equally spaced points
            min_val, max_val = feature_ranges[name]
            values = np.linspace(min_val, max_val, 5)
            value_labels = [f"{v:.1f}" for v in values]
            points = coef * values

        # Plot point axis
        y_pos = n_features - i - 1  # Start from top
        ax1.plot(points, [y_pos] * len(values), 'b-', alpha=0.7, linewidth=2)

        # Add tick marks
        for j, (val_label, point) in enumerate(zip(value_labels, points)):
            ax1.plot([point, point], [y_pos - 0.1, y_pos + 0.1], 'k-', alpha=0.7)
            ax1.text(point, y_pos - 0.2, val_label, ha='center', va='top', fontsize=10)

        # Add feature names
        feature_label = {
            'lacrimal_volume': 'Lacrimal Volume',
            'affected_nasal_volume': 'Affected Nasal Volume',
            'Previous_treatment_history': 'Previous Treatment History'
        }.get(name, name)

        ax1.text(min(points) - 1.0, y_pos, feature_label, ha='right', va='center',
                 fontsize=12, fontweight='bold')

    # Calculate total points range
    total_points_min = 0
    total_points_max = 0
    for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
        if name == 'Previous_treatment_history':
            total_points_min += min(coef * 0, coef * 1)
            total_points_max += max(coef * 0, coef * 1)
        else:
            min_val, max_val = feature_ranges[name]
            total_points_min += coef * min_val
            total_points_max += coef * max_val

    total_points_min += intercept
    total_points_max += intercept

    # Plot total points axis
    total_points_range = np.linspace(total_points_min, total_points_max, 10)
    ax2.plot(total_points_range, [0] * len(total_points_range), 'r-', linewidth=3)

    # Add total points ticks
    for point in total_points_range[::2]:
        ax2.plot([point, point], [-0.1, 0.1], 'k-', alpha=0.7)
        ax2.text(point, 0.2, f"{point:.1f}", ha='center', va='bottom', fontsize=10)

    # Add probability axis
    probability_range = 1 / (1 + np.exp(-total_points_range))

    # Plot probability axis (above total points axis)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(probability_range, [0] * len(probability_range), 'g-', linewidth=3)

    # Add probability ticks
    for prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
        # Find closest probability point
        idx = np.argmin(np.abs(probability_range - prob))
        ax2_twin.plot([probability_range[idx], probability_range[idx]],
                      [-0.1, 0.1], 'k-', alpha=0.7)
        ax2_twin.text(probability_range[idx], 0.2, f"{prob:.1f}",
                      ha='center', va='bottom', fontsize=10)

    # Set axis labels
    ax1.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Point Contribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Total Points', fontsize=12, fontweight='bold')
    ax2_twin.set_xlabel('Surgical Difficulty Probability', fontsize=12, fontweight='bold')

    # Hide unnecessary ticks
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax2_twin.set_yticks([])

    # Set title
    ax1.set_title('Surgical Difficulty Prediction Nomogram\n(Normal = 0, Difficult = 1)', fontsize=14, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, 'surgical_difficulty_nomogram.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Print usage instructions
    print("\nNomogram Usage Instructions:")
    print("1. For each patient, find their original value on each feature axis")
    print("2. Read the corresponding point contribution value")
    print("3. Sum all point contributions, plus the intercept, to get total points")
    print("4. Find the corresponding position on the total points axis, then read the surgical difficulty probability on the probability axis")
    print("5. Probability > 0.5 predicts difficult surgery (Difficult = 1), probability ≤ 0.5 predicts normal surgery (Normal = 0)")
    print("\nFeature Description:")
    print("- lacrimal_volume: Lacrimal volume (original values)")
    print("- affected_nasal_volume: Affected nasal volume (original values)")
    print("- Previous_treatment_history: Previous treatment history (0=No, 1=Yes)")

# Get logistic regression model coefficients
if best_model_name == "Logistic Regression" and hasattr(best_model, 'coef_'):
    coefficients = best_model.coef_[0]
    intercept = best_model.intercept_[0]

    print("Logistic Regression Model Coefficients:")
    for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
        print(f"{name}: {coef:.4f}")
    print(f"Intercept: {intercept:.4f}")

    # Plot nomogram using original training data (unscaled)
    plot_logistic_nomogram(feature_names, coefficients, intercept, output_dir, X_train)
else:
    print("Best model is not Logistic Regression or coefficients not available, cannot plot nomogram")

# 14. Save results
test_results.to_csv(os.path.join(output_dir, 'model_performance_results.csv'))
print("All analyses completed. Results saved to files.")