import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LassoCV, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
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

warnings.filterwarnings('ignore')

# Set output directory
output_dir = "./ML_results_corrected"
os.makedirs(output_dir, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# 1. Load data
train_df = pd.read_csv("ML_shape_feature_train.csv")
test_df = pd.read_csv("ML_shape_feature_test.csv")

# Check data
print("Training set columns:", train_df.columns.tolist())
print("Test set columns:", test_df.columns.tolist())
print(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")

# 2. Data preprocessing
# Separate features and labels
id_columns = [col for col in train_df.columns if 'id' in col.lower() or 'patient' in col.lower()]
print("Potential ID columns:", id_columns)

# Extract patient ID column
patient_id_column = 'case_id' if 'case_id' in test_df.columns else None
if patient_id_column:
    test_patient_ids = test_df[patient_id_column].values

# Define clinical features
clinical_categorical = ['Eye', 'Sex', 'Previous_treatment_history', 'Systemic_medical_history', 'Severity_of_symptoms']
clinical_numerical = ['Age', 'Duration_of_symptoms']

# Separate features and target
X_train_full = train_df.drop(['difficulty'] + id_columns, axis=1, errors='ignore')
y_train_full = train_df['difficulty']
X_test = test_df.drop(['difficulty'] + id_columns, axis=1, errors='ignore')
y_test = test_df['difficulty']

print(f"Original class distribution - Train: {np.bincount(y_train_full)}")
print(f"Original class distribution - Test: {np.bincount(y_test)}")

# Identify imaging features (all columns that are not clinical)
all_features = X_train_full.columns.tolist()
imaging_features = [f for f in all_features if f not in clinical_categorical + clinical_numerical]

print(f"Clinical features: {len(clinical_categorical + clinical_numerical)}")
print(f"Imaging features: {len(imaging_features)}")

# 3. Create preprocessing pipeline
# Define column types
categorical_features = [f for f in clinical_categorical if f in X_train_full.columns]
numerical_features = clinical_numerical + imaging_features
numerical_features = [f for f in numerical_features if f in X_train_full.columns]

print(f"Categorical features to encode: {categorical_features}")
print(f"Numerical features to scale: {len(numerical_features)}")

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ])

# 4. Nested Cross-Validation Setup
# Outer CV for model evaluation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=36)
# Inner CV for hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=36)

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
    "XGBoost": {
        'clf': XGBClassifier(random_state=36, eval_metric='logloss'),
        'params': {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [3, 5, 7],
            'clf__learning_rate': [0.01, 0.1, 0.3]
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
outer_results = {}
best_models = {}
feature_importances = {}

for name, model_info in classifiers.items():
    print(f"\n{'='*60}")
    print(f"Training {name} with nested cross-validation")
    print(f"{'='*60}")
    
    # Create pipeline with SMOTE inside
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('feature_selector', SelectFromModel(
            LogisticRegression(penalty='l1', solver='liblinear', random_state=36, C=0.1),
            threshold='median'
        )),
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
        print(f"\n  Outer Fold {fold + 1}/5")
        
        # Split data for this fold
        X_train_fold = X_train_full.iloc[train_idx]
        y_train_fold = y_train_full.iloc[train_idx]
        X_val_fold = X_train_full.iloc[val_idx]
        y_val_fold = y_train_full.iloc[val_idx]
        
        # Fit pipeline with grid search
        grid_search.fit(X_train_fold, y_train_fold)
        
        # Get best model from this fold
        best_model_fold = grid_search.best_estimator_
        
        # Predict on validation fold
        y_pred = best_model_fold.predict(X_val_fold)
        y_prob = best_model_fold.predict_proba(X_val_fold)[:, 1]
        
        # Calculate metrics
        fold_metrics = {
            'accuracy': accuracy_score(y_val_fold, y_pred),
            'roc_auc': roc_auc_score(y_val_fold, y_prob),
            'precision': precision_score(y_val_fold, y_pred, zero_division=0),
            'recall': recall_score(y_val_fold, y_pred, zero_division=0),
            'f1': f1_score(y_val_fold, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_val_fold, y_pred),
            'mcc': matthews_corrcoef(y_val_fold, y_pred)
        }
        
        outer_fold_results.append(fold_metrics)
        outer_fold_predictions.append({
            'true': y_val_fold.values,
            'pred': y_pred,
            'prob': y_prob
        })
        
        print(f"    AUC: {fold_metrics['roc_auc']:.3f}, "
              f"Accuracy: {fold_metrics['accuracy']:.3f}, "
              f"F1: {fold_metrics['f1']:.3f}")
    
    # Aggregate results
    metrics_df = pd.DataFrame(outer_fold_results)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()
    
    outer_results[name] = {
        'mean': mean_metrics,
        'std': std_metrics,
        'all_folds': outer_fold_results,
        'predictions': outer_fold_predictions
    }
    
    print(f"\n  {name} CV Results:")
    for metric in ['roc_auc', 'accuracy', 'f1']:
        print(f"    {metric}: {mean_metrics[metric]:.3f} (±{std_metrics[metric]:.3f})")
    
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
        # Extract feature selector to get selected features
        feature_selector = best_models[name].named_steps['feature_selector']
        preprocessor_fitted = best_models[name].named_steps['preprocessor']
        
        # Get feature names after preprocessing
        num_feature_names = numerical_features
        cat_feature_names = []
        if categorical_features:
            cat_encoder = preprocessor_fitted.named_transformers_['cat']
            for i, cat in enumerate(categorical_features):
                categories = cat_encoder.categories_[i][1:]  # Skip first category (dropped)
                cat_feature_names.extend([f"{cat}_{cat_val}" for cat_val in categories])
        
        all_feature_names = num_feature_names + cat_feature_names
        
        # Get feature importances/coefficients
        if hasattr(best_models[name].named_steps['clf'], 'feature_importances_'):
            importances = best_models[name].named_steps['clf'].feature_importances_
        elif hasattr(best_models[name].named_steps['clf'], 'coef_'):
            importances = np.abs(best_models[name].named_steps['clf'].coef_[0])
        else:
            importances = None
        
        if importances is not None:
            # Get mask of selected features
            selected_mask = feature_selector.get_support()
            selected_features = [all_feature_names[i] for i in range(len(selected_mask)) if selected_mask[i]]
            selected_importances = importances[selected_mask]
            
            feature_importances[name] = pd.DataFrame({
                'feature': selected_features,
                'importance': selected_importances
            }).sort_values('importance', ascending=False)
    
    except Exception as e:
        print(f"  Could not extract feature importances: {e}")
        feature_importances[name] = None

# 7. Evaluate on test set (ONCE)
print(f"\n{'='*60}")
print("FINAL EVALUATION ON TEST SET")
print(f"{'='*60}")

test_results = pd.DataFrame(index=classifiers.keys(), 
                            columns=['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 
                                     'Balanced_Accuracy', 'MCC', 'Kappa'])

roc_data = {}
pr_data = {}
test_predictions = {}

for name in classifiers.keys():
    print(f"\nEvaluating {name} on test set...")
    
    model = best_models[name]
    
    # Predict on test set
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    test_results.loc[name, 'AUC'] = roc_auc_score(y_test, y_prob_test)
    test_results.loc[name, 'Accuracy'] = accuracy_score(y_test, y_pred_test)
    test_results.loc[name, 'Precision'] = precision_score(y_test, y_pred_test, zero_division=0)
    test_results.loc[name, 'Recall'] = recall_score(y_test, y_pred_test, zero_division=0)
    test_results.loc[name, 'F1'] = f1_score(y_test, y_pred_test, zero_division=0)
    test_results.loc[name, 'Balanced_Accuracy'] = balanced_accuracy_score(y_test, y_pred_test)
    test_results.loc[name, 'MCC'] = matthews_corrcoef(y_test, y_pred_test)
    test_results.loc[name, 'Kappa'] = cohen_kappa_score(y_test, y_pred_test)
    
    # Store ROC and PR curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    roc_auc = auc(fpr, tpr)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_test)
    pr_auc = auc(recall_curve, precision_curve)
    pr_data[name] = {'precision': precision_curve, 'recall': recall_curve, 'auc': pr_auc}
    
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
    
    print(f"  Test AUC: {test_results.loc[name, 'AUC']:.3f}, "
          f"Accuracy: {test_results.loc[name, 'Accuracy']:.3f}, "
          f"F1: {test_results.loc[name, 'F1']:.3f}")

# 8. Select best model based on cross-validation performance
# Use the mean AUC from outer CV
cv_performance = {name: outer_results[name]['mean']['roc_auc'] for name in classifiers.keys()}
best_model_name = max(cv_performance, key=cv_performance.get)
best_model = best_models[best_model_name]

print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name}")
print(f"Cross-validation AUC: {cv_performance[best_model_name]:.3f}")
print(f"Test AUC: {test_results.loc[best_model_name, 'AUC']:.3f}")
print(f"{'='*60}")

# 9. Visualization functions
def plot_roc_curves(roc_data, save_path):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(roc_data)))
    
    for i, (name, data) in enumerate(roc_data.items()):
        plt.plot(data['fpr'], data['tpr'], color=colors[i], lw=2,
                 label=f'{name} (AUC = {data["auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves on Test Set')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curves(pr_data, save_path):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(pr_data)))
    
    for i, (name, data) in enumerate(pr_data.items()):
        plt.plot(data['recall'], data['precision'], color=colors[i], lw=2,
                 label=f'{name} (AUC = {data["auc"]:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves on Test Set')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_for_model(y_true, y_pred, model_name, save_path):
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

def plot_feature_importance(feature_importances_df, model_name, save_path):
    if feature_importances_df is None or len(feature_importances_df) == 0:
        return
    
    # Take top 20 features
    top_features = feature_importances_df.head(20)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    
    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title(f'Top 20 Feature Importance - {model_name}')
    plt.yticks(range(len(top_features)), top_features['feature'])
    
    # Add value labels
    for i, v in enumerate(top_features['importance']):
        plt.text(v, i, f' {v:.3f}', va='center')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_calibration_curve(y_true, y_prob, model_name, save_path, n_bins=10):
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{model_name}")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 10. Generate all visualizations
print("\nGenerating visualizations...")

# ROC curves
plot_roc_curves(roc_data, os.path.join(output_dir, 'all_models_roc_curves.png'))

# PR curves
plot_pr_curves(pr_data, os.path.join(output_dir, 'all_models_pr_curves.png'))

# Confusion matrix for best model
best_preds = test_predictions[best_model_name]
plot_confusion_matrix_for_model(
    best_preds['true'], best_preds['pred'], 
    best_model_name, 
    os.path.join(output_dir, f'best_model_confusion_matrix.png')
)

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

# Calibration curve for best model
plot_calibration_curve(
    best_preds['true'], best_preds['prob'],
    best_model_name,
    os.path.join(output_dir, f'best_model_calibration_curve.png')
)

# 11. SHAP analysis for interpretable models
print("\nPerforming SHAP analysis for interpretable models...")

shap_supported_models = ["Logistic Regression", "XGBoost", "Random Forest"]

for model_name in shap_supported_models:
    if model_name in best_models:
        try:
            print(f"  Computing SHAP values for {model_name}...")
            
            # Get the preprocessed test data
            model = best_models[model_name]
            preprocessor = model.named_steps['preprocessor']
            feature_selector = model.named_steps['feature_selector']
            
            # Transform test data
            X_test_transformed = preprocessor.transform(X_test)
            X_test_selected = feature_selector.transform(X_test_transformed)
            
            # Get selected feature names
            selected_mask = feature_selector.get_support()
            num_feature_names = numerical_features
            cat_feature_names = []
            
            if categorical_features:
                cat_encoder = preprocessor.named_transformers_['cat']
                for i, cat in enumerate(categorical_features):
                    categories = cat_encoder.categories_[i][1:]
                    cat_feature_names.extend([f"{cat}_{cat_val}" for cat_val in categories])
            
            all_feature_names = num_feature_names + cat_feature_names
            selected_feature_names = [all_feature_names[i] for i in range(len(selected_mask)) 
                                     if selected_mask[i]]
            
            # Create SHAP explainer
            if model_name == "Logistic Regression":
                explainer = shap.LinearExplainer(
                    model.named_steps['clf'], 
                    X_test_selected,
                    feature_names=selected_feature_names
                )
            else:
                explainer = shap.TreeExplainer(
                    model.named_steps['clf'],
                    feature_names=selected_feature_names
                )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test_selected)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test_selected, 
                             feature_names=selected_feature_names,
                             show=False)
            plt.title(f"SHAP Summary - {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_shap_summary.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Bar plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_selected, 
                             feature_names=selected_feature_names,
                             plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance - {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_shap_importance.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    SHAP analysis completed for {model_name}")
            
        except Exception as e:
            print(f"    SHAP analysis failed for {model_name}: {e}")

# 12. Statistical comparison of models
print("\nPerforming statistical comparison of models...")

# Compare AUC scores using DeLong test (simplified version)
def compare_models_delong(model1_preds, model2_preds, y_true):
    """Simplified model comparison using bootstrap"""
    n_bootstraps = 1000
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
    
    # Calculate confidence interval and p-value
    auc_diffs = np.array(auc_diffs)
    ci_lower = np.percentile(auc_diffs, 2.5)
    ci_upper = np.percentile(auc_diffs, 97.5)
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
        
        comparison = compare_models_delong(preds1, preds2, y_test.values)
        model_comparisons[f"{model1}_vs_{model2}"] = comparison
        
        print(f"  {model1} vs {model2}:")
        print(f"    AUC difference: {comparison['mean_difference']:.3f}")
        print(f"    95% CI: [{comparison['ci_95'][0]:.3f}, {comparison['ci_95'][1]:.3f}]")
        print(f"    p-value: {comparison['p_value']:.3f}")

# 13. Save all results
print("\nSaving all results...")

# Save test results
test_results.to_csv(os.path.join(output_dir, 'model_test_performance.csv'))

# Save cross-validation results
cv_results = {}
for name in classifiers.keys():
    cv_results[name] = {
        'mean_metrics': outer_results[name]['mean'].to_dict(),
        'std_metrics': outer_results[name]['std'].to_dict()
    }

with open(os.path.join(output_dir, 'cross_validation_results.json'), 'w') as f:
    json.dump(cv_results, f, indent=2)

# Save model comparisons
with open(os.path.join(output_dir, 'model_comparisons.json'), 'w') as f:
    json.dump(model_comparisons, f, indent=2)

# Save configuration
config = {
    'preprocessing': {
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'imaging_features': imaging_features
    },
    'cross_validation': {
        'outer_folds': 5,
        'inner_folds': 3,
        'random_state': 36
    },
    'models_tested': list(classifiers.keys())
}

with open(os.path.join(output_dir, 'experiment_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

# 14. Create summary report
print(f"\n{'='*60}")
print("EXPERIMENT SUMMARY")
print(f"{'='*60}")

print(f"\nBest Model: {best_model_name}")
print(f"Test Performance:")
for metric in ['AUC', 'Accuracy', 'F1', 'Balanced_Accuracy', 'MCC']:
    print(f"  {metric}: {test_results.loc[best_model_name, metric]:.3f}")

print(f"\nAll Models Test Performance:")
print(test_results.round(3).to_string())

print(f"\nResults saved to: {output_dir}")
print(f"{'='*60}")
print("Analysis completed successfully!")
