import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
from sklearn.utils import resample


def calculate_metrics_with_ci(y_true, y_scores, y_pred, n_bootstraps=1000):
    """Calculate all performance metrics with 95% confidence intervals"""

    # First calculate all baseline metrics
    original_metrics = {}

    # AUC
    original_metrics['auc'] = roc_auc_score(y_true, y_scores)

    # Accuracy
    original_metrics['accuracy'] = (y_true == y_pred).mean()

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Sensitivity (Recall)
    original_metrics['sensitivity'] = tp / (tp + fn)

    # Specificity
    original_metrics['specificity'] = tn / (tn + fp)

    # Precision
    original_metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0

    # F1 Score
    original_metrics['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    # Bootstrap to calculate confidence intervals
    metrics_boot = {
        'auc': [], 'accuracy': [], 'sensitivity': [], 'specificity': [],
        'precision': [], 'f1': []
    }

    np.random.seed(42)
    n_samples = len(y_true)

    for i in range(n_bootstraps):
        indices = resample(np.arange(n_samples), replace=True)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]
        y_pred_boot = y_pred[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        # Calculate all metrics
        try:
            # AUC
            metrics_boot['auc'].append(roc_auc_score(y_true_boot, y_scores_boot))

            # Accuracy
            metrics_boot['accuracy'].append((y_true_boot == y_pred_boot).mean())

            # Other metrics
            tn_boot, fp_boot, fn_boot, tp_boot = confusion_matrix(y_true_boot, y_pred_boot).ravel()

            metrics_boot['sensitivity'].append(tp_boot / (tp_boot + fn_boot))
            metrics_boot['specificity'].append(tn_boot / (tn_boot + fp_boot))
            metrics_boot['precision'].append(tp_boot / (tp_boot + fp_boot) if (tp_boot + fp_boot) > 0 else 0)
            metrics_boot['f1'].append(
                2 * tp_boot / (2 * tp_boot + fp_boot + fn_boot) if (2 * tp_boot + fp_boot + fn_boot) > 0 else 0)
        except Exception as e:
            # Skip this iteration if calculation fails
            continue

    # Calculate confidence intervals and combine results
    results = {}
    alpha = 0.05

    for metric_name in original_metrics.keys():
        if metric_name in metrics_boot and len(metrics_boot[metric_name]) > 0:
            values = np.array(metrics_boot[metric_name])
            ci_lower = np.percentile(values, 100 * alpha / 2)
            ci_upper = np.percentile(values, 100 * (1 - alpha / 2))

            results[metric_name] = {
                'value': original_metrics[metric_name],  # Use pre-calculated original value
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        else:
            # If Bootstrap fails, at least return original value
            results[metric_name] = {
                'value': original_metrics[metric_name],
                'ci_lower': original_metrics[metric_name],  # No CI, use original value
                'ci_upper': original_metrics[metric_name]
            }

    return results


# Example usage
# Update this path to your actual file location
file_path = "path/to/your/test_predictions.csv"
df = pd.read_csv(file_path)

y_true = df['true_label'].values
y_scores = df['pred_prob'].values
y_pred = df['pred_label'].values

print(f"Data Overview:")
print(f"Total samples: {len(df)}")
print(f"Hard cases (true_label=1): {df['true_label'].sum()}")
print(f"Prediction probability range: {y_scores.min():.3f} - {y_scores.max():.3f}")

results = calculate_metrics_with_ci(y_true, y_scores, y_pred)

print("\n=== Complete Performance Report ===")
for metric, data in results.items():
    print(f"{metric.upper()}: {data['value']:.3f} (95% CI: {data['ci_lower']:.3f}-{data['ci_upper']:.3f})")

# Additional confusion matrix details
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"\n=== Confusion Matrix Details ===")
print(f"True Negative (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Positive (TP): {tp}")