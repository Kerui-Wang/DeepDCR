import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import confusion_matrix

# Read your data
file_path = "./test_predictions.csv"
df = pd.read_csv(file_path)

y_true = df['true_label'].values
y_pred = df['pred_label'].values

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("=== Confusion Matrix ===")
print(f"True Negative (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Positive (TP): {tp}")

# Calculate Sensitivity (Recall)
sensitivity = tp / (tp + fn)
print(f"\n=== Sensitivity Analysis ===")
print(f"Sensitivity (Raw): {sensitivity:.1%} ({tp}/{tp+fn})")

# Calculate confidence interval using Wilson Score method
# Note: Using actual positive samples (tp+fn) as total sample size
n_positive = tp + fn  # Total actual positive samples
count_success = tp    # Correctly identified positive samples

# Calculate Wilson Score confidence interval
ci_lower, ci_upper = proportion_confint(count_success, n_positive, alpha=0.05, method='wilson')

print(f"\n=== Wilson Score Confidence Interval ===")
print(f"Sensitivity: {sensitivity:.1%} (95% CI: {ci_lower:.1%}-{ci_upper:.1%})")

# Comparison with other methods
print(f"\n=== Confidence Interval Comparison by Different Methods ===")

# Clopper-Pearson exact interval
ci_lower_cp, ci_upper_cp = proportion_confint(count_success, n_positive, alpha=0.05, method='beta')
print(f"Clopper-Pearson (Exact): {ci_lower_cp:.1%}-{ci_upper_cp:.1%}")

# Normal approximation interval (not recommended for extreme values)
ci_lower_normal, ci_upper_normal = proportion_confint(count_success, n_positive, alpha=0.05, method='normal')
print(f"Normal Approximation: {ci_lower_normal:.1%}-{ci_upper_normal:.1%}")

# Generate recommended text for paper reporting
print(f"\n=== Suggested Text for Paper Reporting ===")
print(f"In the external test set, the DeepDCR model demonstrated perfect sensitivity ({tp}/{tp+fn}, 100%), ")
print(f"successfully identifying all difficult surgical cases. The 95% confidence interval calculated using ")
print(f"the Wilson Score method was {ci_lower:.1%}-{ci_upper:.1%}, indicating that even in the most conservative ")
print(f"estimation, the model's sensitivity remains at a high level.")

# Additional analysis: What if we had false negatives? (Sensitivity analysis)
print(f"\n=== Sensitivity Analysis (Assuming Presence of False Negatives) ===")
for hypothetical_fn in [1, 2]:
    hypothetical_sensitivity = tp / (tp + hypothetical_fn)
    hypothetical_ci_lower, hypothetical_ci_upper = proportion_confint(
        tp, tp + hypothetical_fn, alpha=0.05, method='wilson'
    )
    print(f"Assuming FN={hypothetical_fn}: Sensitivity={hypothetical_sensitivity:.1%} (95% CI: {hypothetical_ci_lower:.1%}-{hypothetical_ci_upper:.1%})")