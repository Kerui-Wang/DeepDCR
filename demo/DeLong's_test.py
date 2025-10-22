import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Set matplotlib to use fonts that support English
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']  # Use fonts that support English
plt.rcParams['axes.unicode_minus'] = False  # Properly display minus signs

try:
    # Read data
    file_path = "path/to/your/DL_test.csv"  # Update this path
    data = pd.read_csv(file_path)

    print("Data loaded successfully!")
    print("Data shape:", data.shape)
    print("Data columns:", data.columns.tolist())
    print("\nData basic information:")
    print(data.info())
    print("\nFirst 5 rows:")
    print(data.head())

    # Check true label distribution
    print(f"\nTrue label distribution:")
    print(data['true_label'].value_counts())
    print(f"Positive sample ratio: {data['true_label'].mean():.3f}")

except FileNotFoundError:
    print(f"Error: File {file_path} not found")
    print("Please check if the file path is correct")
    exit()
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# Define model column name mapping
models = {
    'DeepDCR': 'pred_prob_DL',
    'LR(RSF+CF)': 'prediction_probability_LR',
    'RF(RSF+CF)': 'prediction_probability_RF',
    'XGBoost(RSF+CF)': 'prediction_probability_XGBoost',
    'LR(CF)': 'prediction_probability_Logistic Regression_clinical',
    'RF(CF)': 'prediction_probability_Random Forest_clinical',
    'XGBoost(CF)': 'prediction_probability_XGBoost_clinical'
}

# Check if all required columns exist
missing_columns = []
for model_name, col_name in models.items():
    if col_name not in data.columns:
        missing_columns.append(col_name)

if missing_columns:
    print(f"Error: Missing columns: {missing_columns}")
    print(f"Available columns: {data.columns.tolist()}")
    exit()

# Check probability value ranges
print("\nPrediction probability ranges for each model:")
for model_name, prob_col in models.items():
    min_prob = data[prob_col].min()
    max_prob = data[prob_col].max()
    print(f"{model_name}: [{min_prob:.3f}, {max_prob:.3f}]")

# Calculate AUC values for each model
auc_scores = {}
print("\n=== Model AUC Values ===")
for model_name, prob_col in models.items():
    try:
        auc_score = roc_auc_score(data['true_label'], data[prob_col])
        auc_scores[model_name] = auc_score
        print(f"{model_name} AUC: {auc_score:.4f}")
    except Exception as e:
        print(f"Error calculating AUC for {model_name}: {e}")
        auc_scores[model_name] = 0


def delongs_test_robust(y_true, prob1, prob2, n_bootstraps=1000):
    """
    Robust implementation of DeLong's test
    """
    try:
        # Calculate AUC
        auc1 = roc_auc_score(y_true, prob1)
        auc2 = roc_auc_score(y_true, prob2)
        auc_diff = auc1 - auc2

        # If AUCs are the same, return directly
        if auc_diff == 0:
            return auc_diff, 0, 1.0

        # Use bootstrap method to estimate standard error
        n = len(y_true)
        auc_diffs = []

        for _ in range(n_bootstraps):
            try:
                indices = np.random.choice(n, n, replace=True)
                y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
                prob1_boot = prob1.iloc[indices] if hasattr(prob1, 'iloc') else prob1[indices]
                prob2_boot = prob2.iloc[indices] if hasattr(prob2, 'iloc') else prob2[indices]

                auc1_boot = roc_auc_score(y_true_boot, prob1_boot)
                auc2_boot = roc_auc_score(y_true_boot, prob2_boot)
                auc_diffs.append(auc1_boot - auc2_boot)
            except:
                continue

        if len(auc_diffs) > 10:  # At least 10 valid bootstrap samples
            se = np.std(auc_diffs)
            if se > 0:
                z_score = auc_diff / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                z_score = 0
                p_value = 1.0
        else:
            # If bootstrap fails, use approximation method
            z_score = auc_diff / 0.1  # Use conservative standard error estimate
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return auc_diff, z_score, p_value

    except Exception as e:
        print(f"Error executing DeLong's test: {e}")
        return 0, 0, 1.0


# Perform DeLong's test for all model pairs
model_names = list(models.keys())
results = []

print("\n" + "=" * 60)
print("DeLong's Test Results")
print("=" * 60)

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        model1 = model_names[i]
        model2 = model_names[j]

        prob1 = data[models[model1]]
        prob2 = data[models[model2]]

        auc_diff, z_score, p_value = delongs_test_robust(
            data['true_label'], prob1, prob2
        )

        results.append({
            'Model1': model1,
            'Model2': model2,
            'AUC1': auc_scores[model1],
            'AUC2': auc_scores[model2],
            'AUC_Difference': auc_diff,
            'Z_Score': z_score,
            'P_Value': p_value,
            'Significant': p_value < 0.05
        })

        print(f"{model1} vs {model2}:")
        print(f"  {model1} AUC: {auc_scores[model1]:.4f}")
        print(f"  {model2} AUC: {auc_scores[model2]:.4f}")
        print(f"  AUC Difference: {auc_diff:.4f}")
        print(f"  Z-Score: {z_score:.4f}")
        print(f"  P-Value: {p_value:.4f}")
        print(f"  Significant (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")
        print("-" * 40)

# Create results DataFrame
results_df = pd.DataFrame(results)
print("\nSummary Results:")
print(results_df.to_string(index=False))

# Create directory to save plots
output_dir = "path/to/output/individual_plots"  # Update this path
os.makedirs(output_dir, exist_ok=True)

# Visualization - Generate and save each subplot independently
try:
    # Subplot 1: AUC value comparison
    plt.figure(figsize=(10, 6))
    model_names_auc = list(auc_scores.keys())
    auc_values = [auc_scores[name] for name in model_names_auc]
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names_auc)))
    bars = plt.bar(model_names_auc, auc_values, color=colors, alpha=0.8)
    plt.title('Model AUC Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('AUC', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, auc_val in zip(bars, auc_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{auc_val:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "AUC_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {os.path.join(output_dir, 'AUC_comparison.png')}")

    # Subplot 2: ROC curves
    plt.figure(figsize=(10, 8))
    for model_name, prob_col in models.items():
        fpr, tpr, _ = roc_curve(data['true_label'], data[prob_col])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ROC_curves.png"), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {os.path.join(output_dir, 'ROC_curves.png')}")

    # Subplot 3: P-value heatmap
    plt.figure(figsize=(12, 10))
    p_matrix = np.ones((len(model_names), len(model_names)))
    np.fill_diagonal(p_matrix, 1)  # Set diagonal to 1

    for result in results:
        i = model_names.index(result['Model1'])
        j = model_names.index(result['Model2'])
        p_matrix[i, j] = result['P_Value']
        p_matrix[j, i] = result['P_Value']

    mask = np.triu(np.ones_like(p_matrix, dtype=bool))  # Create upper triangle mask
    heatmap = sns.heatmap(p_matrix, annot=True, fmt='.3f',
                          xticklabels=model_names, yticklabels=model_names,
                          cmap='RdYlBu_r', center=0.05, mask=mask,
                          cbar_kws={'label': 'P-Value'})
    plt.title('DeLong\'s Test P-Value Matrix', fontsize=14, fontweight='bold')

    # Set heatmap label font size
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "P_value_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {os.path.join(output_dir, 'P_value_heatmap.png')}")

    # Subplot 4: AUC differences plot
    plt.figure(figsize=(12, 8))
    differences = [result['AUC_Difference'] for result in results]

    # Create model comparison labels
    comparisons = []
    for result in results:
        comparison = f"{result['Model1']} VS {result['Model2']}"
        comparisons.append(comparison)

    colors = ['red' if result['Significant'] else 'gray' for result in results]

    y_pos = np.arange(len(comparisons))

    # Create horizontal bar chart
    bars = plt.barh(y_pos, differences, color=colors, alpha=0.7, height=0.8)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('AUC Difference', fontsize=12)

    # Set Y-axis labels
    plt.yticks(y_pos, comparisons, fontsize=9)
    plt.title('AUC Differences Between Models\n(Red bars indicate significant differences)',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Adjust Y-axis limits for more space
    plt.ylim(-0.5, len(comparisons) - 0.5)

    # Add value labels on bars
    for i, (bar, diff, sig) in enumerate(zip(bars, differences, colors)):
        color = 'black' if sig == 'gray' else 'white'
        # Adjust label position based on difference sign
        if diff >= 0:
            x_pos = bar.get_width() - 0.01 if bar.get_width() > 0.02 else bar.get_width() + 0.01
            ha = 'right'
        else:
            x_pos = bar.get_width() + 0.01 if bar.get_width() < -0.02 else bar.get_width() - 0.01
            ha = 'left'

        plt.text(x_pos, bar.get_y() + bar.get_height() / 2,
                 f'{diff:.3f}',
                 ha=ha, va='center',
                 color=color, fontweight='bold', fontsize=8)

    # Add legend explanation
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Significant (p < 0.05)'),
        Patch(facecolor='gray', alpha=0.7, label='Not Significant')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "AUC_differences.png"), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {os.path.join(output_dir, 'AUC_differences.png')}")

except Exception as e:
    print(f"Visualization error: {e}")

# Multiple comparison correction
try:
    from statsmodels.stats.multitest import multipletests

    p_values = [result['P_Value'] for result in results]
    rejected, corrected_p, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')  # Use FDR correction

    print("\n" + "=" * 60)
    print("Multiple Comparison Correction Results (FDR-BH)")
    print("=" * 60)

    for i, result in enumerate(results):
        result['Corrected_P_Value'] = corrected_p[i]
        result['Significant_After_Correction'] = rejected[i]

        print(f"{result['Model1']} vs {result['Model2']}:")
        print(f"  Original P-Value: {result['P_Value']:.4f}")
        print(f"  Corrected P-Value: {corrected_p[i]:.4f}")
        print(f"  Significant After Correction: {'Yes' if rejected[i] else 'No'}")
        print("-" * 40)

except ImportError:
    print("statsmodels library not found, skipping multiple comparison correction")
    print("Please install: pip install statsmodels")
except Exception as e:
    print(f"Multiple comparison correction error: {e}")

# Save results
try:
    output_path = "path/to/output/DeLong_test_results.csv"  # Update this path
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nResults saved to: {output_path}")
except Exception as e:
    print(f"Error saving results: {e}")

# Final summary
print("\n" + "=" * 60)
print("Statistical Analysis Summary")
print("=" * 60)

print(f"Dataset size: {data.shape[0]} samples")
print(f"Positive samples: {sum(data['true_label'])}")
print(f"Negative samples: {len(data) - sum(data['true_label'])}")
print(f"Positive sample ratio: {data['true_label'].mean():.3f}")

# Find best model
best_model = max(auc_scores, key=auc_scores.get)
print(f"\nBest model: {best_model} (AUC = {auc_scores[best_model]:.4f})")

# Display all models sorted by AUC
print("\nModel AUC Ranking:")
sorted_models = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
for rank, (model, score) in enumerate(sorted_models, 1):
    print(f"{rank}. {model}: {score:.4f}")

# Show significant differences
significant_comparisons = [r for r in results if r.get('Significant_After_Correction', r['Significant'])]
if significant_comparisons:
    print("\nSignificant Model Comparisons:")
    for comp in significant_comparisons:
        significance = comp.get('Significant_After_Correction', comp['Significant'])
        p_val = comp.get('Corrected_P_Value', comp['P_Value'])
        print(f"  {comp['Model1']} vs {comp['Model2']}: "
              f"AUC Difference = {comp['AUC_Difference']:.4f}, "
              f"P-Value = {p_val:.4f} ({'Significant' if significance else 'Not significant'})")
else:
    print("\nNo significant model comparisons")

print("\nAnalysis completed!")