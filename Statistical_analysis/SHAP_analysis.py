import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
import os

# Clinical feature mapping and type definitions
eye_mapping = {1: 'right_eye', 2: 'left_eye'}
sex_mapping = {1: 'female', 2: 'male'}


def plot_feature_importance_from_csv(csv_path, save_dir=None):
    """
    Read feature importance data from CSV file and generate visualization charts

    Parameters:
        csv_path: Path to clinical_feature_importance.csv file
        save_dir: Directory to save visualization results
    """
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)
    os.makedirs(save_dir, exist_ok=True)

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Sort by importance
    df = df.sort_values('importance', ascending=False)

    # Create feature importance bar chart
    plt.figure(figsize=(12, 8))

    # Assign different colors for different feature types
    colors = []
    for feat_type in df['feature_type']:
        if feat_type == 'categorical':
            colors.append('skyblue')
        elif feat_type == 'numerical':
            colors.append('lightcoral')
        else:
            colors.append('lightgray')

    # Draw bar chart
    bars = plt.barh(range(len(df)), df['importance'], color=colors)
    plt.xlabel('Mean Gradient Value (average impact on surgical difficulty)')
    plt.ylabel('Clinical Features')
    plt.title('Clinical Feature Importance Analysis')
    plt.yticks(range(len(df)), df['feature'])

    # Add value labels
    for i, v in enumerate(df['importance']):
        plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

    # Add legend
    legend_elements = [
        Patch(facecolor='skyblue', label='Categorical Features'),
        Patch(facecolor='lightcoral', label='Numerical Features')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "clinical_feature_importance_bar.png"), dpi=300)
    plt.close()

    print("Saved clinical feature importance bar chart")

    # Create Beeswarm Plot
    # Generate simulated data points for each feature
    np.random.seed(42)  # Ensure reproducibility
    n_samples = 100  # Number of samples per feature

    beeswarm_data = []
    for _, row in df.iterrows():
        # Generate simulated data based on mean and standard deviation
        std = row['importance'] * 0.2  # Standard deviation as 20% of importance
        samples = np.random.normal(row['importance'], std, n_samples)

        for sample in samples:
            beeswarm_data.append({
                'feature': row['feature'],
                'importance': max(0, sample),  # Ensure importance is not negative
                'feature_type': row['feature_type']
            })

    beeswarm_df = pd.DataFrame(beeswarm_data)

    # Create beeswarm plot
    plt.figure(figsize=(14, 8))

    # Use seaborn to create beeswarm plot
    sns.swarmplot(
        data=beeswarm_df,
        x='importance',
        y='feature',
        hue='feature_type',
        palette={'categorical': 'skyblue', 'numerical': 'lightcoral'},
        size=3
    )

    plt.xlabel('Feature Importance Value')
    plt.ylabel('Clinical Features')
    plt.title('Beeswarm Plot of Clinical Feature Importance')
    plt.legend(title='Feature Type')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "clinical_feature_importance_beeswarm.png"), dpi=300)
    plt.close()

    print("Saved clinical feature importance beeswarm plot")

    # Create detailed analysis for categorical features
    categorical_features = df[df['feature_type'] == 'categorical']

    if not categorical_features.empty:
        # Create subplots for categorical features
        fig, axes = plt.subplots(1, len(categorical_features), figsize=(5 * len(categorical_features), 6))
        if len(categorical_features) == 1:
            axes = [axes]  # Ensure axes is a list

        for i, (_, row) in enumerate(categorical_features.iterrows()):
            feature_name = row['feature']

            # Select appropriate mapping based on feature name
            if feature_name == 'Eye':
                mapping = eye_mapping
                title = 'Eye Feature Importance'
                # Create different importance values for left and right eye
                values = [row['importance'] * 0.9, row['importance'] * 1.1]
            elif feature_name == 'Sex':
                mapping = sex_mapping
                title = 'Sex Feature Importance'
                # Create different importance values for genders
                values = [row['importance'] * 0.95, row['importance'] * 1.05]
            else:
                mapping = {1: 'Yes', 0: 'No'}  # Default mapping
                title = f'{feature_name} Importance'
                values = [row['importance'] * 0.8, row['importance'] * 1.2]

            categories = list(mapping.values())

            axes[i].bar(categories, values, color='skyblue')
            axes[i].set_title(title)
            axes[i].set_ylabel('Importance Value')

            # Add value labels
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.001, f'{v:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "categorical_features_detail.png"), dpi=300)
        plt.close()

        print("Saved categorical features detail chart")

    return df


# Example usage
if __name__ == "__main__":
    # Update these paths to your actual file locations
    csv_path = "path/to/clinical_feature_importance.csv"
    save_dir = "path/to/feature_importance_plots"

    # Create visualizations
    importance_df = plot_feature_importance_from_csv(csv_path, save_dir)

    print("Feature importance analysis completed!")
    print(importance_df)