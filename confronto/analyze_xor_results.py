import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.table import Table

# Original descriptions (corrected English)
descriptions = {
    1: "2-layer network. Second Ising layer sees energies + original thetas",
    2: "2-layer network. Second Ising layer sees ONLY energies",
    3: "2-layer network. Second Ising layer sees energies + original thetas. No linear layer in between",
    4: "2-layer network. Second Ising layer sees energies + original thetas. Ising node dimension = 8",
    5: "1-layer network. 10 Ising nodes in parallel",
    6: "1-layer network. 10 Ising nodes in parallel. Ising node dimension = 8",
    7: "1-layer network. 5 Ising nodes in parallel. Ising node dimension = 8"
}

# Accuracies data (6 dimensions: 1D to 6D XOR)
accs_data = {
    1: [0.988, 0.975, 0.9, 0.941, 0.879, 0.88],
    2: [0.988, 0.588, 0.503, 0.523, 0.5, 0.5],
    3: [0.988, 0.956, 0.85, 0.861, 0.887, 0.853],
    4: [1, 0.975, 0.897, 0.48, 0.491, 0.502],
    5: [1, 0.956, 0.944, 0.908, 0.87, 0.84],
    6: [1, 0.925, 0.834, 0.872, 0.552, 0.730],
    7: [1, 0.938, 0.678, 0.852, 0.534, 0.682]
}

# Colors for each test
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# 1. Create table with test_n and descriptions
def create_table():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for test_n in sorted(descriptions.keys()):
        table_data.append([f"Test {test_n}", descriptions[test_n]])

    table = ax.table(cellText=table_data,
                     colLabels=['Test', 'Description'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.12, 0.88])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        else:
            if j == 0:
                cell.set_text_props(weight='bold')
            cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')

    plt.title('XOR Test Configurations', fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/root/global/utils/teo/confronto/table_tests.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Table saved: table_tests.png")

# 2. Create mean accuracy comparison across all dimensions
def create_mean_comparison():
    mean_accs = {}
    for test_n, accs in accs_data.items():
        mean_accs[test_n] = np.mean(accs)

    # Sort by mean accuracy descending
    sorted_tests = sorted(mean_accs.items(), key=lambda x: x[1], reverse=True)
    test_labels = [f"Test {t}" for t, _ in sorted_tests]
    mean_values = [acc for _, acc in sorted_tests]
    test_colors = [colors[t-1] for t, _ in sorted_tests]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(test_labels, mean_values, color=test_colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, weight='bold')

    ax.set_ylabel('Mean Accuracy', fontsize=12, weight='bold')
    ax.set_xlabel('Test', fontsize=12, weight='bold')
    ax.set_title('Mean Accuracy Across All Dimensions (1D to 6D XOR)', fontsize=14, weight='bold', pad=15)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.xticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('/root/global/utils/teo/confronto/mean_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Mean comparison saved: mean_accuracy_comparison.png")

# 3. Create individual graphs for each dimension
def create_dimension_graphs():
    for dim_idx in range(6):
        dimension = dim_idx + 1

        # Extract accuracies for this dimension
        dim_accs = {}
        for test_n, accs in accs_data.items():
            dim_accs[test_n] = accs[dim_idx]

        # Sort by accuracy descending
        sorted_tests = sorted(dim_accs.items(), key=lambda x: x[1], reverse=True)
        test_labels = [f"Test {t}" for t, _ in sorted_tests]
        acc_values = [acc for _, acc in sorted_tests]
        test_colors = [colors[t-1] for t, _ in sorted_tests]

        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(test_labels, acc_values, color=test_colors, alpha=0.8, edgecolor='black', linewidth=1.2)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, weight='bold')

        ax.set_ylabel('Accuracy', fontsize=12, weight='bold')
        ax.set_xlabel('Test (sorted by accuracy)', fontsize=12, weight='bold')
        ax.set_title(f'{dimension}D XOR - Accuracy Comparison', fontsize=14, weight='bold', pad=15)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.xticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(f'/root/global/utils/teo/confronto/accuracy_dim_{dimension}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ {dimension}D graph saved: accuracy_dim_{dimension}.png")

# Execute all visualizations
if __name__ == "__main__":
    print("Creating XOR test analysis visualizations...")
    print()
    create_table()
    create_mean_comparison()
    create_dimension_graphs()
    print()
    print("All visualizations created successfully!")
