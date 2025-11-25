import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix


# Set paper style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True

# Set seaborn style for scientific plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

class Plot:

    def __init__(self, output_dir='plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color palette for scientific plots
        self.colors = {
            'primary': '#2E86AB',    # blue
            'secondary': '#A23B72',  # magenta
            'tertiary': '#F18F01',   # orange
            'quaternary': '#C73E1D', # red
            'success': '#6A994E',    # green
            'neutral': '#606060'     # gray
        }

    def plot_loss_accuracy(self, loss, accuracy, dataset_name):
        """
        Creates a plot with two subplots: loss and accuracy per epochs.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        epochs = range(1, len(loss) + 1)

        # Subplot loss (left)
        ax1.plot(epochs, loss, color=self.colors['primary'], linewidth=2)
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('(a) Training Loss', loc='left', fontweight='bold')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Subplot accuracy (right)
        ax2.plot(epochs, accuracy, color=self.colors['success'], linewidth=2)
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Accuracy', fontweight='bold')
        ax2.set_title('(b) Validation Accuracy', loc='left', fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Title
        fig.suptitle(f'Training Metrics - {dataset_name}', fontweight='bold', y=1.02)

        plt.tight_layout()
        output_path = self.output_dir / f'loss_accuracy_{dataset_name}.png'
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    def plot_tot_accuracy(self, accuracy_list, labels):
        """
        Creates a bar chart of accuracies.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        x_pos = np.arange(len(labels))

        bars = ax.bar(x_pos, accuracy_list, color=self.colors['primary'],
                     alpha=0.85, edgecolor='black', linewidth=1.2)

        # Add value labels on top of bars
        for bar, val in zip(bars, accuracy_list):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title('Model Accuracy Across Datasets', fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, max(accuracy_list) * 1.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        output_path = self.output_dir / 'tot_accuracy.png'
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    def plot_compare_accuracy(self, accuracy_list1, accuracy_list2, labels):
        """
        Creates a bar chart comparing two accuracy lists.
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        x_pos = np.arange(len(labels))
        width = 0.35

        ax.bar(x_pos - width/2, accuracy_list1, width, label='Model 1',
               color=self.colors['primary'], alpha=0.85, edgecolor='black', linewidth=1.2)
        ax.bar(x_pos + width/2, accuracy_list2, width, label='Model 2',
               color=self.colors['secondary'], alpha=0.85, edgecolor='black', linewidth=1.2)

        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title('Comparative Model Performance', fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(frameon=True, shadow=True, loc='best')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(max(accuracy_list1), max(accuracy_list2)) * 1.15)

        plt.tight_layout()
        output_path = self.output_dir / 'compare_accuracy.png'
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    def plot_parity_scatter(self, accuracy_list1, accuracy_list2, dataset_names):
        """
        Creates a parity scatter plot comparing two accuracy lists.
        """
        fig, ax = plt.subplots(figsize=(7, 7))

        # Scatter plot
        ax.scatter(accuracy_list1, accuracy_list2, s=120, alpha=0.7,
                  color=self.colors['primary'], edgecolors='black', linewidth=1.5, zorder=3)

        # Labels for each point
        for i, name in enumerate(dataset_names):
            ax.annotate(name, (accuracy_list1[i], accuracy_list2[i]),
                       xytext=(7, 7), textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='gray', alpha=0.7))

        # Diagonal dashed line (y=x)
        min_val = min(min(accuracy_list1), min(accuracy_list2))
        max_val = max(max(accuracy_list1), max(accuracy_list2))
        padding = (max_val - min_val) * 0.05
        ax.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding],
               'k--', linewidth=2, label='Perfect Parity', zorder=2)

        ax.set_xlabel('Model 1 Accuracy', fontweight='bold')
        ax.set_ylabel('Model 2 Accuracy', fontweight='bold')
        ax.set_title('Parity Plot - Model Comparison', fontweight='bold', pad=15)
        ax.legend(frameon=True, shadow=True, loc='lower right')
        ax.set_aspect('equal', adjustable='box')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(min_val - padding, max_val + padding)
        ax.set_ylim(min_val - padding, max_val + padding)

        plt.tight_layout()
        output_path = self.output_dir / 'parity_scatter.png'
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    def box_plot(self, accuracy_data, dataset_names, title='Accuracy Distribution Across Datasets',
                 filename='box_plot', box_color='primary'):
        """
        Creates a box plot of accuracies for each dataset.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Box plot
        ax.boxplot(accuracy_data, labels=dataset_names, patch_artist=True,
                   showmeans=True, meanline=False,
                   meanprops=dict(marker='D', markerfacecolor=self.colors['quaternary'],
                                 markeredgecolor='black', markersize=7),
                   medianprops=dict(color=self.colors['neutral'], linewidth=2.5),
                   boxprops=dict(facecolor=self.colors[box_color], alpha=0.7,
                                edgecolor='black', linewidth=1.5),
                   whiskerprops=dict(color='black', linewidth=1.5),
                   capprops=dict(color='black', linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor='red', markersize=6,
                                  markeredgecolor='black', alpha=0.6))

        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15, fontsize=13)
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, 1.05)

        # Add legend
        legend_elements = [
            Patch(facecolor=self.colors[box_color], alpha=0.7, edgecolor='black',
                  label='Interquartile Range (IQR)'),
            plt.Line2D([0], [0], color=self.colors['neutral'], linewidth=2.5, label='Median'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=self.colors['quaternary'],
                      markersize=7, markeredgecolor='black', label='Mean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                      markersize=6, markeredgecolor='black', label='Outliers')
        ]
        ax.legend(handles=legend_elements, loc='lower left', frameon=True, shadow=True)

        plt.tight_layout()
        output_path = self.output_dir / f'{filename}.png'
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, labels=None, save_path=None):
        """Create and save a confusion matrix"""
        plt.style.use('classic')
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Greys', cbar=False, ax=ax,
            annot_kws={'size': 14, 'weight': 'bold'}, linewidths=0.5, linecolor='black'
        )
        ax.set_xlabel('Predicted label', fontsize=14, fontname='serif')
        ax.set_ylabel('True label', fontsize=14, fontname='serif')
        ax.set_xticklabels(labels, fontsize=12, fontname='serif')
        ax.set_yticklabels(labels, fontsize=12, fontname='serif', rotation=0)
        ax.set_title('Confusion Matrix', fontsize=16, fontname='serif', pad=12)
        plt.tight_layout()
        output_path = self.output_dir / f'confusion_matrix.png'
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')