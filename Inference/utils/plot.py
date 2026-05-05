import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path


class Plots:
    """
    Scientific-paper-ready figures saved as SVG + PDF.

    Style: serif fonts, thin rules, column-width figures, no in-figure titles
    by default (captions belong in the manuscript). `svg.fonttype='none'` keeps
    text editable in Inkscape/Illustrator.
    """

    PAPER_RC = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'CMU Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 0.7,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.color': '#b0b0b0',
        'grid.linestyle': ':',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.6,
        'axes.axisbelow': True,
        'legend.frameon': False,
        'legend.handlelength': 1.5,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
    }

    COL_SINGLE = (3.5, 2.6)
    COL_SINGLE_SQUARE = (3.5, 3.5)
    COL_DOUBLE = (7.16, 3.2)
    COL_DOUBLE_TALL = (7.16, 4.0)

    PALETTE = {
        'model1_face': '#4C72B0',
        'model1_edge': '#1f3b73',
        'model2_face': '#DD8452',
        'model2_edge': '#8a3d1a',
        'accent': '#1f3b73',
        'neutral_gray': '#555555',
        'light_gray': '#a0a0a0',
        'alpha_fill': 0.75,
    }

    def __init__(self, output_dir='plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -------- internal helpers --------

    def _new_fig(self, figsize):
        return plt.subplots(figsize=figsize)

    def _clean_spines(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.6)
        ax.spines['bottom'].set_linewidth(0.6)

    def _rotate_xticks(self, ax, labels, angle=30):
        ha = 'right' if angle not in (0, 90) else 'center'
        ax.set_xticklabels(labels, rotation=angle, ha=ha)

    def _save(self, fig, filename):
        svg_path = self.output_dir / f'{filename}.svg'
        pdf_path = self.output_dir / f'{filename}.pdf'
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        return svg_path

    def _box_style(self, model_idx):
        if model_idx == 0:
            face = self.PALETTE['model1_face']
            edge = self.PALETTE['model1_edge']
        else:
            face = self.PALETTE['model2_face']
            edge = self.PALETTE['model2_edge']
        return dict(
            boxprops=dict(facecolor=face, edgecolor=edge,
                          linewidth=0.9, alpha=self.PALETTE['alpha_fill']),
            medianprops=dict(color='black', linewidth=1.1),
            whiskerprops=dict(color=edge, linewidth=0.7),
            capprops=dict(color=edge, linewidth=0.7),
            flierprops=dict(marker='o', markersize=3,
                            markerfacecolor=face,
                            markeredgecolor=edge,
                            markeredgewidth=0.5, linestyle='none',
                            alpha=0.6),
        )

    # -------- single-model plots --------

    def plot_tot_accuracy(self, accuracy_list, labels,
                          filename='tot_accuracy',
                          xtick_rotation=30):
        with plt.rc_context(self.PAPER_RC):
            fig, ax = self._new_fig(self.COL_DOUBLE)
            x = np.arange(len(labels))
            ax.bar(x, accuracy_list, width=0.65,
                   facecolor=self.PALETTE['model1_face'],
                   edgecolor=self.PALETTE['model1_edge'],
                   linewidth=0.9, alpha=self.PALETTE['alpha_fill'])
            ax.set_xticks(x)
            self._rotate_xticks(ax, labels, angle=xtick_rotation)
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0.0, 1.0)
            self._clean_spines(ax)
            fig.tight_layout()
            return self._save(fig, filename)

    def box_plot(self, accuracy_data, dataset_names, title=None,
                 filename='box_plot'):
        with plt.rc_context(self.PAPER_RC):
            fig, ax = self._new_fig(self.COL_DOUBLE_TALL)
            style = self._box_style(0)
            ax.boxplot(accuracy_data, labels=dataset_names,
                       patch_artist=True, widths=0.55,
                       showmeans=False, **style)
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Test accuracy')
            if title:
                ax.set_title(title)
            self._rotate_xticks(ax, dataset_names, angle=30)
            self._clean_spines(ax)
            flat = [v for row in accuracy_data for v in row]
            if flat:
                lo = max(0.0, min(flat) - 0.02)
                ax.set_ylim(lo, 1.0)
            fig.tight_layout()
            return self._save(fig, filename)

    def plot_loss(self, loss, dataset_name, filename_prefix='loss'):
        with plt.rc_context(self.PAPER_RC):
            fig, ax = self._new_fig(self.COL_SINGLE)
            epochs = np.arange(1, len(loss) + 1)
            ax.plot(epochs, loss, color=self.PALETTE['model1_face'], linewidth=1.2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Training loss')
            self._clean_spines(ax)
            fig.tight_layout()
            return self._save(fig, f'{filename_prefix}_{dataset_name}')

    # -------- two-model plots --------

    def plot_compare_accuracy(self, accuracy_list1, accuracy_list2, labels,
                              model_name1: str, model_name2: str,
                              filename='compare_accuracy',
                              xtick_rotation=30):
        with plt.rc_context(self.PAPER_RC):
            fig, ax = self._new_fig(self.COL_DOUBLE)
            x = np.arange(len(labels))
            width = 0.38
            ax.bar(x - width / 2, accuracy_list1, width,
                   facecolor=self.PALETTE['model1_face'],
                   edgecolor=self.PALETTE['model1_edge'],
                   linewidth=0.9, alpha=self.PALETTE['alpha_fill'],
                   label=model_name1)
            ax.bar(x + width / 2, accuracy_list2, width,
                   facecolor=self.PALETTE['model2_face'],
                   edgecolor=self.PALETTE['model2_edge'],
                   linewidth=0.9, alpha=self.PALETTE['alpha_fill'],
                   label=model_name2)
            ax.set_xticks(x)
            self._rotate_xticks(ax, labels, angle=xtick_rotation)
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0.0, 1.0)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                      ncol=2, frameon=False, borderaxespad=0.0)
            self._clean_spines(ax)
            fig.tight_layout()
            return self._save(fig, filename)

    def plot_parity_scatter(self, accuracy_list1, accuracy_list2,
                            dataset_names,
                            model_name1: str, model_name2: str,
                            filename='parity_scatter'):
        with plt.rc_context(self.PAPER_RC):
            fig, ax = self._new_fig(self.COL_SINGLE_SQUARE)
            lo = min(min(accuracy_list1), min(accuracy_list2))
            hi = max(max(accuracy_list1), max(accuracy_list2))
            pad = max((hi - lo) * 0.05, 0.01)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    color=self.PALETTE['light_gray'], linewidth=0.7,
                    linestyle='--', label=r'$y = x$', zorder=2)
            ax.scatter(accuracy_list1, accuracy_list2, s=42,
                       facecolor=self.PALETTE['model1_face'],
                       edgecolor=self.PALETTE['model1_edge'],
                       alpha=self.PALETTE['alpha_fill'],
                       linewidth=0.9, zorder=3)
            for i, name in enumerate(dataset_names):
                ax.annotate(name, (accuracy_list1[i], accuracy_list2[i]),
                            xytext=(4, 4), textcoords='offset points',
                            fontsize=7, color=self.PALETTE['neutral_gray'])
            ax.set_xlabel(f'{model_name1} accuracy')
            ax.set_ylabel(f'{model_name2} accuracy')
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect('equal', adjustable='box')
            ax.legend(loc='upper left', frameon=False)
            self._clean_spines(ax)
            fig.tight_layout()
            return self._save(fig, filename)

    def plot_combined_boxplot(self, data1, data2, dataset_names,
                              model_name1: str, model_name2: str,
                              filename='combined_boxplot'):
        with plt.rc_context(self.PAPER_RC):
            n = len(dataset_names)
            fig, ax = self._new_fig(self.COL_DOUBLE_TALL)
            centers = np.arange(1, n + 1) * 3
            pos1 = centers - 0.7
            pos2 = centers + 0.7
            s1 = self._box_style(0)
            s2 = self._box_style(1)
            ax.boxplot(data1, positions=pos1, widths=1.0,
                       patch_artist=True, showmeans=False, **s1)
            ax.boxplot(data2, positions=pos2, widths=1.0,
                       patch_artist=True, showmeans=False, **s2)
            ax.set_xticks(centers)
            self._rotate_xticks(ax, dataset_names, angle=45)
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Test accuracy')
            flat = [v for row in (list(data1) + list(data2)) for v in row]
            if flat:
                lo = max(0.0, min(flat) - 0.02)
                ax.set_ylim(lo, 1.0)
            legend_elements = [
                Patch(facecolor=self.PALETTE['model1_face'],
                      edgecolor=self.PALETTE['model1_edge'],
                      linewidth=0.9, alpha=self.PALETTE['alpha_fill'],
                      label=model_name1),
                Patch(facecolor=self.PALETTE['model2_face'],
                      edgecolor=self.PALETTE['model2_edge'],
                      linewidth=0.9, alpha=self.PALETTE['alpha_fill'],
                      label=model_name2),
            ]
            ax.legend(handles=legend_elements,
                      loc='lower center', bbox_to_anchor=(0.5, 1.02),
                      ncol=2, frameon=False, borderaxespad=0.0)
            self._clean_spines(ax)
            fig.tight_layout()
            return self._save(fig, filename)

    def plot_metrics_bar_comparison(self, metrics1, metrics2, dataset_names,
                                    errors1, errors2,
                                    model_name1: str, model_name2: str,
                                    filename='metrics_bar_comparison'):
        """metrics{1,2}, errors{1,2}: dict {metric_name: [value_per_dataset]}."""
        keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

        with plt.rc_context(self.PAPER_RC):
            fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.6))
            axes = axes.flatten()
            x = np.arange(len(dataset_names))
            width = 0.38

            for i, (key, label) in enumerate(zip(keys, labels)):
                ax = axes[i]
                ax.bar(x - width / 2, metrics1[key], width,
                       yerr=errors1[key],
                       facecolor=self.PALETTE['model1_face'],
                       edgecolor=self.PALETTE['model1_edge'],
                       linewidth=0.7, alpha=self.PALETTE['alpha_fill'],
                       error_kw={'elinewidth': 0.7},
                       label=model_name1)
                ax.bar(x + width / 2, metrics2[key], width,
                       yerr=errors2[key],
                       facecolor=self.PALETTE['model2_face'],
                       edgecolor=self.PALETTE['model2_edge'],
                       linewidth=0.7, alpha=self.PALETTE['alpha_fill'],
                       error_kw={'elinewidth': 0.7},
                       label=model_name2)
                ax.set_title(f'({chr(97 + i)}) {label}', loc='left')
                ax.set_xticks(x)
                ax.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=7)
                ax.set_ylim(0, 1.0)
                self._clean_spines(ax)

            axes[-1].set_visible(False)

            handles = [
                Patch(facecolor=self.PALETTE['model1_face'],
                      edgecolor=self.PALETTE['model1_edge'],
                      alpha=self.PALETTE['alpha_fill'], label=model_name1),
                Patch(facecolor=self.PALETTE['model2_face'],
                      edgecolor=self.PALETTE['model2_edge'],
                      alpha=self.PALETTE['alpha_fill'], label=model_name2),
            ]
            fig.legend(handles=handles, loc='lower right', ncol=2, frameon=False)
            fig.tight_layout()
            return self._save(fig, filename)

    def plot_metrics_heatmap(self, df1, df2,
                             model_name1: str, model_name2: str,
                             filename='metrics_heatmap'):
        """df{1,2}: DataFrame index=dataset, columns=metric labels (values in [0,1])."""
        with plt.rc_context(self.PAPER_RC):
            height = max(2.6, len(df1) * 0.4 + 1.6)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, height))
            for ax, df, title in [(ax1, df1, model_name1), (ax2, df2, model_name2)]:
                sns.heatmap(df, ax=ax, annot=True, fmt='.3f', cmap='YlOrRd',
                            vmin=0, vmax=1, linewidths=0.5, linecolor='lightgray',
                            cbar_kws={'shrink': 0.8})
                ax.set_title(title)
                ax.set_xlabel('Metric')
                ax.set_ylabel('Dataset')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            fig.tight_layout()
            return self._save(fig, filename)

    def plot_comparison_accuracies(self, accs1, accs2,
                                   model_name1: str, model_name2: str,
                                   filename='comparison_accuracies'):
        """Per-fold accuracy line plot for two models."""
        with plt.rc_context(self.PAPER_RC):
            fig, ax = self._new_fig(self.COL_DOUBLE)
            folds = np.arange(1, len(accs1) + 1)
            ax.plot(folds, accs1, marker='o', linewidth=1.2, markersize=4,
                    color=self.PALETTE['model1_face'], label=model_name1)
            ax.plot(folds, accs2, marker='s', linewidth=1.2, markersize=4,
                    color=self.PALETTE['model2_face'], label=model_name2)
            ax.set_xlabel('Fold')
            ax.set_ylabel('Accuracy')
            ax.set_xticks(folds)
            ax.set_ylim(0.0, 1.0)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                      ncol=2, frameon=False, borderaxespad=0.0)
            self._clean_spines(ax)
            fig.tight_layout()
            return self._save(fig, filename)
