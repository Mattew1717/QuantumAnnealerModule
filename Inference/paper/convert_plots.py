"""Regenerate paper-ready SVG figures from results stored in plots_old/.

Reads the per-fold UCI CSV (produced by the patched test_datasetsUCI.py) and
the single-run XOR CSVs, then delegates all rendering to PlotPaper.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from Inference.utils.plot import PlotPaper


def _load_per_fold_accuracy(csv_path: Path):
    """Return {model: (dataset_names, [[fold_acc, ...], ...])} from a long CSV."""
    df = pd.read_csv(csv_path)
    out = {}
    for model, g in df.groupby('model'):
        ds_names = sorted(g['dataset'].unique())
        vals = [
            g[g['dataset'] == d].sort_values('fold')['accuracy'].tolist()
            for d in ds_names
        ]
        out[model] = (ds_names, vals)
    return out


def _load_summary_means(csv_path: Path):
    """Return {model: (dataset_names, [accuracy_mean, ...])} from the summary CSV."""
    df = pd.read_csv(csv_path)
    out = {}
    for model, g in df.groupby('model'):
        g = g.sort_values('dataset')
        out[model] = (g['dataset'].tolist(), g['accuracy_mean'].tolist())
    return out


def _load_xor_single(csv_path: Path):
    """Return (dataset_names, accuracies) from a single-run XOR CSV."""
    df = pd.read_csv(csv_path).sort_values('dataset')
    return df['dataset'].tolist(), df['accuracy'].tolist()


def _check_aligned(a, b, what):
    if a != b:
        raise ValueError(f'{what} mismatch: {a} vs {b}')


def main():
    default_root = Path(__file__).resolve().parent / 'plots_old'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--uci-dir', type=Path, default=default_root / 'UCI_Plots')
    ap.add_argument('--xor-fullising-dir', type=Path,
                    default=default_root / 'Xor_FullIising')
    ap.add_argument('--xor-1l-dir', type=Path,
                    default=default_root / 'plots_xor_1L_20260419_130519')
    ap.add_argument('--out-dir', type=Path,
                    default=Path(__file__).resolve().parent / 'plots_paper_svg')
    args = ap.parse_args()

    pp = PlotPaper(output_dir=str(args.out_dir))

    # --- UCI boxplots & parity scatter ---
    per_fold = _load_per_fold_accuracy(args.uci_dir / 'metrics_per_fold.csv')
    ds_fi, fi_vals = per_fold['FullIsingModule']
    ds_nn, nn_vals = per_fold['NeuralNet']
    _check_aligned(ds_fi, ds_nn, 'UCI dataset ordering')

    pp.box_plot(fi_vals, ds_fi, filename='uci_boxplot_fullising')
    pp.box_plot(nn_vals, ds_nn, filename='uci_boxplot_multiising')
    pp.plot_combined_boxplot(
        fi_vals, nn_vals, ds_fi,
        model_name1='FullIsingModule',
        model_name2='MultiIsingNetwork',
        filename='uci_combined_boxplot',
    )

    summary = _load_summary_means(args.uci_dir / 'metrics_summary.csv')
    ds_fi_s, fi_mean = summary['FullIsingModule']
    ds_nn_s, nn_mean = summary['NeuralNet']
    _check_aligned(ds_fi_s, ds_nn_s, 'UCI summary dataset ordering')
    pp.plot_parity_scatter(
        fi_mean, nn_mean, ds_fi_s,
        model_name1='FullIsingModule',
        model_name2='MultiIsingNetwork',
        filename='uci_parity_scatter',
    )

    # --- XOR single-run accuracies ---
    ds_x_fi, acc_fi = _load_xor_single(args.xor_fullising_dir / 'metrics_summary.csv')
    ds_x_1l, acc_1l = _load_xor_single(args.xor_1l_dir / 'metrics_summary.csv')

    pp.plot_tot_accuracy(acc_fi, ds_x_fi, filename='xor_accuracy_fullising')
    pp.plot_tot_accuracy(acc_1l, ds_x_1l, filename='xor_accuracy_1l')

    _check_aligned(ds_x_fi, ds_x_1l, 'XOR dimension ordering')
    pp.plot_compare_accuracy(
        acc_fi, acc_1l, ds_x_fi,
        model_name1='FullIsingModule',
        model_name2='Network_1L',
        filename='xor_accuracy_compare',
    )

    print(f'SVG figures written to {args.out_dir}')


if __name__ == '__main__':
    main()
