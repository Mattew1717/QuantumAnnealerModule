"""Batch experiments for QuantumAnnealerModule.

This module implements four independent experiments requested in
``Inference/prompt.md`` and stores all outputs under a timestamped
``experiments/`` directory.
"""

import os
import sys
import traceback
import json
from pathlib import Path
from datetime import datetime
from time import perf_counter
from typing import Any
import multiprocessing

import numpy as np
import pandas as pd
import torch
import dotenv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

dotenv.load_dotenv(dotenv_path=Path(__file__).parent / ".env")

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_src_root = os.path.join(_repo_root, "src")
for _path in (_repo_root, _src_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("RANDOM_SEED", "42")

from Inference.utils.logger import Logger
from Inference.utils.plot import Plot
from Inference.utils.utils import (
    flatten_logits,
    compute_metrics,
    save_metrics_csv,
    generate_xor_balanced,
    METRICS,
)
from Inference.utils.dataset_manager import DatasetManager
from full_ising_model.full_ising_module import FullIsingModule
from full_ising_model.annealers import AnnealingSettings, AnnealerType
from ModularNetwork.Network_1L import MultiIsingNetwork


sns.set_style("whitegrid")

DEVICE = "cpu"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_XOR_SAMPLES = 100
DEFAULT_HIDDEN_OFFSET = -0.02
GRID_NUM_NODES = [1, 2, 3, 5, 8]
SIZE_MULTIPLIERS = [1, 2, 3, 4]
SIZE_FIXED = [10, 20, 30, 50]
XOR_DIMS = [1, 2, 3]
EXP2_VARIANTS = [
    "FullIsing_default",
    "FullIsing_1/N",
    "Net1L_default",
    "Net1L_1/N",
]
GAMMA_STRATEGIES = [
    "zeros",
    "small_randn",
    "medium_randn",
    "large_randn",
    "theta_ratio",
]
PALETTE = {
    "FullIsing": "#2E86AB",
    "Net1L": "#A23B72",
    "Accuracy": "#2E86AB",
    "staged": "#2E86AB",
    "baseline": "#A23B72",
    "FullIsing_default": "#2E86AB",
    "FullIsing_1/N": "#6A994E",
    "Net1L_default": "#A23B72",
    "Net1L_1/N": "#F18F01",
    "staged_FullIsing": "#2E86AB",
    "baseline_FullIsing": "#7FB3D5",
    "staged_Net1L": "#A23B72",
    "baseline_Net1L": "#E598C0",
}


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_sa_settings(
    beta_range: list[int],
    num_reads: int,
    num_sweeps: int,
    sweeps_per_beta: int,
) -> AnnealingSettings:
    settings = AnnealingSettings()
    settings.beta_range = beta_range
    settings.num_reads = num_reads
    settings.num_sweeps = num_sweeps
    settings.num_sweeps_per_beta = sweeps_per_beta
    return settings


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1e-8
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    return X_train_std.astype(np.float32), X_test_std.astype(np.float32)


def split_and_standardize(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    X_train, X_test = standardize_train_test(X_train, X_test)
    return X_train, X_test, y_train.astype(np.float32), y_test.astype(np.float32)


def prepare_xor_data(
    dim: int,
    n_samples: int,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = generate_xor_balanced(
        dim=dim,
        n_samples_dim=n_samples,
        shuffle=True,
        random_seed=seed,
    )
    return split_and_standardize(X, y, test_size, seed)


def make_loader(
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
) -> DataLoader:
    tx = torch.tensor(X_train, dtype=torch.float32)
    ty = torch.tensor(y_train, dtype=torch.float32)
    return DataLoader(
        TensorDataset(tx, ty),
        batch_size=batch_size,
        shuffle=True,
    )


def node_size_configs(num_features: int) -> list[tuple[int, str]]:
    configs: list[tuple[int, str]] = []
    for multiplier in SIZE_MULTIPLIERS:
        configs.append((max(num_features * multiplier, 1), f"x{multiplier}F"))
    for size in SIZE_FIXED:
        configs.append((size, str(size)))
    return configs


def list_uci_datasets() -> list[tuple[str, Path]]:
    data_dir = Path(__file__).parent / "Datasets"
    datasets: list[tuple[str, Path]] = []
    for path in sorted(data_dir.glob("*.csv")):
        stem = path.stem
        name = stem.split("_", 1)[1] if "_" in stem else stem
        datasets.append((name.lower(), path))
    return datasets


def nanmean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def nanstd_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanstd(arr))


def log_exception(logger: Logger, prefix: str, exc: Exception) -> None:
    logger.error(f"{prefix}: {exc}")
    logger.error(traceback.format_exc())


def heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    filepath: Path,
    cmap: str = "YlOrRd",
    fmt: str = ".3f",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_label: str = "",
    xlabel: str = "Node size  (size_annealer)",
    ylabel: str = "Rows",
) -> None:
    fig, ax = plt.subplots(
        figsize=(max(8, len(col_labels) * 1.1), max(5, len(row_labels) * 0.9))
    )
    mask = np.isnan(matrix)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        mask=mask,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 9},
        cbar_kws={"label": cbar_label} if cbar_label else {},
    )
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def timing_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    filepath: Path,
    xlabel: str = "Node size  (size_annealer)",
    ylabel: str = "Rows",
) -> None:
    annot = np.vectorize(lambda v: f"{v:.0f}s" if not np.isnan(v) else "ERR")(matrix)
    fig, ax = plt.subplots(
        figsize=(max(8, len(col_labels) * 1.1), max(5, len(row_labels) * 0.9))
    )
    sns.heatmap(
        matrix,
        annot=annot,
        fmt="",
        cmap="Oranges",
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 9},
        cbar_kws={"label": "seconds"},
    )
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_side_by_side_heatmaps(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    left_rows: list[str],
    right_rows: list[str],
    col_labels: list[str],
    left_title: str,
    right_title: str,
    filepath: Path,
    cmap: str = "YlOrRd",
    fmt: str = ".3f",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_label: str = "",
    xlabel: str = "Node size  (size_annealer)",
    left_ylabel: str = "Model",
    right_ylabel: str = "Num perceptrons",
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(14, len(col_labels) * 2.1), max(4.5, len(right_rows) * 0.8 + 2)),
        constrained_layout=True,
    )
    for ax, matrix, rows, title, ylabel in (
        (axes[0], left_matrix, left_rows, left_title, left_ylabel),
        (axes[1], right_matrix, right_rows, right_title, right_ylabel),
    ):
        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            mask=np.isnan(matrix),
            xticklabels=col_labels,
            yticklabels=rows,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"size": 9},
            cbar_kws={"label": cbar_label} if cbar_label else {},
        )
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_side_by_side_timing_heatmaps(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    left_rows: list[str],
    right_rows: list[str],
    col_labels: list[str],
    left_title: str,
    right_title: str,
    filepath: Path,
    xlabel: str = "Node size  (size_annealer)",
    left_ylabel: str = "Model",
    right_ylabel: str = "Num perceptrons",
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(14, len(col_labels) * 2.1), max(4.5, len(right_rows) * 0.8 + 2)),
        constrained_layout=True,
    )
    for ax, matrix, rows, title, ylabel in (
        (axes[0], left_matrix, left_rows, left_title, left_ylabel),
        (axes[1], right_matrix, right_rows, right_title, right_ylabel),
    ):
        annot = np.vectorize(lambda v: f"{v:.0f}s" if not np.isnan(v) else "ERR")(matrix)
        sns.heatmap(
            matrix,
            annot=annot,
            fmt="",
            cmap="Oranges",
            xticklabels=col_labels,
            yticklabels=rows,
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"size": 9},
            cbar_kws={"label": "seconds"},
        )
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_heatmap_df(
    df: pd.DataFrame,
    title: str,
    filepath: Path,
    cmap: str = "YlOrRd",
    fmt: str = ".3f",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_label: str = "Score",
) -> None:
    fig, ax = plt.subplots(
        figsize=(max(7, len(df.columns) * 1.4), max(4, len(df.index) * 0.6 + 2))
    )
    sns.heatmap(
        df,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel(df.columns.name or "", fontweight="bold")
    ax.set_ylabel(df.index.name or "", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_mean_loss_accuracy(
    loss_histories: list[list[float]],
    acc_histories: list[list[float]],
    filepath: Path,
    title: str,
    vertical_line: int | None = None,
) -> None:
    if not loss_histories or not acc_histories:
        return
    loss_arr = np.asarray(loss_histories, dtype=float)
    acc_arr = np.asarray(acc_histories, dtype=float)
    epochs = np.arange(1, loss_arr.shape[1] + 1)

    loss_mean = np.nanmean(loss_arr, axis=0)
    loss_std = np.nanstd(loss_arr, axis=0)
    acc_mean = np.nanmean(acc_arr, axis=0)
    acc_std = np.nanstd(acc_arr, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, loss_mean, color="#2E86AB", linewidth=2)
    if loss_arr.shape[0] > 1:
        axes[0].fill_between(
            epochs,
            loss_mean - loss_std,
            loss_mean + loss_std,
            alpha=0.2,
            color="#2E86AB",
        )
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel("Loss", fontweight="bold")

    axes[1].plot(epochs, acc_mean, color="#6A994E", linewidth=2)
    if acc_arr.shape[0] > 1:
        axes[1].fill_between(
            epochs,
            acc_mean - acc_std,
            acc_mean + acc_std,
            alpha=0.2,
            color="#6A994E",
        )
    axes[1].set_title("Validation Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel("Accuracy", fontweight="bold")
    axes[1].set_ylim(0, 1.0)

    if vertical_line is not None:
        for ax in axes:
            ax.axvline(vertical_line, color="black", linestyle="--", linewidth=1.2)

    fig.suptitle(title, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_dual_overlay_histories(
    left_curves: dict[str, list[float]],
    right_curves: dict[str, list[float]],
    filepath: Path,
    title: str,
    ylabel: str,
    left_title: str,
    right_title: str,
    vertical_line: int | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, curves, subplot_title in (
        (axes[0], left_curves, left_title),
        (axes[1], right_curves, right_title),
    ):
        for label, values in curves.items():
            epochs = np.arange(1, len(values) + 1)
            ax.plot(
                epochs,
                values,
                label=label,
                linewidth=2,
                color=PALETTE.get(label),
            )
        if vertical_line is not None:
            ax.axvline(vertical_line, color="black", linestyle="--", linewidth=1.2)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(subplot_title, fontweight="bold")
        ax.legend(frameon=True)
    fig.suptitle(title, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_overlay_curves(
    curves: dict[str, list[float]],
    filepath: Path,
    title: str,
    ylabel: str,
    vertical_line: int | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, values in curves.items():
        epochs = np.arange(1, len(values) + 1)
        ax.plot(
            epochs,
            values,
            label=label,
            linewidth=2,
            color=PALETTE.get(label),
        )
    if vertical_line is not None:
        ax.axvline(vertical_line, color="black", linestyle="--", linewidth=1.2, label="Phase switch")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_grouped_bar(
    labels: list[str],
    series: dict[str, list[float]],
    filepath: Path,
    title: str,
    ylabel: str,
    errors: dict[str, list[float]] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 5))
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(series))
    offsets = np.linspace(
        -0.4 + width / 2,
        0.4 - width / 2,
        num=max(1, len(series)),
    )

    for offset, (name, values) in zip(offsets, series.items()):
        yerr = errors.get(name) if errors else None
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=name,
            color=PALETTE.get(name),
            edgecolor="black",
            linewidth=1.0,
            yerr=yerr,
            capsize=4,
        )
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_multi_radar(
    series: dict[str, dict[str, float]],
    filepath: Path,
    title: str,
) -> None:
    keys = ["accuracy", "precision", "recall", "f1", "auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for name, metrics in series.items():
        values = [float(metrics.get(key, np.nan)) for key in keys]
        values = [0.0 if np.isnan(v) else v for v in values]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=name, color=PALETTE.get(name))
        ax.fill(angles, values, alpha=0.12, color=PALETTE.get(name))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_title(title, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), frameon=True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_boxplot_long(
    df: pd.DataFrame,
    filepath: Path,
    title: str,
) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(10, df["dataset"].nunique() * 1.5), 6))
    sns.boxplot(
        data=df,
        x="dataset",
        y="accuracy",
        hue="variant",
        palette=[PALETTE.get(v) for v in df["variant"].drop_duplicates()],
        ax=ax,
    )
    ax.set_xlabel("Dataset", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def evaluate_model(
    model: torch.nn.Module,
    X_eval_tensor: torch.Tensor,
    y_true: np.ndarray,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = flatten_logits(model(X_eval_tensor))
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    metrics = compute_metrics(y_true.astype(int), probs)
    return {
        "metrics": {k: float(v) if not np.isnan(v) else float("nan") for k, v in metrics.items()},
        "probs": probs,
        "preds": preds,
    }


def build_full_model(
    *,
    size_annealer: int,
    sa_settings: AnnealingSettings,
    lambda_init: float,
    offset_init: float,
    gamma_init: torch.Tensor | None = None,
    hidden_nodes_offset_value: float = DEFAULT_HIDDEN_OFFSET,
) -> FullIsingModule:
    return FullIsingModule(
        size_annealer=size_annealer,
        annealer_type=AnnealerType.SIMULATED,
        annealing_settings=sa_settings,
        lambda_init=lambda_init,
        offset_init=offset_init,
        gamma_init=gamma_init.clone() if gamma_init is not None else None,
        hidden_nodes_offset_value=hidden_nodes_offset_value,
    ).to(DEVICE)


def set_net_hidden_offset(model: MultiIsingNetwork, hidden_nodes_offset_value: float) -> None:
    for module in model.ising_perceptrons_layer:
        module.hidden_nodes_config.fun_args = (hidden_nodes_offset_value,)
        module.hidden_nodes_config._resize_cache = {}


def set_net_gamma(model: MultiIsingNetwork, gamma_init: torch.Tensor) -> None:
    for module in model.ising_perceptrons_layer:
        with torch.no_grad():
            module.gamma.copy_(gamma_init)


def build_net_model(
    *,
    num_ising_perceptrons: int,
    size_annealer: int,
    sa_settings: AnnealingSettings,
    lambda_init: float,
    offset_init: float,
    gamma_init: torch.Tensor | None = None,
    hidden_nodes_offset_value: float = DEFAULT_HIDDEN_OFFSET,
) -> MultiIsingNetwork:
    model = MultiIsingNetwork(
        num_ising_perceptrons=num_ising_perceptrons,
        size_annealer=size_annealer,
        annealing_settings=sa_settings,
        annealer_type=AnnealerType.SIMULATED,
        lambda_init=lambda_init,
        offset_init=offset_init,
        partition_input=False,
    ).to(DEVICE)
    set_net_hidden_offset(model, hidden_nodes_offset_value)
    if gamma_init is not None:
        set_net_gamma(model, gamma_init)
    return model


def build_full_optimizer(
    model: FullIsingModule,
    *,
    optimizer_name: str,
    lr_gamma: float,
    lr_lambda: float | None = None,
    lr_offset: float | None = None,
    momentum: float = 0.0,
) -> torch.optim.Optimizer:
    groups = [{"params": [model.gamma], "lr": lr_gamma}]
    if lr_lambda is not None and model.lmd.requires_grad:
        groups.append({"params": [model.lmd], "lr": lr_lambda})
    if lr_offset is not None and model.offset.requires_grad:
        groups.append({"params": [model.offset], "lr": lr_offset})
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(groups, momentum=momentum)
    return torch.optim.Adam(groups)


def build_net_optimizer(
    model: MultiIsingNetwork,
    *,
    optimizer_name: str,
    lr_gamma: float,
    lr_lambda: float | None = None,
    lr_offset: float | None = None,
    lr_combiner: float | None = None,
    momentum: float = 0.0,
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = []
    for module in model.ising_perceptrons_layer:
        if module.gamma.requires_grad:
            groups.append({"params": [module.gamma], "lr": lr_gamma})
        if lr_lambda is not None and module.lmd.requires_grad:
            groups.append({"params": [module.lmd], "lr": lr_lambda})
        if lr_offset is not None and module.offset.requires_grad:
            groups.append({"params": [module.offset], "lr": lr_offset})
    if lr_combiner is not None:
        combiner_params = [p for p in model.combiner_layer.parameters() if p.requires_grad]
        if combiner_params:
            groups.append({"params": combiner_params, "lr": lr_combiner})
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(groups, momentum=momentum)
    return torch.optim.Adam(groups)


def _run_training_phase(
    model: torch.nn.Module,
    train_loader: DataLoader,
    X_eval_tensor: torch.Tensor,
    y_eval: np.ndarray,
    optimizer: torch.optim.Optimizer,
    *,
    phase_epochs: int,
    start_epoch: int,
    total_epochs: int,
    loss_mode: str,
    logger: Logger | None = None,
    log_prefix: str = "",
    losses: list[float],
    val_accuracies: list[float],
    capture_epochs: set[int] | None = None,
    captured_metrics: dict[int, dict[str, float]] | None = None,
) -> dict[str, Any]:
    if loss_mode == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    last_eval: dict[str, Any] | None = None
    for local_epoch in range(phase_epochs):
        global_epoch = start_epoch + local_epoch
        model.train()
        batch_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE).float()
            optimizer.zero_grad()
            logits = flatten_logits(model(xb))
            if loss_mode == "mse":
                loss = loss_fn(torch.sigmoid(logits), yb)
            else:
                loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        losses.append(avg_loss)

        last_eval = evaluate_model(model, X_eval_tensor, y_eval)
        val_accuracies.append(last_eval["metrics"]["accuracy"])
        if capture_epochs and captured_metrics is not None and global_epoch in capture_epochs:
            captured_metrics[global_epoch] = dict(last_eval["metrics"])
        if logger and (global_epoch == 1 or global_epoch % 20 == 0 or global_epoch == total_epochs):
            logger.info(
                f"{log_prefix} epoch {global_epoch:03d}/{total_epochs} | "
                f"loss={avg_loss:.4f} | acc={last_eval['metrics']['accuracy']:.4f}"
            )
    if last_eval is None:
        last_eval = evaluate_model(model, X_eval_tensor, y_eval)
    return last_eval


def _finalize_training_result(
    model: torch.nn.Module,
    final_eval: dict[str, Any],
    losses: list[float],
    val_accuracies: list[float],
    captured_metrics: dict[int, dict[str, float]],
    elapsed: float,
) -> dict[str, Any]:
    result = dict(final_eval["metrics"])
    result.update(
        {
            "probs": final_eval["probs"],
            "preds": final_eval["preds"],
            "training_losses": losses,
            "val_accuracies": val_accuracies,
            "best_val_acc": float(np.nanmax(val_accuracies)) if val_accuracies else float("nan"),
            "final_loss": losses[-1] if losses else float("nan"),
            "min_loss": min(losses) if losses else float("nan"),
            "training_time_s": float(elapsed),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "captured_metrics": captured_metrics,
        }
    )
    return result


def train_standard_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    loss_mode: str,
    logger: Logger | None = None,
    log_prefix: str = "",
    capture_epochs: set[int] | None = None,
) -> dict[str, Any]:
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32, device=DEVICE)
    losses: list[float] = []
    val_accuracies: list[float] = []
    captured_metrics: dict[int, dict[str, float]] = {}
    start = perf_counter()
    final_eval = _run_training_phase(
        model,
        train_loader,
        X_eval_tensor,
        y_eval,
        optimizer,
        phase_epochs=epochs,
        start_epoch=1,
        total_epochs=epochs,
        loss_mode=loss_mode,
        logger=logger,
        log_prefix=log_prefix,
        losses=losses,
        val_accuracies=val_accuracies,
        capture_epochs=capture_epochs,
        captured_metrics=captured_metrics,
    )
    elapsed = perf_counter() - start
    return _finalize_training_result(
        model,
        final_eval,
        losses,
        val_accuracies,
        captured_metrics,
        elapsed,
    )


def run_full_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    batch_size: int,
    epochs: int,
    size_annealer: int,
    lambda_init: float,
    offset_init: float,
    lr_gamma: float,
    lr_lambda: float,
    lr_offset: float,
    sa_settings: AnnealingSettings,
    optimizer_name: str,
    loss_mode: str,
    momentum: float = 0.0,
    gamma_init: torch.Tensor | None = None,
    hidden_nodes_offset_value: float = DEFAULT_HIDDEN_OFFSET,
    logger: Logger | None = None,
    log_prefix: str = "",
    capture_epochs: set[int] | None = None,
) -> dict[str, Any]:
    set_global_seed(seed)
    train_loader = make_loader(X_train, y_train, batch_size)
    model = build_full_model(
        size_annealer=size_annealer,
        sa_settings=sa_settings,
        lambda_init=lambda_init,
        offset_init=offset_init,
        gamma_init=gamma_init,
        hidden_nodes_offset_value=hidden_nodes_offset_value,
    )
    optimizer = build_full_optimizer(
        model,
        optimizer_name=optimizer_name,
        lr_gamma=lr_gamma,
        lr_lambda=lr_lambda,
        lr_offset=lr_offset,
        momentum=momentum,
    )
    return train_standard_model(
        model,
        train_loader,
        X_test,
        y_test,
        optimizer,
        epochs=epochs,
        loss_mode=loss_mode,
        logger=logger,
        log_prefix=log_prefix,
        capture_epochs=capture_epochs,
    )


def run_net_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    batch_size: int,
    epochs: int,
    num_ising_perceptrons: int,
    size_annealer: int,
    lambda_init: float,
    offset_init: float,
    lr_gamma: float,
    lr_lambda: float,
    lr_offset: float,
    lr_combiner: float,
    sa_settings: AnnealingSettings,
    optimizer_name: str,
    loss_mode: str,
    momentum: float = 0.0,
    gamma_init: torch.Tensor | None = None,
    hidden_nodes_offset_value: float = DEFAULT_HIDDEN_OFFSET,
    logger: Logger | None = None,
    log_prefix: str = "",
    capture_epochs: set[int] | None = None,
) -> dict[str, Any]:
    set_global_seed(seed)
    train_loader = make_loader(X_train, y_train, batch_size)
    model = build_net_model(
        num_ising_perceptrons=num_ising_perceptrons,
        size_annealer=size_annealer,
        sa_settings=sa_settings,
        lambda_init=lambda_init,
        offset_init=offset_init,
        gamma_init=gamma_init,
        hidden_nodes_offset_value=hidden_nodes_offset_value,
    )
    optimizer = build_net_optimizer(
        model,
        optimizer_name=optimizer_name,
        lr_gamma=lr_gamma,
        lr_lambda=lr_lambda,
        lr_offset=lr_offset,
        lr_combiner=lr_combiner,
        momentum=momentum,
    )
    return train_standard_model(
        model,
        train_loader,
        X_test,
        y_test,
        optimizer,
        epochs=epochs,
        loss_mode=loss_mode,
        logger=logger,
        log_prefix=log_prefix,
        capture_epochs=capture_epochs,
    )


def run_full_training_staged(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    batch_size: int,
    size_annealer: int,
    lambda_init: float,
    offset_init: float,
    sa_settings: AnnealingSettings,
    phase1_epochs: int,
    total_epochs: int,
    baseline: bool,
    logger: Logger | None = None,
    log_prefix: str = "",
) -> dict[str, Any]:
    set_global_seed(seed)
    train_loader = make_loader(X_train, y_train, batch_size)
    model = build_full_model(
        size_annealer=size_annealer,
        sa_settings=sa_settings,
        lambda_init=lambda_init,
        offset_init=offset_init,
    )
    if baseline:
        optimizer = build_full_optimizer(
            model,
            optimizer_name="adam",
            lr_gamma=0.0005,
            lr_lambda=0.005,
            lr_offset=0.01,
        )
        return train_standard_model(
            model,
            train_loader,
            X_test,
            y_test,
            optimizer,
            epochs=total_epochs,
            loss_mode="bce",
            logger=logger,
            log_prefix=log_prefix,
            capture_epochs={phase1_epochs},
        )

    X_eval_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    losses: list[float] = []
    val_accuracies: list[float] = []
    captured_metrics: dict[int, dict[str, float]] = {}

    model.lmd.requires_grad = False
    model.offset.requires_grad = False
    opt_phase1 = build_full_optimizer(
        model,
        optimizer_name="adam",
        lr_gamma=0.005,
        lr_lambda=None,
        lr_offset=None,
    )
    start = perf_counter()
    phase1_eval = _run_training_phase(
        model,
        train_loader,
        X_eval_tensor,
        y_test,
        opt_phase1,
        phase_epochs=phase1_epochs,
        start_epoch=1,
        total_epochs=total_epochs,
        loss_mode="bce",
        logger=logger,
        log_prefix=f"{log_prefix}[phase1] ",
        losses=losses,
        val_accuracies=val_accuracies,
        capture_epochs={phase1_epochs},
        captured_metrics=captured_metrics,
    )

    model.lmd.requires_grad = True
    model.offset.requires_grad = True
    opt_phase2 = build_full_optimizer(
        model,
        optimizer_name="adam",
        lr_gamma=0.0005,
        lr_lambda=0.005,
        lr_offset=0.01,
    )
    final_eval = _run_training_phase(
        model,
        train_loader,
        X_eval_tensor,
        y_test,
        opt_phase2,
        phase_epochs=total_epochs - phase1_epochs,
        start_epoch=phase1_epochs + 1,
        total_epochs=total_epochs,
        loss_mode="bce",
        logger=logger,
        log_prefix=f"{log_prefix}[phase2] ",
        losses=losses,
        val_accuracies=val_accuracies,
        capture_epochs=None,
        captured_metrics=captured_metrics,
    )
    elapsed = perf_counter() - start
    result = _finalize_training_result(
        model,
        final_eval,
        losses,
        val_accuracies,
        captured_metrics,
        elapsed,
    )
    result["phase1_final_eval"] = phase1_eval
    return result


def run_net_training_staged(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    batch_size: int,
    num_ising_perceptrons: int,
    size_annealer: int,
    lambda_init: float,
    offset_init: float,
    sa_settings: AnnealingSettings,
    phase1_epochs: int,
    total_epochs: int,
    baseline: bool,
    logger: Logger | None = None,
    log_prefix: str = "",
) -> dict[str, Any]:
    set_global_seed(seed)
    train_loader = make_loader(X_train, y_train, batch_size)
    model = build_net_model(
        num_ising_perceptrons=num_ising_perceptrons,
        size_annealer=size_annealer,
        sa_settings=sa_settings,
        lambda_init=lambda_init,
        offset_init=offset_init,
    )
    if baseline:
        optimizer = build_net_optimizer(
            model,
            optimizer_name="adam",
            lr_gamma=0.0005,
            lr_lambda=0.005,
            lr_offset=0.01,
            lr_combiner=0.005,
        )
        return train_standard_model(
            model,
            train_loader,
            X_test,
            y_test,
            optimizer,
            epochs=total_epochs,
            loss_mode="bce",
            logger=logger,
            log_prefix=log_prefix,
            capture_epochs={phase1_epochs},
        )

    X_eval_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    losses: list[float] = []
    val_accuracies: list[float] = []
    captured_metrics: dict[int, dict[str, float]] = {}

    for module in model.ising_perceptrons_layer:
        module.lmd.requires_grad = False
        module.offset.requires_grad = False
    for param in model.combiner_layer.parameters():
        param.requires_grad = False

    opt_phase1 = build_net_optimizer(
        model,
        optimizer_name="adam",
        lr_gamma=0.005,
        lr_lambda=None,
        lr_offset=None,
        lr_combiner=None,
    )
    start = perf_counter()
    phase1_eval = _run_training_phase(
        model,
        train_loader,
        X_eval_tensor,
        y_test,
        opt_phase1,
        phase_epochs=phase1_epochs,
        start_epoch=1,
        total_epochs=total_epochs,
        loss_mode="bce",
        logger=logger,
        log_prefix=f"{log_prefix}[phase1] ",
        losses=losses,
        val_accuracies=val_accuracies,
        capture_epochs={phase1_epochs},
        captured_metrics=captured_metrics,
    )

    for module in model.ising_perceptrons_layer:
        module.lmd.requires_grad = True
        module.offset.requires_grad = True
    for param in model.combiner_layer.parameters():
        param.requires_grad = True

    opt_phase2 = build_net_optimizer(
        model,
        optimizer_name="adam",
        lr_gamma=0.0005,
        lr_lambda=0.005,
        lr_offset=0.01,
        lr_combiner=0.005,
    )
    final_eval = _run_training_phase(
        model,
        train_loader,
        X_eval_tensor,
        y_test,
        opt_phase2,
        phase_epochs=total_epochs - phase1_epochs,
        start_epoch=phase1_epochs + 1,
        total_epochs=total_epochs,
        loss_mode="bce",
        logger=logger,
        log_prefix=f"{log_prefix}[phase2] ",
        losses=losses,
        val_accuracies=val_accuracies,
        capture_epochs=None,
        captured_metrics=captured_metrics,
    )
    elapsed = perf_counter() - start
    result = _finalize_training_result(
        model,
        final_eval,
        losses,
        val_accuracies,
        captured_metrics,
        elapsed,
    )
    result["phase1_final_eval"] = phase1_eval
    return result


def build_gamma_init(
    strategy: str,
    X_train_std: np.ndarray,
    size_annealer: int,
    *,
    seed: int,
    hidden_nodes_offset_value: float = DEFAULT_HIDDEN_OFFSET,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if strategy == "zeros":
        return torch.zeros((size_annealer, size_annealer), dtype=torch.float32)
    if strategy == "small_randn":
        return torch.triu(torch.randn((size_annealer, size_annealer), generator=generator) * 0.001, diagonal=1)
    if strategy == "medium_randn":
        return torch.triu(torch.randn((size_annealer, size_annealer), generator=generator) * 0.1, diagonal=1)
    if strategy == "large_randn":
        return torch.triu(torch.randn((size_annealer, size_annealer), generator=generator) * 1.0, diagonal=1)
    if strategy == "theta_ratio":
        mean_theta = torch.tensor(X_train_std.mean(axis=0), dtype=torch.float32)
        indices = torch.arange(size_annealer)
        theta_padded = mean_theta[indices % len(mean_theta)] + (
            (indices // len(mean_theta)).float() * hidden_nodes_offset_value
        )
        eps = 1e-8
        gamma_init = theta_padded.unsqueeze(0) / (theta_padded.unsqueeze(1) + eps)
        return torch.triu(gamma_init, diagonal=1)
    raise ValueError(f"Unknown gamma initialization strategy: {strategy}")


def convergence_epoch(val_accuracies: list[float], final_accuracy: float, ratio: float = 0.9) -> float:
    if not val_accuracies or np.isnan(final_accuracy):
        return float("nan")
    threshold = final_accuracy * ratio
    for idx, value in enumerate(val_accuracies, start=1):
        if value >= threshold:
            return float(idx)
    return float("nan")


def summarize_variant_results(results: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc", "training_time_s"]
    for key in metric_keys:
        values = [float(r.get(key, np.nan)) for r in results]
        summary[f"{key}_mean"] = nanmean_or_nan(values)
        summary[f"{key}_std"] = nanstd_or_nan(values)
    return summary


def run_exp1(exp_dir: Path) -> None:
    torch.set_num_threads(1)
    ensure_dir(exp_dir)
    dataset_manager = DatasetManager()
    logger = Logger(log_dir=str(exp_dir))
    dataset_manager.logger = logger

    params = {
        "seed": 42,
        "batch_size": 32,
        "epochs": 150,
        "lambda_init": -0.1,
        "offset_init": 0.0,
        "lr_gamma": 0.01,
        "lr_lambda": 0.01,
        "lr_offset": 0.05,
        "lr_combiner": 0.01,
        "momentum": 0.9,
        "test_size": 0.2,
        "n_samples_per_region": 100,
    }
    sa_settings = make_sa_settings([1, 10], 1, 1000, 1)
    summary_records: list[dict[str, Any]] = []

    cases: list[tuple[str, dict[str, Any]]] = [
        ("xor_1d", {"kind": "xor", "dim": 1}),
        ("xor_2d", {"kind": "xor", "dim": 2}),
        ("xor_3d", {"kind": "xor", "dim": 3}),
        (
            "iris",
            {
                "kind": "uci",
                "path": Path("Inference/Datasets/00_iris_versicolor_virginica.csv"),
            },
        ),
    ]

    logger.info("=== EXP1: Matrix FullIsing vs Network1L ===")
    for dataset_name, spec in cases:
        dataset_dir = ensure_dir(exp_dir / dataset_name)
        logger.info(f"\n[EXP1] Dataset: {dataset_name}")
        if spec["kind"] == "xor":
            X_train, X_test, y_train, y_test = prepare_xor_data(
                spec["dim"],
                params["n_samples_per_region"],
                params["test_size"],
                params["seed"],
            )
        else:
            X, y = dataset_manager.load_csv_dataset(str(spec["path"]))
            X_train, X_test, y_train, y_test = split_and_standardize(
                X,
                y,
                params["test_size"],
                params["seed"],
            )

        num_features = X_train.shape[1]
        configs = node_size_configs(num_features)
        plot_cols = [
            f"{label}\n({size})" if label.startswith("x") else label
            for size, label in configs
        ]

        full_acc = np.full((1, len(configs)), np.nan)
        full_f1 = np.full((1, len(configs)), np.nan)
        full_auc = np.full((1, len(configs)), np.nan)
        full_time = np.full((1, len(configs)), np.nan)

        net_acc = np.full((len(GRID_NUM_NODES), len(configs)), np.nan)
        net_f1 = np.full((len(GRID_NUM_NODES), len(configs)), np.nan)
        net_auc = np.full((len(GRID_NUM_NODES), len(configs)), np.nan)
        net_time = np.full((len(GRID_NUM_NODES), len(configs)), np.nan)

        detailed_records: list[dict[str, Any]] = []
        for col_idx, (size_annealer, size_label) in enumerate(configs):
            try:
                full_result = run_full_training(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    seed=params["seed"],
                    batch_size=params["batch_size"],
                    epochs=params["epochs"],
                    size_annealer=size_annealer,
                    lambda_init=params["lambda_init"],
                    offset_init=params["offset_init"],
                    lr_gamma=params["lr_gamma"],
                    lr_lambda=params["lr_lambda"],
                    lr_offset=params["lr_offset"],
                    sa_settings=sa_settings,
                    optimizer_name="sgd",
                    loss_mode="mse",
                    momentum=params["momentum"],
                    logger=logger,
                    log_prefix=f"[EXP1][{dataset_name}][Full][size={size_label}]",
                )
                full_acc[0, col_idx] = full_result["accuracy"]
                full_f1[0, col_idx] = full_result["f1"]
                full_auc[0, col_idx] = full_result["auc"]
                full_time[0, col_idx] = full_result["training_time_s"]
                detailed_records.append(
                    {
                        "dataset": dataset_name,
                        "model": "FullIsing",
                        "num_nodi": 1,
                        "size_annealer": size_annealer,
                        "size_label": size_label,
                        "accuracy": full_result["accuracy"],
                        "precision": full_result["precision"],
                        "recall": full_result["recall"],
                        "f1": full_result["f1"],
                        "auc": full_result["auc"],
                        "training_time_s": full_result["training_time_s"],
                        "error": "",
                    }
                )
            except Exception as exc:
                log_exception(logger, f"[EXP1][{dataset_name}][Full][size={size_label}]", exc)
                detailed_records.append(
                    {
                        "dataset": dataset_name,
                        "model": "FullIsing",
                        "num_nodi": 1,
                        "size_annealer": size_annealer,
                        "size_label": size_label,
                        "accuracy": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "auc": np.nan,
                        "training_time_s": np.nan,
                        "error": str(exc),
                    }
                )

            for row_idx, num_nodi in enumerate(GRID_NUM_NODES):
                try:
                    net_result = run_net_training(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        seed=params["seed"],
                        batch_size=params["batch_size"],
                        epochs=params["epochs"],
                        num_ising_perceptrons=num_nodi,
                        size_annealer=size_annealer,
                        lambda_init=params["lambda_init"],
                        offset_init=params["offset_init"],
                        lr_gamma=params["lr_gamma"],
                        lr_lambda=params["lr_lambda"],
                        lr_offset=params["lr_offset"],
                        lr_combiner=params["lr_combiner"],
                        sa_settings=sa_settings,
                        optimizer_name="sgd",
                        loss_mode="mse",
                        momentum=params["momentum"],
                        logger=logger,
                        log_prefix=f"[EXP1][{dataset_name}][Net1L][n={num_nodi}][size={size_label}]",
                    )
                    net_acc[row_idx, col_idx] = net_result["accuracy"]
                    net_f1[row_idx, col_idx] = net_result["f1"]
                    net_auc[row_idx, col_idx] = net_result["auc"]
                    net_time[row_idx, col_idx] = net_result["training_time_s"]
                    detailed_records.append(
                        {
                            "dataset": dataset_name,
                            "model": "Net1L",
                            "num_nodi": num_nodi,
                            "size_annealer": size_annealer,
                            "size_label": size_label,
                            "accuracy": net_result["accuracy"],
                            "precision": net_result["precision"],
                            "recall": net_result["recall"],
                            "f1": net_result["f1"],
                            "auc": net_result["auc"],
                            "training_time_s": net_result["training_time_s"],
                            "error": "",
                        }
                    )
                except Exception as exc:
                    log_exception(
                        logger,
                        f"[EXP1][{dataset_name}][Net1L][n={num_nodi}][size={size_label}]",
                        exc,
                    )
                    detailed_records.append(
                        {
                            "dataset": dataset_name,
                            "model": "Net1L",
                            "num_nodi": num_nodi,
                            "size_annealer": size_annealer,
                            "size_label": size_label,
                            "accuracy": np.nan,
                            "precision": np.nan,
                            "recall": np.nan,
                            "f1": np.nan,
                            "auc": np.nan,
                            "training_time_s": np.nan,
                            "error": str(exc),
                        }
                    )

        detail_df = pd.DataFrame(detailed_records)
        detail_df.to_csv(dataset_dir / "detailed_metrics.csv", index=False)
        summary_records.extend(detailed_records)

        plot_side_by_side_heatmaps(
            full_acc,
            net_acc,
            ["FullIsing"],
            [str(x) for x in GRID_NUM_NODES],
            plot_cols,
            "FullIsing Accuracy",
            "Net1L Accuracy",
            dataset_dir / "accuracy_heatmap.png",
            cmap="YlOrRd",
            cbar_label="Accuracy",
            left_ylabel="Model",
            right_ylabel="Num perceptrons",
        )
        plot_side_by_side_heatmaps(
            full_f1,
            net_f1,
            ["FullIsing"],
            [str(x) for x in GRID_NUM_NODES],
            plot_cols,
            "FullIsing F1",
            "Net1L F1",
            dataset_dir / "f1_heatmap.png",
            cmap="Blues",
            cbar_label="F1",
            left_ylabel="Model",
            right_ylabel="Num perceptrons",
        )
        plot_side_by_side_heatmaps(
            full_auc,
            net_auc,
            ["FullIsing"],
            [str(x) for x in GRID_NUM_NODES],
            plot_cols,
            "FullIsing AUC",
            "Net1L AUC",
            dataset_dir / "auc_heatmap.png",
            cmap="Greens",
            cbar_label="AUC",
            left_ylabel="Model",
            right_ylabel="Num perceptrons",
        )
        plot_side_by_side_timing_heatmaps(
            full_time,
            net_time,
            ["FullIsing"],
            [str(x) for x in GRID_NUM_NODES],
            plot_cols,
            "FullIsing Training Time",
            "Net1L Training Time",
            dataset_dir / "timing_heatmap.png",
            left_ylabel="Model",
            right_ylabel="Num perceptrons",
        )

    pd.DataFrame(summary_records).to_csv(exp_dir / "summary.csv", index=False)


def run_exp2(exp_dir: Path) -> None:
    torch.set_num_threads(1)
    ensure_dir(exp_dir)
    dataset_manager = DatasetManager()
    logger = Logger(log_dir=str(exp_dir))
    dataset_manager.logger = logger

    params = {
        "seed": 42,
        "batch_size": 32,
        "epochs": 200,
        "size_annealer": 50,
        "num_ising_perceptrons": 5,
        "lambda_init": -0.1,
        "offset_init": 0.0,
        "lr_gamma": 0.001,
        "lr_lambda": 0.005,
        "lr_offset": 0.01,
        "lr_combiner": 0.005,
        "test_size": 0.2,
        "n_samples_per_region": 100,
        "k_folds": 5,
    }
    sa_settings = make_sa_settings([1, 10], 1, 1000, 1)
    summary_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    boxplot_rows: list[dict[str, Any]] = []
    metric_tables: dict[str, dict[str, list[float]]] = {
        "accuracy": {variant: [] for variant in EXP2_VARIANTS},
        "f1": {variant: [] for variant in EXP2_VARIANTS},
        "auc": {variant: [] for variant in EXP2_VARIANTS},
        "training_time_s": {variant: [] for variant in EXP2_VARIANTS},
    }
    dataset_order: list[str] = []

    dataset_specs: list[tuple[str, dict[str, Any]]] = [
        ("xor_1d", {"kind": "xor", "dim": 1}),
        ("xor_2d", {"kind": "xor", "dim": 2}),
        ("xor_3d", {"kind": "xor", "dim": 3}),
    ]
    dataset_specs.extend((name, {"kind": "uci", "path": path}) for name, path in list_uci_datasets())

    logger.info("=== EXP2: Offset Sweep ===")
    for dataset_name, spec in dataset_specs:
        dataset_order.append(dataset_name)
        dataset_dir = ensure_dir(exp_dir / dataset_name)
        logger.info(f"\n[EXP2] Dataset: {dataset_name}")

        if spec["kind"] == "xor":
            X_train, X_test, y_train, y_test = prepare_xor_data(
                spec["dim"],
                params["n_samples_per_region"],
                params["test_size"],
                params["seed"],
            )
            splits = [(X_train, y_train, X_test, y_test)]
        else:
            X, y = dataset_manager.load_csv_dataset(str(spec["path"]))
            splits = dataset_manager.generate_k_folds(X, y, params["k_folds"])

        variant_results: dict[str, list[dict[str, Any]]] = {variant: [] for variant in EXP2_VARIANTS}
        variant_losses: dict[str, list[list[float]]] = {variant: [] for variant in EXP2_VARIANTS}
        variant_accs: dict[str, list[list[float]]] = {variant: [] for variant in EXP2_VARIANTS}

        for fold_idx, split in enumerate(splits, start=1):
            fold_X_train, fold_y_train, fold_X_test, fold_y_test = split
            fold_X_train, fold_X_test = standardize_train_test(fold_X_train, fold_X_test)
            fold_y_train = fold_y_train.astype(np.float32)
            fold_y_test = fold_y_test.astype(np.float32)

            for variant in EXP2_VARIANTS:
                hidden_offset = DEFAULT_HIDDEN_OFFSET
                if variant.endswith("1/N"):
                    hidden_offset = 1.0 / params["size_annealer"]
                try:
                    if variant.startswith("FullIsing"):
                        result = run_full_training(
                            fold_X_train,
                            fold_y_train,
                            fold_X_test,
                            fold_y_test,
                            seed=params["seed"],
                            batch_size=params["batch_size"],
                            epochs=params["epochs"],
                            size_annealer=params["size_annealer"],
                            lambda_init=params["lambda_init"],
                            offset_init=params["offset_init"],
                            lr_gamma=params["lr_gamma"],
                            lr_lambda=params["lr_lambda"],
                            lr_offset=params["lr_offset"],
                            sa_settings=sa_settings,
                            optimizer_name="adam",
                            loss_mode="bce",
                            hidden_nodes_offset_value=hidden_offset,
                            logger=logger,
                            log_prefix=f"[EXP2][{dataset_name}][{variant}][fold={fold_idx}]",
                        )
                    else:
                        result = run_net_training(
                            fold_X_train,
                            fold_y_train,
                            fold_X_test,
                            fold_y_test,
                            seed=params["seed"],
                            batch_size=params["batch_size"],
                            epochs=params["epochs"],
                            num_ising_perceptrons=params["num_ising_perceptrons"],
                            size_annealer=params["size_annealer"],
                            lambda_init=params["lambda_init"],
                            offset_init=params["offset_init"],
                            lr_gamma=params["lr_gamma"],
                            lr_lambda=params["lr_lambda"],
                            lr_offset=params["lr_offset"],
                            lr_combiner=params["lr_combiner"],
                            sa_settings=sa_settings,
                            optimizer_name="adam",
                            loss_mode="bce",
                            hidden_nodes_offset_value=hidden_offset,
                            logger=logger,
                            log_prefix=f"[EXP2][{dataset_name}][{variant}][fold={fold_idx}]",
                        )
                    variant_results[variant].append(result)
                    variant_losses[variant].append(result["training_losses"])
                    variant_accs[variant].append(result["val_accuracies"])
                    detailed_rows.append(
                        {
                            "dataset": dataset_name,
                            "variant": variant,
                            "fold": fold_idx,
                            "accuracy": result["accuracy"],
                            "precision": result["precision"],
                            "recall": result["recall"],
                            "f1": result["f1"],
                            "auc": result["auc"],
                            "training_time_s": result["training_time_s"],
                            "error": "",
                        }
                    )
                    if spec["kind"] == "uci":
                        boxplot_rows.append(
                            {"dataset": dataset_name, "variant": variant, "accuracy": result["accuracy"]}
                        )
                except Exception as exc:
                    log_exception(logger, f"[EXP2][{dataset_name}][{variant}][fold={fold_idx}]", exc)
                    detailed_rows.append(
                        {
                            "dataset": dataset_name,
                            "variant": variant,
                            "fold": fold_idx,
                            "accuracy": np.nan,
                            "precision": np.nan,
                            "recall": np.nan,
                            "f1": np.nan,
                            "auc": np.nan,
                            "training_time_s": np.nan,
                            "error": str(exc),
                        }
                    )

        bar_series = {"accuracy": [], "accuracy_std": []}
        for variant in EXP2_VARIANTS:
            result_list = variant_results[variant]
            summary = summarize_variant_results(result_list)
            summary_row = {"dataset": dataset_name, "variant": variant, **summary}
            summary_rows.append(summary_row)
            metric_tables["accuracy"][variant].append(summary["accuracy_mean"])
            metric_tables["f1"][variant].append(summary["f1_mean"])
            metric_tables["auc"][variant].append(summary["auc_mean"])
            metric_tables["training_time_s"][variant].append(summary["training_time_s_mean"])
            bar_series["accuracy"].append(summary["accuracy_mean"])
            bar_series["accuracy_std"].append(summary["accuracy_std"])

            if variant_losses[variant] and variant_accs[variant]:
                plot_mean_loss_accuracy(
                    variant_losses[variant],
                    variant_accs[variant],
                    dataset_dir / f"loss_accuracy_{variant}.png",
                    f"{dataset_name} - {variant}",
                )

        plot_grouped_bar(
            EXP2_VARIANTS,
            {"Accuracy": bar_series["accuracy"]},
            dataset_dir / "accuracy_comparison.png",
            f"Offset Sweep Accuracy - {dataset_name}",
            "Accuracy",
            errors={"Accuracy": bar_series["accuracy_std"]},
            ylim=(0, 1.0),
        )

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(detailed_rows)
    summary_df.to_csv(exp_dir / "summary.csv", index=False)
    detailed_df.to_csv(exp_dir / "detailed_folds.csv", index=False)

    for metric_name, cmap in (
        ("accuracy", "YlOrRd"),
        ("f1", "Blues"),
        ("auc", "Greens"),
    ):
        heat_df = pd.DataFrame(
            {variant: metric_tables[metric_name][variant] for variant in EXP2_VARIANTS},
            index=dataset_order,
        )
        heat_df.index.name = "dataset"
        heat_df.columns.name = "variant"
        plot_heatmap_df(
            heat_df,
            f"EXP2 {metric_name.upper()} Heatmap",
            exp_dir / f"{metric_name}_heatmap.png",
            cmap=cmap,
            cbar_label=metric_name.upper(),
        )

    time_df = pd.DataFrame(
        {variant: metric_tables["training_time_s"][variant] for variant in EXP2_VARIANTS},
        index=dataset_order,
    )
    time_df.index.name = "dataset"
    time_df.columns.name = "variant"
    plot_heatmap_df(
        time_df,
        "EXP2 Training Time Heatmap",
        exp_dir / "timing_heatmap.png",
        cmap="Oranges",
        fmt=".1f",
        vmin=float(np.nanmin(time_df.values)) if not np.all(np.isnan(time_df.values)) else 0.0,
        vmax=float(np.nanmax(time_df.values)) if not np.all(np.isnan(time_df.values)) else 1.0,
        cbar_label="seconds",
    )

    radar_series: dict[str, dict[str, float]] = {}
    for variant in EXP2_VARIANTS:
        variant_rows = summary_df[summary_df["variant"] == variant]
        radar_series[variant] = {
            "accuracy": nanmean_or_nan(variant_rows["accuracy_mean"].tolist()),
            "precision": nanmean_or_nan(variant_rows["precision_mean"].tolist()),
            "recall": nanmean_or_nan(variant_rows["recall_mean"].tolist()),
            "f1": nanmean_or_nan(variant_rows["f1_mean"].tolist()),
            "auc": nanmean_or_nan(variant_rows["auc_mean"].tolist()),
        }
    plot_multi_radar(
        radar_series,
        exp_dir / "radar_chart.png",
        "EXP2 Global Radar Chart",
    )

    plot_boxplot_long(
        pd.DataFrame(boxplot_rows),
        exp_dir / "boxplot_kfold_uci.png",
        "EXP2 UCI K-Fold Accuracy Distribution",
    )


def run_exp3(exp_dir: Path) -> None:
    torch.set_num_threads(1)
    ensure_dir(exp_dir)
    logger = Logger(log_dir=str(exp_dir))

    params = {
        "seed": 42,
        "batch_size": 32,
        "epochs": 200,
        "size_annealer": 50,
        "num_ising_perceptrons": 5,
        "lambda_init": -0.1,
        "offset_init": 0.0,
        "lr_gamma": 0.001,
        "lr_lambda": 0.005,
        "lr_offset": 0.01,
        "lr_combiner": 0.005,
        "test_size": 0.2,
        "n_samples_per_region": 100,
    }
    sa_settings = make_sa_settings([1, 10], 1, 1000, 1)
    summary_rows: list[dict[str, Any]] = []

    logger.info("=== EXP3: Gamma Initialization Strategies ===")
    for dim in XOR_DIMS:
        dataset_name = f"xor_{dim}d"
        dataset_dir = ensure_dir(exp_dir / dataset_name)
        logger.info(f"\n[EXP3] Dataset: {dataset_name}")

        X_train, X_test, y_train, y_test = prepare_xor_data(
            dim,
            params["n_samples_per_region"],
            params["test_size"],
            params["seed"],
        )

        full_loss_curves: dict[str, list[float]] = {}
        net_loss_curves: dict[str, list[float]] = {}
        full_acc_curves: dict[str, list[float]] = {}
        net_acc_curves: dict[str, list[float]] = {}
        full_metrics_rows: list[dict[str, Any]] = []
        net_metrics_rows: list[dict[str, Any]] = []

        detailed_rows: list[dict[str, Any]] = []
        for strategy_idx, strategy in enumerate(GAMMA_STRATEGIES):
            gamma_init = build_gamma_init(
                strategy,
                X_train,
                params["size_annealer"],
                seed=params["seed"] + strategy_idx,
                hidden_nodes_offset_value=DEFAULT_HIDDEN_OFFSET,
            )
            try:
                full_result = run_full_training(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    seed=params["seed"],
                    batch_size=params["batch_size"],
                    epochs=params["epochs"],
                    size_annealer=params["size_annealer"],
                    lambda_init=params["lambda_init"],
                    offset_init=params["offset_init"],
                    lr_gamma=params["lr_gamma"],
                    lr_lambda=params["lr_lambda"],
                    lr_offset=params["lr_offset"],
                    sa_settings=sa_settings,
                    optimizer_name="adam",
                    loss_mode="bce",
                    gamma_init=gamma_init,
                    logger=logger,
                    log_prefix=f"[EXP3][{dataset_name}][Full][{strategy}]",
                )
                full_loss_curves[strategy] = full_result["training_losses"]
                full_acc_curves[strategy] = full_result["val_accuracies"]
                full_metrics_rows.append(
                    {
                        "strategy": strategy,
                        "accuracy": full_result["accuracy"],
                        "f1": full_result["f1"],
                        "auc": full_result["auc"],
                    }
                )
                detailed_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": "FullIsing",
                        "strategy": strategy,
                        "accuracy": full_result["accuracy"],
                        "precision": full_result["precision"],
                        "recall": full_result["recall"],
                        "f1": full_result["f1"],
                        "auc": full_result["auc"],
                        "training_time_s": full_result["training_time_s"],
                        "convergence_epoch_90pct": convergence_epoch(
                            full_result["val_accuracies"],
                            full_result["accuracy"],
                        ),
                        "error": "",
                    }
                )
            except Exception as exc:
                log_exception(logger, f"[EXP3][{dataset_name}][Full][{strategy}]", exc)
                detailed_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": "FullIsing",
                        "strategy": strategy,
                        "accuracy": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "auc": np.nan,
                        "training_time_s": np.nan,
                        "convergence_epoch_90pct": np.nan,
                        "error": str(exc),
                    }
                )

            try:
                net_result = run_net_training(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    seed=params["seed"],
                    batch_size=params["batch_size"],
                    epochs=params["epochs"],
                    num_ising_perceptrons=params["num_ising_perceptrons"],
                    size_annealer=params["size_annealer"],
                    lambda_init=params["lambda_init"],
                    offset_init=params["offset_init"],
                    lr_gamma=params["lr_gamma"],
                    lr_lambda=params["lr_lambda"],
                    lr_offset=params["lr_offset"],
                    lr_combiner=params["lr_combiner"],
                    sa_settings=sa_settings,
                    optimizer_name="adam",
                    loss_mode="bce",
                    gamma_init=gamma_init,
                    logger=logger,
                    log_prefix=f"[EXP3][{dataset_name}][Net1L][{strategy}]",
                )
                net_loss_curves[strategy] = net_result["training_losses"]
                net_acc_curves[strategy] = net_result["val_accuracies"]
                net_metrics_rows.append(
                    {
                        "strategy": strategy,
                        "accuracy": net_result["accuracy"],
                        "f1": net_result["f1"],
                        "auc": net_result["auc"],
                    }
                )
                detailed_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": "Net1L",
                        "strategy": strategy,
                        "accuracy": net_result["accuracy"],
                        "precision": net_result["precision"],
                        "recall": net_result["recall"],
                        "f1": net_result["f1"],
                        "auc": net_result["auc"],
                        "training_time_s": net_result["training_time_s"],
                        "convergence_epoch_90pct": convergence_epoch(
                            net_result["val_accuracies"],
                            net_result["accuracy"],
                        ),
                        "error": "",
                    }
                )
            except Exception as exc:
                log_exception(logger, f"[EXP3][{dataset_name}][Net1L][{strategy}]", exc)
                detailed_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": "Net1L",
                        "strategy": strategy,
                        "accuracy": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "auc": np.nan,
                        "training_time_s": np.nan,
                        "convergence_epoch_90pct": np.nan,
                        "error": str(exc),
                    }
                )

        detail_df = pd.DataFrame(detailed_rows)
        detail_df.to_csv(dataset_dir / "metrics.csv", index=False)
        summary_rows.extend(detailed_rows)

        plot_dual_overlay_histories(
            full_loss_curves,
            net_loss_curves,
            dataset_dir / "loss_curves.png",
            f"EXP3 Training Loss - {dataset_name}",
            "Loss",
            "FullIsing",
            "Net1L",
        )
        plot_dual_overlay_histories(
            full_acc_curves,
            net_acc_curves,
            dataset_dir / "accuracy_curves.png",
            f"EXP3 Validation Accuracy - {dataset_name}",
            "Accuracy",
            "FullIsing",
            "Net1L",
            ylim=(0, 1.0),
        )

        bar_series = {
            "FullIsing": [
                detail_df[(detail_df["model"] == "FullIsing") & (detail_df["strategy"] == strategy)]["accuracy"].iloc[0]
                if not detail_df[(detail_df["model"] == "FullIsing") & (detail_df["strategy"] == strategy)].empty
                else np.nan
                for strategy in GAMMA_STRATEGIES
            ],
            "Net1L": [
                detail_df[(detail_df["model"] == "Net1L") & (detail_df["strategy"] == strategy)]["accuracy"].iloc[0]
                if not detail_df[(detail_df["model"] == "Net1L") & (detail_df["strategy"] == strategy)].empty
                else np.nan
                for strategy in GAMMA_STRATEGIES
            ],
        }
        plot_grouped_bar(
            GAMMA_STRATEGIES,
            bar_series,
            dataset_dir / "accuracy_bar.png",
            f"EXP3 Final Accuracy - {dataset_name}",
            "Accuracy",
            ylim=(0, 1.0),
        )

        full_df = pd.DataFrame(full_metrics_rows).set_index("strategy") if full_metrics_rows else pd.DataFrame()
        net_df = pd.DataFrame(net_metrics_rows).set_index("strategy") if net_metrics_rows else pd.DataFrame()
        if not full_df.empty:
            full_df.index.name = "strategy"
            full_df.columns.name = "metric"
            plot_heatmap_df(
                full_df[["accuracy", "f1", "auc"]],
                f"FullIsing Metrics - {dataset_name}",
                dataset_dir / "heatmap_fullising.png",
                cmap="YlOrRd",
                cbar_label="Score",
            )
        if not net_df.empty:
            net_df.index.name = "strategy"
            net_df.columns.name = "metric"
            plot_heatmap_df(
                net_df[["accuracy", "f1", "auc"]],
                f"Net1L Metrics - {dataset_name}",
                dataset_dir / "heatmap_net1l.png",
                cmap="YlOrRd",
                cbar_label="Score",
            )

        conv_series = {
            "FullIsing": [
                detail_df[(detail_df["model"] == "FullIsing") & (detail_df["strategy"] == strategy)]["convergence_epoch_90pct"].iloc[0]
                if not detail_df[(detail_df["model"] == "FullIsing") & (detail_df["strategy"] == strategy)].empty
                else np.nan
                for strategy in GAMMA_STRATEGIES
            ],
            "Net1L": [
                detail_df[(detail_df["model"] == "Net1L") & (detail_df["strategy"] == strategy)]["convergence_epoch_90pct"].iloc[0]
                if not detail_df[(detail_df["model"] == "Net1L") & (detail_df["strategy"] == strategy)].empty
                else np.nan
                for strategy in GAMMA_STRATEGIES
            ],
        }
        plot_grouped_bar(
            GAMMA_STRATEGIES,
            conv_series,
            dataset_dir / "convergence_bar.png",
            f"EXP3 Convergence Epoch (90% final acc) - {dataset_name}",
            "Epoch",
        )

    pd.DataFrame(summary_rows).to_csv(exp_dir / "summary.csv", index=False)


def run_exp4(exp_dir: Path) -> None:
    torch.set_num_threads(1)
    ensure_dir(exp_dir)
    logger = Logger(log_dir=str(exp_dir))

    params = {
        "seed": 42,
        "batch_size": 32,
        "phase1_epochs": 100,
        "total_epochs": 250,
        "size_annealer": 50,
        "num_ising_perceptrons": 5,
        "lambda_init": -0.1,
        "offset_init": 0.0,
        "test_size": 0.2,
        "n_samples_per_region": 100,
    }
    sa_settings = make_sa_settings([1, 10], 1, 1000, 1)
    summary_rows: list[dict[str, Any]] = []

    logger.info("=== EXP4: Staged Training ===")
    for dim in XOR_DIMS:
        dataset_name = f"xor_{dim}d"
        dataset_dir = ensure_dir(exp_dir / dataset_name)
        logger.info(f"\n[EXP4] Dataset: {dataset_name}")

        X_train, X_test, y_train, y_test = prepare_xor_data(
            dim,
            params["n_samples_per_region"],
            params["test_size"],
            params["seed"],
        )

        result_builders = {
            "staged_FullIsing": lambda: run_full_training_staged(
                X_train,
                y_train,
                X_test,
                y_test,
                seed=params["seed"],
                batch_size=params["batch_size"],
                size_annealer=params["size_annealer"],
                lambda_init=params["lambda_init"],
                offset_init=params["offset_init"],
                sa_settings=sa_settings,
                phase1_epochs=params["phase1_epochs"],
                total_epochs=params["total_epochs"],
                baseline=False,
                logger=logger,
                log_prefix=f"[EXP4][{dataset_name}][staged_FullIsing]",
            ),
            "baseline_FullIsing": lambda: run_full_training_staged(
                X_train,
                y_train,
                X_test,
                y_test,
                seed=params["seed"],
                batch_size=params["batch_size"],
                size_annealer=params["size_annealer"],
                lambda_init=params["lambda_init"],
                offset_init=params["offset_init"],
                sa_settings=sa_settings,
                phase1_epochs=params["phase1_epochs"],
                total_epochs=params["total_epochs"],
                baseline=True,
                logger=logger,
                log_prefix=f"[EXP4][{dataset_name}][baseline_FullIsing]",
            ),
            "staged_Net1L": lambda: run_net_training_staged(
                X_train,
                y_train,
                X_test,
                y_test,
                seed=params["seed"],
                batch_size=params["batch_size"],
                num_ising_perceptrons=params["num_ising_perceptrons"],
                size_annealer=params["size_annealer"],
                lambda_init=params["lambda_init"],
                offset_init=params["offset_init"],
                sa_settings=sa_settings,
                phase1_epochs=params["phase1_epochs"],
                total_epochs=params["total_epochs"],
                baseline=False,
                logger=logger,
                log_prefix=f"[EXP4][{dataset_name}][staged_Net1L]",
            ),
            "baseline_Net1L": lambda: run_net_training_staged(
                X_train,
                y_train,
                X_test,
                y_test,
                seed=params["seed"],
                batch_size=params["batch_size"],
                num_ising_perceptrons=params["num_ising_perceptrons"],
                size_annealer=params["size_annealer"],
                lambda_init=params["lambda_init"],
                offset_init=params["offset_init"],
                sa_settings=sa_settings,
                phase1_epochs=params["phase1_epochs"],
                total_epochs=params["total_epochs"],
                baseline=True,
                logger=logger,
                log_prefix=f"[EXP4][{dataset_name}][baseline_Net1L]",
            ),
        }
        results_map: dict[str, dict[str, Any] | None] = {}
        dataset_rows: list[dict[str, Any]] = []
        for variant, builder in result_builders.items():
            try:
                results_map[variant] = builder()
            except Exception as exc:
                log_exception(logger, f"[EXP4][{dataset_name}][{variant}]", exc)
                results_map[variant] = None

            result = results_map[variant]
            if result is None:
                dataset_rows.append(
                    {
                        "dataset": dataset_name,
                        "variant": variant,
                        "final_accuracy": np.nan,
                        "final_precision": np.nan,
                        "final_recall": np.nan,
                        "final_f1": np.nan,
                        "final_auc": np.nan,
                        "phase1_accuracy": np.nan,
                        "phase1_precision": np.nan,
                        "phase1_recall": np.nan,
                        "phase1_f1": np.nan,
                        "phase1_auc": np.nan,
                        "training_time_s": np.nan,
                    }
                )
                continue

            phase1_metrics = result["captured_metrics"].get(params["phase1_epochs"], {})
            dataset_rows.append(
                {
                    "dataset": dataset_name,
                    "variant": variant,
                    "final_accuracy": result["accuracy"],
                    "final_precision": result["precision"],
                    "final_recall": result["recall"],
                    "final_f1": result["f1"],
                    "final_auc": result["auc"],
                    "phase1_accuracy": phase1_metrics.get("accuracy", np.nan),
                    "phase1_precision": phase1_metrics.get("precision", np.nan),
                    "phase1_recall": phase1_metrics.get("recall", np.nan),
                    "phase1_f1": phase1_metrics.get("f1", np.nan),
                    "phase1_auc": phase1_metrics.get("auc", np.nan),
                    "training_time_s": result["training_time_s"],
                }
            )

        summary_rows.extend(dataset_rows)
        valid_loss_curves = {
            variant: result["training_losses"]
            for variant, result in results_map.items()
            if result is not None
        }
        valid_acc_curves = {
            variant: result["val_accuracies"]
            for variant, result in results_map.items()
            if result is not None
        }
        if valid_loss_curves:
            plot_overlay_curves(
                valid_loss_curves,
                dataset_dir / "loss_curves.png",
                f"EXP4 Training Loss - {dataset_name}",
                "Loss",
                vertical_line=params["phase1_epochs"],
            )
        if valid_acc_curves:
            plot_overlay_curves(
                valid_acc_curves,
                dataset_dir / "accuracy_curves.png",
                f"EXP4 Validation Accuracy - {dataset_name}",
                "Accuracy",
                vertical_line=params["phase1_epochs"],
                ylim=(0, 1.0),
            )
        plot_grouped_bar(
            ["FullIsing", "Net1L"],
            {
                "staged": [
                    results_map["staged_FullIsing"]["accuracy"] if results_map["staged_FullIsing"] is not None else np.nan,
                    results_map["staged_Net1L"]["accuracy"] if results_map["staged_Net1L"] is not None else np.nan,
                ],
                "baseline": [
                    results_map["baseline_FullIsing"]["accuracy"] if results_map["baseline_FullIsing"] is not None else np.nan,
                    results_map["baseline_Net1L"]["accuracy"] if results_map["baseline_Net1L"] is not None else np.nan,
                ],
            },
            dataset_dir / "accuracy_bar.png",
            f"EXP4 Final Accuracy - {dataset_name}",
            "Accuracy",
            ylim=(0, 1.0),
        )
        pd.DataFrame(dataset_rows).to_csv(dataset_dir / "metrics.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(exp_dir / "summary.csv", index=False)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(f"experiments/experiments_{timestamp}")
    base_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("exp1_matrix_fullising_vs_net1l", run_exp1),
        ("exp2_offset_sweep", run_exp2),
        ("exp3_gamma_init", run_exp3),
        ("exp4_staged_training", run_exp4),
    ]

    processes: list[multiprocessing.Process] = []
    for name, func in experiments:
        exp_dir = base_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        process = multiprocessing.Process(target=func, args=(exp_dir,), name=name)
        processes.append(process)
        process.start()
        print(f"[MAIN] Avviato esperimento: {name} (PID {process.pid})")

    for process in processes:
        process.join()
        status = "OK" if process.exitcode == 0 else f"ERRORE (exit={process.exitcode})"
        print(f"[MAIN] Completato: {process.name} -> {status}")

    print(f"\n[MAIN] Tutti gli esperimenti completati. Risultati in: {base_dir.resolve()}")
