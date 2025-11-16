import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
#  Plot: Training Loss Curves per Dataset
# ============================================================
def plot_training_loss(dataset_names, single_losses, neural_losses, run_timestamp):
    plots_dir = Path("Plots")
    plots_dir.mkdir(exist_ok=True)

    for name, s_loss, n_loss in zip(dataset_names, single_losses, neural_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(s_loss, label="Single Ising", linewidth=2)
        plt.plot(n_loss, label="Neural Ising Network", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss – {name} [{run_timestamp}]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = plots_dir / f"loss_curve_{name}_{run_timestamp}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()


# ============================================================
#  Heatmap modelli × dataset
# ============================================================
def plot_heatmap(dataset_names, acc_single, acc_neural, run_timestamp):
    plots_dir = Path("Plots")
    plots_dir.mkdir(exist_ok=True)

    data = np.array([acc_single, acc_neural])
    labels = ["Single Ising", "Neural Ising Network"]

    plt.figure(figsize=(12, 4))
    sns.heatmap(
        data,
        annot=True,
        xticklabels=dataset_names,
        yticklabels=labels,
        cmap="viridis",
        vmin=0,
        vmax=1
    )
    plt.title(f"Accuracy Heatmap [{run_timestamp}]")
    out = plots_dir / f"heatmap_{run_timestamp}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
#  Improvement bars (green/red)
# ============================================================
def plot_improvement_bars(dataset_names, acc_single, acc_neural, run_timestamp):
    plots_dir = Path("Plots")
    plots_dir.mkdir(exist_ok=True)

    diff = np.array(acc_neural) - np.array(acc_single)
    colors = ["green" if d > 0 else "red" for d in diff]

    plt.figure(figsize=(12, 5))
    plt.bar(dataset_names, diff, color=colors)
    plt.axhline(0, linestyle="--", color="black")
    plt.ylabel("Accuracy Improvement (Neural – Single)")
    plt.title(f"Model Improvement by Dataset [{run_timestamp}]")
    plt.xticks(rotation=45, ha="right")

    out = plots_dir / f"improvement_bars_{run_timestamp}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
#  Scatter + parity line
# ============================================================
def plot_parity_scatter(dataset_names, acc_single, acc_neural, run_timestamp):
    plots_dir = Path("Plots")
    plots_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(acc_single, acc_neural, s=60)

    # Parity line
    lims = [min(acc_single + acc_neural), max(acc_single + acc_neural)]
    plt.plot(lims, lims, 'k--')

    for i, name in enumerate(dataset_names):
        plt.text(acc_single[i] + 0.005, acc_neural[i] + 0.005, name, fontsize=8)

    plt.xlabel("Single Ising Accuracy")
    plt.ylabel("Neural Ising Network Accuracy")
    plt.title(f"Parity Scatter Plot [{run_timestamp}]")
    plt.grid(True, alpha=0.3)

    out = plots_dir / f"parity_scatter_{run_timestamp}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
#  Violin Plot delle distribuzioni accuracies
# ============================================================
def plot_violin(acc_single, acc_neural, run_timestamp):
    plots_dir = Path("Plots")
    plots_dir.mkdir(exist_ok=True)

    data = [acc_single, acc_neural]
    labels = ["Single Ising", "Neural Ising Network"]

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=data)
    plt.xticks([0, 1], labels)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Distribution Violin Plot [{run_timestamp}]")
    plt.grid(axis="y", alpha=0.3)

    out = plots_dir / f"violin_plot_{run_timestamp}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
#  Dashboard multipannello
# ============================================================
def plot_dashboard(dataset_names, acc_single, acc_neural, run_timestamp):
    plots_dir = Path("Plots")
    plots_dir.mkdir(exist_ok=True)

    diff = np.array(acc_neural) - np.array(acc_single)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1 — Heatmap
    sns.heatmap(
        np.array([acc_single, acc_neural]),
        annot=True,
        xticklabels=dataset_names,
        yticklabels=["Single Ising", "Neural Ising Network"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("Accuracy Heatmap")

    # 2 — Improvement bars
    colors = ["green" if d > 0 else "red" for d in diff]
    axes[0, 1].bar(dataset_names, diff, color=colors)
    axes[0, 1].set_title("Improvement Bars")
    axes[0, 1].axhline(0, linestyle="--", color="black")
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3 — Parity Scatter
    axes[0, 2].scatter(acc_single, acc_neural, s=60)
    lims = [min(acc_single + acc_neural), max(acc_single + acc_neural)]
    axes[0, 2].plot(lims, lims, 'k--')
    axes[0, 2].set_title("Parity Scatter")
    for i, name in enumerate(dataset_names):
        axes[0, 2].text(acc_single[i] + 0.005, acc_neural[i] + 0.005, name, fontsize=8)

    # 4 — Violin plot
    sns.violinplot(data=[acc_single, acc_neural], ax=axes[1, 0])
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(["Single Ising", "Neural Ising Network"])
    axes[1, 0].set_title("Accuracy Violin Plot")

    # 5 & 6 left empty or decorated
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')

    fig.suptitle(f"Multi-Panel Summary Dashboard [{run_timestamp}]", fontsize=16)
    fig.tight_layout()

    out = plots_dir / f"dashboard_{run_timestamp}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
