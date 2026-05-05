import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

from full_ising_model.annealers import AnnealingSettings


METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc']


def flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    """Flatten model output to one logit per sample."""
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits.squeeze(1)
    raise ValueError(f"Unexpected model output shape: {logits.shape}")


def compute_metrics(y_true, probs):
    """Compute accuracy, precision, recall, F1, AUC from predicted probabilities."""
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float('nan')
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}


def save_metrics_csv(dataset_names, all_metrics_1, all_metrics_2, output_dir,
                     model_name1: str, model_name2: str):
    """Save mean ± std metrics for two models to CSV."""
    rows = []
    for i, ds_name in enumerate(dataset_names):
        for model_name, model_metrics in [
            (model_name1, all_metrics_1[i]),
            (model_name2, all_metrics_2[i]),
        ]:
            row = {'dataset': ds_name, 'model': model_name}
            for m in METRICS:
                vals = model_metrics[m]
                row[f'{m}_mean'] = np.nanmean(vals)
                row[f'{m}_std'] = np.nanstd(vals)
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(str(output_dir), 'metrics_summary.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    return csv_path


def standardize_train_test(X_train, X_test):
    """Standardize features using statistics fitted on the training split only."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def build_annealing_settings(params: dict) -> AnnealingSettings:
    """Build an AnnealingSettings from a strict params dict."""
    return AnnealingSettings(
        beta_range=params['sa_beta_range'],
        num_reads=params['num_reads'],
        num_sweeps=params['sa_num_sweeps'],
        num_sweeps_per_beta=params['sa_sweeps_per_beta'],
    )


def resolve_model_size(n_features: int, params: dict) -> int:
    """Return params['model_size'] if explicit, else max(n_features, params['minimum_model_size'])."""
    if params['model_size'] == -1:
        return max(n_features, params['minimum_model_size'])
    return params['model_size']


def generate_xor_balanced(dim: int, n_samples_dim: int, shuffle: bool, random_seed: int):
    """Generate XOR data in U[0,1]^d with balanced classes."""
    np.random.seed(random_seed)
    samples = np.random.random(size=(2 ** dim * n_samples_dim, dim))
    for i in range(2 ** dim):
        signs = np.array([1 if int((i // 2 ** d) % 2) == 0 else -1 for d in range(dim)])
        samples[i * n_samples_dim:(i + 1) * n_samples_dim] *= signs
    labels = np.sign(np.prod(samples, axis=1))
    if shuffle:
        perm = np.random.permutation(2 ** dim * n_samples_dim)
        samples = samples[perm]
        labels = labels[perm]
    labels = (labels > 0).astype(int)
    return samples, labels
