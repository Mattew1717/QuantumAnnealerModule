import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)


def flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    """Safely flatten model output to one logit per sample."""
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits.squeeze(1)
    raise ValueError(f"Unexpected model output shape: {logits.shape}")


METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc']


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
                     model_name1='FullIsingModule', model_name2='Network_1L'):
    """Save mean +/- std metrics for two models to CSV."""
    import os
    import pandas as pd
    rows = []
    for i, ds_name in enumerate(dataset_names):
        for model_name, model_metrics in [(model_name1, all_metrics_1[i]),
                                           (model_name2, all_metrics_2[i])]:
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


def fquad(x):
    return -(x - 0.5) ** 2


def flin(x):
    return 2 * x - 6


def fcub(x):
    return (x - 0.5) ** 3


def flog(x):
    return np.log(x)

def generate_xor_balanced(dim, n_samples_dim=1000, shuffle=True, random_seed=42):
    """Generate XOR data in U[0,1]^d with balanced classes."""
    if random_seed is not None:
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