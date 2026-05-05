import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from Inference.utils.logger import Logger
from full_ising_model.utils import HiddenNodesInitialization


def _balance_split(X, y, rng):
    """Random undersampling of the majority class to match the minority count."""
    y_int = y.astype(int)
    idx0 = np.where(y_int == 0)[0]
    idx1 = np.where(y_int == 1)[0]
    n = min(len(idx0), len(idx1))
    sel0 = rng.choice(idx0, size=n, replace=False)
    sel1 = rng.choice(idx1, size=n, replace=False)
    sel = np.concatenate([sel0, sel1])
    rng.shuffle(sel)
    return X[sel], y[sel]


class DatasetManager:
    def __init__(self):
        self.logger = Logger()

    def load_csv_dataset(self, csv_path: str, random_seed: int):
        """Load and preprocess CSV dataset. Last column = labels (binary {0,1}; -1 -> 0)."""
        self.logger.info(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)

        label_col = df.columns[-1]
        if (df[label_col] == -1).any():
            self.logger.info("Converting -1 labels to 0 in last column")
            df[label_col] = df[label_col].replace(-1, 0)

        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.float32)

        self.logger.info(f"Dataset num features: X={X.shape},")
        self.logger.info(f"Class distribution: {np.bincount(y.astype(int))}")
        return X, y

    def generate_k_folds(self, X, y, k: int, random_seed: int):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
        rng = np.random.default_rng(random_seed)
        folds = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, y_train = _balance_split(X[train_idx], y[train_idx], rng)
            X_test, y_test = _balance_split(X[test_idx], y[test_idx], rng)
            folds.append((X_train, y_train, X_test, y_test))
        return folds


class SimpleDataset(Dataset):
    """Dataset class for Ising learning model. Resize uses HiddenNodesInitialization (offset rule)."""

    def __init__(self):
        super().__init__()
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None
        self.len: int = 0
        self.data_size: int = 0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], idx

    def resize(self, size: int, hidden_nodes: HiddenNodesInitialization) -> None:
        if size < self.data_size:
            raise ValueError("size must be >= dataset feature dimension")
        if size == self.data_size:
            return
        self.x = hidden_nodes.resize(self.x, size)
        self.data_size = size
