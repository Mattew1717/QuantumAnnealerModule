import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class DatasetManager:
    def __init__(self):
        # Reuse the singleton without re-initializing it (which would drop handlers).
        self.logger = logging.getLogger('IsingComparison')

    def load_csv_dataset(self, csv_path: str, random_seed: int):
        """Load and preprocess CSV dataset. Last column = labels (binary {0,1}; -1 -> 0)."""
        self.logger.info(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path, header=None)

        first_row_numeric = pd.to_numeric(df.iloc[0], errors='coerce').notna().all()
        if not first_row_numeric:
            self.logger.info("Detected header row, reloading with header=0")
            df = pd.read_csv(csv_path, header=0)
            df.columns = range(df.shape[1])

        df = df.apply(pd.to_numeric)

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
        folds = []
        for train_idx, test_idx in skf.split(X, y):
            folds.append((X[train_idx], y[train_idx], X[test_idx], y[test_idx]))
        return folds
