from logger import Logger
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os

class DatasetManager:
    
    def __init__(self):
        self.logger = Logger()
        pass

    def load_csv_dataset(self, csv_path):
        """ Load and preprocess CSV dataset. Expects last column as labels. Binary classification only. Classes 0-1."""

        self.logger.info(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)

        #Shuffle dataset
        df = df.sample(frac=1, random_state=int(os.getenv('RANDOM_SEED'))).reset_index(drop=True)

        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.float32)
        
        # Normalize features
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        self.logger.info(f"Dataset num features: X={X.shape},")
        self.logger.info(f"Class distribution: {np.bincount(y.astype(int))}")

        return X, y

    def generate_k_folds(self, X, y, k):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(os.getenv('RANDOM_SEED')))
        folds = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            folds.append((X_train, y_train, X_test, y_test))

        return folds
    
    def create_dataloader(self, X_train, y_train, X_test, y_test, params):
        """Prepare datasets with hidden nodes initialization."""
        dataset = SimpleDataset()
        test_set = SimpleDataset()

        dataset.x = torch.tensor(X_train, dtype=torch.float32)
        dataset.y = torch.tensor(y_train, dtype=torch.float32)
        dataset.data_size = len(X_train[0])
        dataset.len = len(y_train)

        test_set.x = torch.tensor(X_test, dtype=torch.float32)
        test_set.y = torch.tensor(y_test, dtype=torch.float32)
        test_set.data_size = len(X_test[0])
        test_set.len = len(y_test)

        # Hidden nodes initialization
        hn = HiddenNodesInitialization('function')
        hn.function = SimpleDataset.offset
        hn.fun_args = [-0.02]

        if params['model_size'] == -1:
            params['model_size'] = dataset.data_size if dataset.data_size > 10 else 10
            
        # Resize datasets
        if params['partition_input']:
            target_size = params['model_size'] * params['num_ising_perceptrons']
        else:
            target_size = params['model_size']
        dataset.resize(target_size, hn)
        test_set.resize(target_size, hn)

        # Update data_size after resize
        dataset.data_size = target_size
        test_set.data_size = target_size

        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(dataset.x, dataset.y),
            batch_size=params['batch_size'],
            shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(test_set.x, test_set.y),
            batch_size=params['batch_size'],
            shuffle=False
        )
        return dataset, test_set, train_loader, test_loader


class HiddenNodesInitialization:
    """Hidden nodes initialization settings for the model."""

    mode: str = "repeat"
    random_range: tuple[torch.Tensor, torch.Tensor] = (-1, 1)
    function: callable = None
    fun_args: tuple = None

    def __init__(self, mode) -> None:
        self._function = None
        if mode == "repeat" or "random" or "zeros":
            self.mode = mode
        elif mode == "function":
            self.mode = mode
            self._function = lambda theta, index_new: theta[
                index_new % len(theta)
            ]
        else:
            msg = "invalid gamma initialization mode"
            raise ValueError(msg)

class SimpleDataset(Dataset):
    """ Dataset class for ising learning model """
    x = torch.Tensor
    y = torch.Tensor
    len = int
    data_size = int
    _gamma_data = np.array

    def __init__(self):
        super().__init__()

    def create_data_fun(
        self, function: callable, num_samples: int, ranges: list
    ):
        """ Create Dataset containing data from a given function"""

        xs = []
        ys = []
        for i in range(num_samples):
            x = [
                np.random.uniform(value_range[0], value_range[1])
                for value_range in ranges
            ]
            xs.append(torch.Tensor(x))
            try:
                ys.append(function(*x))
            except TypeError:
                msg = "number of arguments in function does not match number of ranges"
                raise TypeError(msg)
        self.x = torch.stack(xs)
        self.y = torch.Tensor(ys)
        self.len = len(self.y)
        self.data_size = len(x)

    def create_data_rand(self,
                         size: int,
                         num_samples: int,
                         value_range_biases: tuple[float, float] = (-1, 1),
                         value_range_energies: tuple[float, float] = (-1, 0),
                         seed: int = 42,
                         ):
        np.random.seed(seed)
        self.x = torch.Tensor(np.random.uniform(value_range_biases[0], value_range_biases[1], (num_samples, size)))
        self.y = torch.Tensor(np.random.uniform(value_range_energies[0], value_range_energies[1], num_samples))
        self.len = len(self.y)
        self.data_size = size

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], idx

    def resize(
        self, size: int, hidden_nodes: HiddenNodesInitialization
    ) -> None:
        """Resize the dataset to the given size by adding hidden nodes."""
        if size < self.data_size:
            msg = "size must be greater or equal to the size of the dataset"
            raise ValueError(msg)
        elif size == self.data_size:
            return

        if hidden_nodes.mode == "random":
            hidden_nodes._random_range = (torch.min(self.x), torch.max(self.x))
        if hidden_nodes.mode == "function":
            if hidden_nodes.function is None:
                msg = "function must be given when mode is function"
                raise ValueError(msg)

        x_new = []
        for theta in self.x:
            if hidden_nodes.mode == "function":
                if hidden_nodes.fun_args is None:
                    x_new.append(
                        torch.Tensor(
                            [
                                hidden_nodes.function(theta, index_new)
                                for index_new in range(size)
                            ]
                        )
                    )
                else:
                    x_new.append(
                        torch.Tensor(
                            [
                                hidden_nodes.function(
                                    theta, index_new, hidden_nodes.fun_args
                                )
                                for index_new in range(size)
                            ]
                        )
                    )
            else:
                x_new.append(
                    torch.Tensor(
                        [
                            SimpleDataset._create_entry(
                                theta, index_new, hidden_nodes
                            )
                            for index_new in range(size)
                        ]
                    )
                )
        self.x = torch.stack(x_new)

    @staticmethod
    def _create_entry(
        theta: torch.Tensor,
        index_new: int,
        hidden_nodes: HiddenNodesInitialization,
    ) -> float:
        """Create a new value for the given index of the theta tensor."""
        multiple = index_new // len(theta)

        if hidden_nodes.mode == "zeros":
            if multiple == 0:
                return theta[index_new]
            else:
                return 0
        elif hidden_nodes.mode == "repeat":
            return theta[index_new % len(theta)]
        elif hidden_nodes.mode == "random":
            if multiple == 0:
                return theta[index_new]
            else:
                return np.random.uniform(
                    hidden_nodes.random_range[0], hidden_nodes.random_range[1]
                )

    @staticmethod
    def lin_scaling(
        theta: torch.Tensor, index_new: int, fun_args: tuple
    ) -> float:
        """Linear scaling of the theta vector."""
        mod = index_new % len(theta)
        mult = index_new // len(theta) + 1
        return (theta[mod] * (mult * fun_args[0]))**3 + fun_args[1]

    @staticmethod
    def offset(theta: torch.Tensor, index_new: int, fun_args: tuple) -> float:
        """
        Calculates a new value for the given index of
        the theta tensor by adding an offset.
        """
        offset = fun_args[0]
        return theta[index_new % len(theta)] + index_new // len(theta) * offset

    @staticmethod
    def offset_fixed(
        theta: torch.Tensor, index_new: int, fun_args: tuple
    ) -> float:
        """
        Calculates a new value for the given index of the theta tensor by adding
        an offset.
        """
        offset = fun_args[0]
        if index_new == 19:
            return 10000
        return theta[index_new % len(theta)] + index_new // len(theta) * offset

    @staticmethod
    def offset_random(
        theta: torch.Tensor, index_new: int, fun_args: tuple
    ) -> float:
        """
        Calculates a new value for the given index of the theta tensor by adding
        an offset.
        """
        offset = fun_args[0]
        if index_new == 0:
            return 10
        return np.random.uniform(-offset, offset)
