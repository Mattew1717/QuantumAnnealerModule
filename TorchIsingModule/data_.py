""" Dataset class for ising learning model,
which also return the index for mini batches. """
from __future__ import annotations
import torch
from torch.utils.data import Dataset
import numpy as np

from .utils import GammaInitialization, utils
from dimod import ExactSolver


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
    _ising_configs = list

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
