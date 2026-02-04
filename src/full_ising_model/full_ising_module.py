import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import threading

from .utils import utils, HiddenNodesInitialization, offset
from .annealers import (
    Annealer,
    AnnealerFactory,
    AnnealerType,
    AnnealingSettings,
)


class IsingEnergyFunction(Function):
    """
    Custom autograd Function computing Ising energy via an external annealer.
    Backward propagates gradients only to gamma.
    """

    @staticmethod
    def forward(
        ctx,
        thetas: torch.Tensor,
        gammas: torch.Tensor,
        annealer: Annealer,
        num_workers: int,
    ):
        batch_size, _ = thetas.shape
        energies_bulk = [None] * batch_size
        configurations_bulk = [None] * batch_size

        # Convert gamma to couplings ONCE for the entire batch (same for all samples)
        J = utils.gamma_to_couplings(gammas.detach().cpu().numpy())

        def worker(start_idx: int, end_idx: int):
            """Process a slice of the batch."""
            for i in range(start_idx, end_idx):
                theta = thetas[i].detach().cpu().numpy()
                h = utils.vector_to_biases(theta)
                # Use the pre-computed J (shared across all samples in batch)
                sample_set = annealer.sample(h, J)
                energies_bulk[i] = sample_set.first.energy
                configurations_bulk[i] = list(sample_set.first.sample.values())

        # Split batch across workers
        chunk_size = (batch_size + num_workers - 1) // num_workers
        threads = []
        for w in range(num_workers):
            start = w * chunk_size
            end = min((w + 1) * chunk_size, batch_size)
            if start >= end:
                continue
            t = threading.Thread(target=worker, args=(start, end))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        energies_bulk = torch.tensor(
            energies_bulk, dtype=thetas.dtype, device=thetas.device
        )
        configurations_bulk = torch.tensor(
            np.array(configurations_bulk), dtype=thetas.dtype, device=thetas.device
        )

        # Save configurations for backward
        ctx.save_for_backward(configurations_bulk)

        return energies_bulk

    @staticmethod
    def backward(ctx, grad_energies_bulk: torch.Tensor):
        """
        Gradient of gamma using outer products of spin configurations.
        """
        (configurations_bulk,) = ctx.saved_tensors
        batch_size, _ = configurations_bulk.shape

        z_i = configurations_bulk.unsqueeze(2)
        z_j = configurations_bulk.unsqueeze(1)
        outer_prod = z_i * z_j

        grad_gamma_per_sample = grad_energies_bulk.view(batch_size, 1, 1) * outer_prod
        grad_gamma = torch.mean(grad_gamma_per_sample, dim=0)
        grad_gamma = utils.make_upper_triangular_torch(grad_gamma)

        # No gradient for thetas and annealer
        return None, grad_gamma, None, None


class IsingLayer(nn.Module):
    """Ising energy layer wrapping an annealer."""

    def __init__(
        self,
        size_annealer: int,
        annealer_type: AnnealerType,
        annealing_settings: AnnealingSettings | None = None,
        num_workers: int = 1,
        **annealer_kwargs,
    ):
        super().__init__()
        self.size_annealer = size_annealer
        self.num_workers = num_workers

        # Create annealer instance
        self.annealer = AnnealerFactory.create(
            annealer_type=annealer_type,
            size=size_annealer,
            settings=annealing_settings,
            **annealer_kwargs,
        )

    def forward(self, thetas: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return IsingEnergyFunction.apply(
            thetas, gamma, self.annealer, self.num_workers
        )


class FullIsingModule(nn.Module):
    """Full Ising model with scaling of the energy."""

    def __init__(
        self,
        size_annealer: int,
        annealer_type: AnnealerType,
        annealing_settings: AnnealingSettings | None = None,
        lambda_init: float = 1.0,
        offset_init: float = 0.0,
        gamma_init: torch.Tensor | None = None,
        num_workers: int = 1,
        hidden_nodes_offset_value: float = 0.1,
        **annealer_kwargs,
    ):
        super().__init__()
        self.size_annealer = size_annealer

        # Setup hidden nodes initialization with offset mode
        self.hidden_nodes_config = HiddenNodesInitialization("function")
        self.hidden_nodes_config.function = offset
        self.hidden_nodes_config.fun_args = (hidden_nodes_offset_value,)

        self.ising_layer = IsingLayer(
            size_annealer=size_annealer,
            annealer_type=annealer_type,
            annealing_settings=annealing_settings,
            num_workers=num_workers,
            **annealer_kwargs,
        )

        # Learnable coupling matrix (upper triangular)
        if gamma_init is None:
            initial_gamma = torch.randn((size_annealer, size_annealer), dtype=torch.float32) * 0.01
            initial_gamma = torch.triu(initial_gamma, diagonal=1)
        else:
            initial_gamma = gamma_init
        self.gamma = nn.Parameter(initial_gamma)

        # Learnable scaling and offset
        self.lmd = nn.Parameter(torch.tensor(lambda_init))
        self.offset = nn.Parameter(torch.tensor(offset_init))

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # Automatically resize if needed
        thetas = self.hidden_nodes_config.resize(thetas, self.size_annealer)

        E0 = self.ising_layer(thetas, self.gamma)
        return self.lmd * E0 + self.offset
