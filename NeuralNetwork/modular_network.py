import torch
import torch.nn as nn
import numpy as np

from full_ising_model.annealers import AnnealingSettings, AnnealerType
from full_ising_model.full_ising_module import FullIsingModule


class ModularNetwork(nn.Module):
    """Multiple parallel FullIsingModule perceptrons combined by a final Linear layer."""

    def __init__(
        self,
        num_ising_perceptrons: int,
        size_annealer: int,
        annealing_settings: AnnealingSettings | None,
        annealer_type: AnnealerType,
        lambda_init: float,
        offset_init: float,
        hidden_nodes_offset_value: float,
        num_workers: int,
        combiner_bias: bool,
        partition_input: bool,
        random_seed: int,
        profile: str | None = None,
        num_reads: int | None = None,
    ):
        super().__init__()
        self.num_ising_perceptrons = num_ising_perceptrons
        self.size_annealer = size_annealer
        self.partition_input = partition_input

        rng = np.random.default_rng(random_seed)

        self.ising_perceptrons_layer = nn.ModuleList()
        for _ in range(num_ising_perceptrons):
            lambda_i = lambda_init + float(rng.uniform(-0.1, 0.1))
            offset_i = offset_init + float(rng.uniform(-0.1, 0.1))

            module = FullIsingModule(
                size_annealer=size_annealer,
                annealer_type=annealer_type,
                annealing_settings=annealing_settings,
                lambda_init=lambda_i,
                offset_init=offset_i,
                num_workers=num_workers,
                hidden_nodes_offset_value=hidden_nodes_offset_value,
                profile=profile,
                num_reads=num_reads,
            )
            self.ising_perceptrons_layer.append(module)

        self.combiner_layer = nn.Linear(num_ising_perceptrons, 1, bias=combiner_bias)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        if self.partition_input:
            total_size = thetas.shape[1]
            chunk_size = total_size // self.num_ising_perceptrons
            if chunk_size == 0:
                raise ValueError(
                    f"partition_input=True requires total_size ({total_size}) >= "
                    f"num_ising_perceptrons ({self.num_ising_perceptrons})."
                )

            perceptron_outputs = []
            for i, perceptron in enumerate(self.ising_perceptrons_layer):
                start = i * chunk_size
                end = (
                    (i + 1) * chunk_size
                    if i < self.num_ising_perceptrons - 1
                    else total_size
                )
                perceptron_outputs.append(perceptron(thetas[:, start:end]))
        else:
            perceptron_outputs = [p(thetas) for p in self.ising_perceptrons_layer]

        stacked = torch.stack(perceptron_outputs, dim=1)
        combined = self.combiner_layer(stacked)
        return combined.squeeze(-1)
