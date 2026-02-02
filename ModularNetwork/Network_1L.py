import torch
import torch.nn as nn
import numpy as np
from full_ising_model.annealers import AnnealingSettings, AnnealerType
from full_ising_model.full_ising_module import FullIsingModule

class MultiIsingNetwork(nn.Module):

    def __init__(self,
             num_ising_perceptrons: int,
             size_annealer: int,
             annealing_settings: AnnealingSettings,
             annealer_type: AnnealerType = AnnealerType.SIMULATED,
             lambda_init: float = 1.0,
             offset_init: float = 0.0,
             combiner_bias: bool = True,
             partition_input: bool = False):

        super().__init__()
        self.num_ising_perceptrons = num_ising_perceptrons
        self.size_annealer = size_annealer
        self.partition_input = partition_input
    
        self.ising_perceptrons_layer = nn.ModuleList()
        for i in range(num_ising_perceptrons):
            # Noise to the initial parameters
            lambda_i = lambda_init + np.random.uniform(-0.1, 0.1)
            offset_i = offset_init + np.random.uniform(-0.1, 0.1)

            module = FullIsingModule(
                size_annealer=size_annealer,
                annealer_type=annealer_type,
                annealing_settings=annealing_settings,
                lambda_init=lambda_i,
                offset_init=offset_i
            )

            self.ising_perceptrons_layer.append(module)

        self.combiner_layer = nn.Linear(num_ising_perceptrons, 1, bias=combiner_bias)


    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        if self.partition_input:
            total_size = thetas.shape[1]
            chunk_size = total_size // self.num_ising_perceptrons

            perceptron_outputs = []
            for i, perceptron in enumerate(self.ising_perceptrons_layer):
                start = i * chunk_size
                end = (i + 1) * chunk_size
                chunk = thetas[:, start:end]
                perceptron_outputs.append(perceptron(chunk))
        else:
            perceptron_outputs = []
            for perceptron in self.ising_perceptrons_layer:
                perceptron_outputs.append(perceptron(thetas))

        stacked_outputs = torch.stack(perceptron_outputs, dim=1)
        combined_output = self.combiner_layer(stacked_outputs) 

        return combined_output.squeeze(-1)
