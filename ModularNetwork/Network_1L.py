import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from IsingModule.utils import AnnealingSettings
from torch.utils.data import DataLoader
from IsingModule.FullIsingModule import FullIsingModule

class MultiIsingNetwork(nn.Module):

    def __init__(self,
             num_ising_perceptrons: int,
             sizeAnnealModel: int,
             anneal_settings: AnnealingSettings,
             lambda_init: float = 1.0,
             offset_init: float = 0.0,
             combiner_bias: bool = True,
             partition_input: bool = False):

        super().__init__()
        self.num_ising_perceptrons = num_ising_perceptrons
        self.sizeAnnealModel = sizeAnnealModel
        self.partition_input = partition_input
    
        self.ising_perceptrons_layer = nn.ModuleList()
        for i in range(num_ising_perceptrons):
            # Noise to the initial parameters
            #lambda_i = (-1 ** (i //2) )*lambda_init 
            lambda_i = lambda_init + np.random.uniform(-0.1, 0.1)
            offset_i = offset_init + np.random.uniform(-0.1, 0.1)
            #lambda_i = lambda_init
            #offset_i = offset_init

            module = FullIsingModule(
                sizeAnnealModel=sizeAnnealModel,
                anneal_settings=anneal_settings,
                lambda_init=lambda_i,
                offset_init=offset_i
            )

            with torch.no_grad():
                #random_gamma = torch.randn(sizeAnnealModel, sizeAnnealModel) * 0.01 + np.random.uniform(-0.1, 0.1)
                random_gamma = torch.randn(sizeAnnealModel, sizeAnnealModel) * 0
                random_gamma = torch.triu(random_gamma, diagonal=1)
                module.ising_layer.gamma.copy_(random_gamma)

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
