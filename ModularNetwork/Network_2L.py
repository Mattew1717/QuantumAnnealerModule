import torch
import torch.nn as nn
import numpy as np
from IsingModule.utils import AnnealingSettings
from IsingModule.FullIsingModule import FullIsingModule


class TwoLayerIsingNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_ising_1: int,
        num_ising_2: int,
        anneal_settings: AnnealingSettings,
        lambda_init: float = 0.1,
        offset_init: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        # ---- First Ising ----
        self.ising1 = nn.ModuleList([
            FullIsingModule(
                sizeAnnealModel=input_dim,
                anneal_settings=anneal_settings,
                lambda_init=lambda_init + np.random.uniform(-0.1, 0.1),
                offset_init=offset_init + np.random.uniform(-0.1, 0.1),
            )
            for _ in range(num_ising_1)
        ])

        # Linear layer
        self.lin1 = nn.Linear(num_ising_1, num_ising_1, bias=bias)

        # ---- Second Ising ----
        # input = [mixed Ising outputs | original x]
        second_input_dim = num_ising_1 + input_dim

        self.ising2 = nn.ModuleList([
            FullIsingModule(
                sizeAnnealModel=second_input_dim,
                anneal_settings=anneal_settings,
                lambda_init=lambda_init + np.random.uniform(-0.1, 0.1),
                offset_init=offset_init + np.random.uniform(-0.1, 0.1),
            )
            for _ in range(num_ising_2)
        ])

        self.lin2 = nn.Linear(num_ising_2, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- pass through first Ising ----
        E1 = torch.stack([ising(x) for ising in self.ising1], dim=1)
        
        # ---- Linear Combination ----
        E1_lin = self.lin1(E1)

        # ---- Combine Energies + Input. [E1, E2, ... En | x1, x2, xn] ----
        z = torch.cat([E1_lin, x], dim=1)

        # ---- pass through second Ising ----
        E2 = torch.stack([ising(z) for ising in self.ising2], dim=1)

        # ---- Final linear combination ----
        E2_lin = self.lin2(E2).squeeze(-1)

        return E2_lin
