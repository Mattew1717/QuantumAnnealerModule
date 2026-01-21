import torch
import torch.nn as nn
import numpy as np
from IsingModule.utils import AnnealingSettings
from IsingModule.FullIsingModule import FullIsingModule


class TwoStageIsingNetwork(nn.Module):

    def __init__(
        self,
        sizeModule: int,         # n + hidden (deciso dal main)
        num_ising_1: int,        # k
        num_ising_2: int,        # k2
        anneal_settings: AnnealingSettings,
        lambda_init: float = 0.0,
        offset_init: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        # -------- First Ising layer (dimension sizeModule) --------
        self.ising_layer1 = nn.ModuleList([
            FullIsingModule(
                sizeAnnealModel=sizeModule,
                anneal_settings=anneal_settings,
                lambda_init=lambda_init + np.random.uniform(-0.1, 0.1),
                offset_init=offset_init + np.random.uniform(-0.1, 0.1),
            )
            for _ in range(num_ising_1)
        ])

        # Normalization after first Ising stack
        self.norm1 = nn.LayerNorm(num_ising_1)

        # -------- Classical projection to 20 --------
        self.to_20 = nn.Linear(num_ising_1, 10, bias=bias)
        self.activation = nn.Tanh()

        # -------- Second Ising layer (fixed size = 20) --------
        self.ising_layer2 = nn.ModuleList([
            FullIsingModule(
                sizeAnnealModel=10,
                anneal_settings=anneal_settings,
                lambda_init=lambda_init + np.random.uniform(-0.1, 0.1),
                offset_init=offset_init + np.random.uniform(-0.1, 0.1),
            )
            for _ in range(num_ising_2)
        ])

        # Normalization after second Ising stack
        self.norm2 = nn.LayerNorm(num_ising_2)

        # -------- Final output --------
        self.output_layer = nn.Linear(num_ising_2, 1, bias=bias)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        thetas: (batch, sizeModule)

        Optimized forward with pre-allocated tensors and in-place operations.
        """
        batch_size = thetas.size(0)

        # ----- First Ising layer -----
        # Pre-allocate output tensor
        E1 = torch.empty(batch_size, len(self.ising_layer1),
                        dtype=thetas.dtype, device=thetas.device)

        for i, ising in enumerate(self.ising_layer1):
            E1[:, i] = ising(thetas)

        E1 = self.norm1(E1)

        # ----- Classical projection to 20 -----
        h20 = self.to_20(E1)
        h20 = self.activation(h20)

        # ----- Residual connection -----
        h20_res = h20

        # ----- Second Ising layer -----
        # Pre-allocate output tensor
        E2 = torch.empty(batch_size, len(self.ising_layer2),
                        dtype=h20_res.dtype, device=h20_res.device)

        for i, ising in enumerate(self.ising_layer2):
            E2[:, i] = ising(h20_res)

        E2 = self.norm2(E2)

        # ----- Output -----
        out = self.output_layer(E2)
        return out.squeeze(-1)
