import torch
import numpy as np


class utils:
    @staticmethod
    def make_upper_triangular_torch(gamma_tensor: torch.Tensor) -> torch.Tensor:
        size = gamma_tensor.shape[0]
        with torch.no_grad():   # Disable gradient tracking
            for i in range(size):
                gamma_tensor[i, i] = 0  # Zero out diagonal
                for j in range(i):      # Zero out lower triangle (j < i)
                    gamma_tensor[i, j] = 0
        return gamma_tensor
    
    @staticmethod
    def vector_to_biases(theta: np.array) -> dict:
        """
        Convert the theta vector to biases of an Ising model.

        param theta: the theta vector
        type: np.array

        return: the bias values
        rtype: dict
        """
        return {k: v for k, v in enumerate(theta.tolist())}

    @staticmethod
    def gamma_to_couplings(gamma: np.array) -> dict:
        """
        Convert the gamma matrix to couplings of an Ising model.

        param gamma: the gamma matrix
        type: np.array

        return: the coupling values
        rtype: dict
        """
        J = {
            (qubit_i, qubit_j): weight
            for (qubit_i, qubit_j), weight in np.ndenumerate(gamma)
            if qubit_i < qubit_j
        }
        return J
