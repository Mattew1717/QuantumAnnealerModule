import torch
import numpy as np


def offset(theta: torch.Tensor, index_new: int, offset_value: float) -> float:
    """Repeat-with-additive-offset rule for hidden-node padding."""
    n = len(theta)
    return theta[index_new % n] + (index_new // n) * offset_value


class HiddenNodesInitialization:
    """
    Hidden-node padding via the `offset` rule:
        x_new[k] = x[k mod n] + (k // n) * offset_value
    Vectorized resize with per-(input_dim, device, dtype) cache.
    """

    def __init__(self, offset_value: float) -> None:
        self.offset_value = offset_value
        self._resize_cache: dict = {}

    def resize(self, thetas: torch.Tensor, target_size: int) -> torch.Tensor:
        input_dim = thetas.shape[1]

        if input_dim == target_size:
            return thetas
        if input_dim > target_size:
            raise ValueError(
                f"Input dimension ({input_dim}) exceeds target_size ({target_size}). "
                "size_annealer must be >= input dimension."
            )
        if input_dim == 0:
            raise ValueError("Input tensor has zero feature dimension; cannot resize.")

        cache_key = (input_dim, thetas.device, thetas.dtype)
        cached = self._resize_cache.get(cache_key)
        if cached is None:
            num_full_repeats = target_size // input_dim
            offsets = torch.arange(
                num_full_repeats + 1, device=thetas.device, dtype=thetas.dtype
            ) * self.offset_value
            offset_pattern = offsets.repeat_interleave(input_dim)[:target_size]
            cached = {
                'num_full_repeats': num_full_repeats,
                'offset_pattern': offset_pattern,
            }
            self._resize_cache[cache_key] = cached

        num_full_repeats = cached['num_full_repeats']
        offset_pattern = cached['offset_pattern']

        if num_full_repeats > 0:
            thetas_repeated = thetas.repeat(1, num_full_repeats + 1)[:, :target_size]
        else:
            thetas_repeated = thetas[:, :target_size]

        return thetas_repeated + offset_pattern


class utils:
    @staticmethod
    def make_upper_triangular_torch(gamma_tensor: torch.Tensor) -> torch.Tensor:
        return torch.triu(gamma_tensor, diagonal=1)

    @staticmethod
    def vector_to_biases(theta: np.ndarray) -> dict:
        return {k: v for k, v in enumerate(theta.tolist())}

    @staticmethod
    def gamma_to_couplings(gamma: np.ndarray) -> dict:
        return {
            (i, j): w
            for (i, j), w in np.ndenumerate(gamma)
            if i < j
        }
