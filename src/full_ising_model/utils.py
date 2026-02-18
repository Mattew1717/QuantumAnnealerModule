import torch
import numpy as np


class HiddenNodesInitialization:
    """Hidden nodes initialization settings for the model."""

    mode: str = "repeat"
    random_range: tuple[torch.Tensor, torch.Tensor] = (-1, 1)
    function: callable = None
    fun_args: tuple = None

    def __init__(self, mode) -> None:
        self._resize_cache = {}
        self.mode = mode

    def resize(
        self, thetas: torch.Tensor, target_size: int
    ) -> torch.Tensor:
        """
        Resize thetas to target size with caching for efficient repeated operations.

        Args:
            thetas: Input tensor of shape (batch_size, input_dim)
            target_size: Target size for resizing

        Returns:
            Resized tensor of shape (batch_size, target_size)
        """
        input_dim = thetas.shape[1]

        # No resize needed
        if input_dim == target_size:
            return thetas

        cache_key = (input_dim, thetas.device, thetas.dtype)

        # Check cache
        if cache_key not in self._resize_cache:
            # Compute and cache transformation parameters
            if (self.mode == "function" and
                self.function == offset and
                self.fun_args is not None):
                # Cache transformation parameters for efficient repeat+add
                offset_value = self.fun_args[0]
                num_full_repeats = target_size // input_dim
                remainder = target_size % input_dim

                # Pre-compute the offset pattern to add
                offsets = torch.arange(num_full_repeats + 1, device=thetas.device, dtype=thetas.dtype)
                offsets = offsets * offset_value

                self._resize_cache[cache_key] = {
                    'num_full_repeats': num_full_repeats,
                    'remainder': remainder,
                    'offsets': offsets,
                    'offset_value': offset_value
                }
            else:
                # For other modes, cache the full transformation (less common)
                self._resize_cache[cache_key] = None

        # Apply cached transformation
        cached_data = self._resize_cache[cache_key]
        if cached_data is not None:
            # Use efficient repeat + add pattern
            num_full_repeats = cached_data['num_full_repeats']
            offsets = cached_data['offsets']

            # Repeat the input tensor
            if num_full_repeats > 0:
                thetas_repeated = thetas.repeat(1, num_full_repeats + 1)[:, :target_size]
            else:
                thetas_repeated = thetas[:, :target_size]

            # Add offsets: each repeated block gets progressively larger offset
            offset_pattern = offsets.repeat_interleave(input_dim)[:target_size]
            return thetas_repeated + offset_pattern
        else:
            # Fallback to full resize (for non-offset functions)
            return resize_tensor(thetas, target_size, self)


def offset(theta: torch.Tensor, index_new: int, fun_args: tuple) -> float:
    """
    Calculates a new value for the given index of
    the theta tensor by adding an offset.
    """
    offset_value = fun_args[0]
    return theta[index_new % len(theta)] + index_new // len(theta) * offset_value


def resize_tensor(
    thetas: torch.Tensor, target_size: int, hidden_nodes: HiddenNodesInitialization
) -> torch.Tensor:
    """
    Resize a batch of theta tensors to the target size using hidden nodes initialization.

    Args:
        thetas: Input tensor of shape (batch_size, input_dim)
        target_size: Target size for resizing
        hidden_nodes: HiddenNodesInitialization configuration

    Returns:
        Resized tensor of shape (batch_size, target_size)
    """
    batch_size, input_dim = thetas.shape

    if target_size <= input_dim:
        return thetas[:, :target_size]

    if hidden_nodes.mode == "function":
        if hidden_nodes.function is None:
            msg = "function must be given when mode is function"
            raise ValueError(msg)

        # Vectorized implementation for offset function (most common case)
        if hidden_nodes.function == offset and hidden_nodes.fun_args is not None:
            offset_value = hidden_nodes.fun_args[0]
            # Create indices for efficient computation
            indices = torch.arange(target_size, device=thetas.device)
            base_indices = indices % input_dim
            multipliers = indices // input_dim

            # Vectorized: thetas[:, base_indices] + multipliers * offset_value
            thetas_resized = thetas[:, base_indices] + multipliers.float() * offset_value
            return thetas_resized

        # Fallback for generic functions (slower)
        thetas_resized = torch.zeros(
            (batch_size, target_size), dtype=thetas.dtype, device=thetas.device, requires_grad=thetas.requires_grad
        )
        for batch_idx in range(batch_size):
            theta = thetas[batch_idx]
            if hidden_nodes.fun_args is None:
                values = [hidden_nodes.function(theta, idx) for idx in range(target_size)]
            else:
                values = [hidden_nodes.function(theta, idx, hidden_nodes.fun_args) for idx in range(target_size)]
            thetas_resized[batch_idx] = torch.stack(values)
        return thetas_resized

    # Vectorized implementation for standard modes
    thetas_resized = torch.zeros(
        (batch_size, target_size), dtype=thetas.dtype, device=thetas.device
    )

    if hidden_nodes.mode == "repeat":
        # Vectorized repeat: use modulo indexing
        indices = torch.arange(target_size, device=thetas.device) % input_dim
        thetas_resized = thetas[:, indices]
    elif hidden_nodes.mode == "zeros":
        # Copy original values, rest are already zeros
        thetas_resized[:, :input_dim] = thetas
    elif hidden_nodes.mode == "random":
        # Copy original values
        thetas_resized[:, :input_dim] = thetas
        # Fill rest with random values
        num_random = target_size - input_dim
        if num_random > 0:
            random_vals = torch.rand(batch_size, num_random, device=thetas.device)
            random_vals = random_vals * (hidden_nodes.random_range[1] - hidden_nodes.random_range[0]) + hidden_nodes.random_range[0]
            thetas_resized[:, input_dim:] = random_vals

    return thetas_resized


class utils:
    @staticmethod
    def make_upper_triangular_torch(gamma_tensor: torch.Tensor) -> torch.Tensor:
        return torch.triu(gamma_tensor, diagonal=1)

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
