import torch

from dimod import ExactSolver,SampleSet
from neal import SimulatedAnnealingSampler
from abc import ABC
import numpy as np

from dwave.system import (
    DWaveSampler,
    EmbeddingComposite,
    FixedEmbeddingComposite,
)
from networkx import complete_graph

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

class GammaInitialization:
    """Gamma initialization settings for the model."""

    mode: str
    value_range: None | tuple[float, float]
    _gamma_init: torch.Tensor

    def __init__(self, mode: str) -> None:
        if mode in ["zeros", "random", "fixed"]:
            self.mode = mode
            self.value_range = None
            self._gamma_init = None
        else:
            msg = "invalid gamma initialization mode, choose zeros or random"
            raise ValueError(msg)

    def initialize(
        self, size: int, value_range: tuple[float, float] = None
    ) -> torch.Tensor:
        """Initialize the gamma values."""
        gamma = torch.zeros((size, size))
        if self.mode == "zeros":
            return gamma
        elif self.mode == "random":
            if value_range is None and self.value_range is None:
                msg = (
                    "value range for random gamma initialization not specified"
                )
                raise ValueError(msg)
            if self.value_range is None:
                self.value_range = value_range
            for i in range(size):
                for j in range(i + 1, size):
                    gamma[i, j] = torch.random.uniform(
                        self.value_range[0], self.value_range[1]
                    )
            return gamma
        elif self.mode == "fixed":
            if self._gamma_init is None:
                msg = "fixed gamma initialization not specified"
                raise ValueError(msg)
            if self._gamma_init.shape != gamma.shape:
                msg = "matrix of fixed gamma initialization has wrong shape"
                raise ValueError(msg)
            return self._gamma_init
    
class AnnealingSettings:
    """Settings for the simulated annealing sampler."""
    beta_range: list
    num_reads: int
    num_sweeps: int
    num_sweeps_per_beta: int
    beta_schedule_type: str

    def __init__(self) -> None:
        self.beta_range = None
        self.num_reads = 1
        self.num_sweeps = 100
        self.num_sweeps_per_beta = 1
        self.beta_schedule_type = "geometric"

class ModelSetting:
    """Model settings for the model."""

    gamma_init: GammaInitialization
    mini_batch_size: int
    num_reads: int
    optim_steps: int
    learning_rate_gamma: float
    learning_rate_lmd: float
    learning_rate_offset : float

    def __init__(self) -> None:
        self.gamma_init = GammaInitialization("zeros")
        self.mini_batch_size = 1
        self.num_reads = 1
        self.optim_steps = 1
        self.learning_rate_gamma = 0.1
        self.learning_rate_lmd = 0.1
        self.learning_rate_offset = 0.1

class Model(ABC):
    """
    Abstract class for the model. The derived classes use differnt Ising machines to sample the Ising model.
    In particular, the derived classes implement the _sample_single method, which samples the Ising model
    using the respective Ising machine.
    Examples for derived classes are ExactModel, SimulatedAnnealingModel, DwaveSamplerModel, which use
    the dimod ExactSolver, dimod SimulatedAnnealingSampler and the D-Wave sampler, respectively.
    """
    size: int
    settings: ModelSetting
    _gamma: torch.Tensor
    _sampler: any
    _embedding: any

    def __init__(self, size: int):
        self.size = size
        self.settings = ModelSetting()
        self._gamma = torch.zeros((size, size))

class SimAnnealModel(Model):
    """Solves the provided Ising problem using the simulated annealing sampler from DWAVE neal."""
    def __init__(
        self, size: int, settings: AnnealingSettings = AnnealingSettings()
    ):
        super().__init__(size)
        self._sampler = SimulatedAnnealingSampler()
        self._embedding = None
        self.annealing_settings = settings
    
    
    def _sample_single(self, h: dict, j: dict) -> SampleSet:
        return self._sampler.sample_ising(
            h,
            j,
            beta_range=self.annealing_settings.beta_range,
            num_reads=self.annealing_settings.num_reads,
            num_sweeps=self.annealing_settings.num_sweeps,
            num_sweeps_per_beta=self.annealing_settings.num_sweeps_per_beta,
            beta_schedule_type=self.annealing_settings.beta_schedule_type,
        )

class ExactModel(Model):
    """Solves the provided Ising problem exactly using the dimod ExactSolver."""
    def __init__(self, size: int):
        super().__init__(size)
        self._sampler = ExactSolver()
        self._embedding = None

    def _sample_single(self, h: dict, j: dict) -> SampleSet:
        return self._sampler.sample_ising(h, j)
    
class QPUModel(Model):
    _num_reads: int
    _embedding: dict
    _sampler: FixedEmbeddingComposite
    _profile: str

    def __init__(
        self, size: int, profile: str = "default", num_reads: int = 1
    ):
        super().__init__(size)
        # Find embedding once and use it for all samples
        print("Searching QPU and computing embedding...")
        toy_sampler = EmbeddingComposite(DWaveSampler(profile=profile))
        self._embedding = toy_sampler.find_embedding(
            complete_graph(size).edges(), toy_sampler.child.edgelist
        )
        print("Embedding found.")
        self._sampler = FixedEmbeddingComposite(
            DWaveSampler(profile=profile), self._embedding
        )
        print(f"using QPU: {self._sampler.child.solver.id}")
        self._num_reads = num_reads
        self._profile = profile

    def _sample_single(self, h: dict, j: dict) -> SampleSet:
        return self._sampler.sample_ising(h, j, num_reads=self._num_reads)