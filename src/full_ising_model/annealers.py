from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict
import queue as _queue


from dimod import SampleSet, ExactSolver
from neal import SimulatedAnnealingSampler
try:
    from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
except Exception:
    DWaveSampler = None
    EmbeddingComposite = None
    FixedEmbeddingComposite = None

import networkx as nx


class AnnealingSettings:
    """
    Configuration container for annealing-based samplers.
    """

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

class Annealer(ABC):
    """
    Abstract base class for Ising model samplers.
    """

    def __init__(self, size: int):
        self.size = size

    @abstractmethod
    def sample(self, h: Dict[int, float], j: Dict[tuple, float]) -> SampleSet:
        """
        Sample the Ising model defined by local fields h and couplings j.
        """
        pass


class SimulatedAnnealing(Annealer):
    """
    Classical simulated annealing using neal.
    """

    def __init__(self, size: int, settings: AnnealingSettings):
        super().__init__(size)
        self._sampler = SimulatedAnnealingSampler()
        self._settings = settings

    def sample(self, h: dict, j: dict) -> SampleSet:
        return self._sampler.sample_ising(
            h,
            j,
            beta_range=self._settings.beta_range,
            num_reads=self._settings.num_reads,
            num_sweeps=self._settings.num_sweeps,
            num_sweeps_per_beta=self._settings.num_sweeps_per_beta,
            beta_schedule_type=self._settings.beta_schedule_type,
        )


class ExactAnnealing(Annealer):
    """
    Exact Ising solver based on dimod ExactSolver.
    """

    def __init__(self, size: int):
        super().__init__(size)
        self._sampler = ExactSolver()

    def sample(self, h: dict, j: dict) -> SampleSet:
        return self._sampler.sample_ising(h, j)


class QuantumAnnealing(Annealer):
    """
    Quantum annealing on D-Wave QPU with fixed embedding.

    Computes the embedding once at init, then creates `num_workers` independent
    FixedEmbeddingComposite instances (each with its own DWaveSampler connection)
    stored in a thread-safe Queue.  Concurrent threads each check out their own
    sampler, submit a job, resolve it, and return the sampler to the pool.
    """

    def __init__(
        self,
        size: int,
        profile: str = "default",
        num_reads: int = 1,
        num_workers: int = 1,
    ):
        super().__init__(size)

        # Check availability of dwave system package
        if DWaveSampler is None or EmbeddingComposite is None or FixedEmbeddingComposite is None:
            raise ImportError("dwave.system (D-Wave SDK) is required for QuantumAnnealing but is not installed.")

        # Find embedding once and reuse it across all samplers in the pool
        print("Searching QPU and computing embedding...")
        probe = EmbeddingComposite(DWaveSampler(profile=profile))
        embedding = probe.find_embedding(
            nx.complete_graph(size).edges(),
            probe.child.edgelist,
        )

        # Build a pool of num_workers independent samplers, all sharing the same embedding
        self._pool: _queue.Queue = _queue.Queue()
        for _ in range(num_workers):
            self._pool.put(
                FixedEmbeddingComposite(DWaveSampler(profile=profile), embedding)
            )
        self._num_reads = num_reads

        # Print QPU id (borrow and return the first sampler)
        first = self._pool.get()
        print(f"Using QPU: {first.child.solver.id}  (pool size: {num_workers})")
        self._pool.put(first)

    def sample(self, h: dict, j: dict) -> SampleSet:
        sampler = self._pool.get()          # blocks only if all workers are busy
        try:
            ss = sampler.sample_ising(h, j, num_reads=self._num_reads)
            ss.resolve()                    # wait for QPU result before returning
            return ss
        finally:
            self._pool.put(sampler)         # always return to pool

class AnnealerType(str, Enum):
    """
    Supported annealer backends.
    """
    SIMULATED = "simulated"
    QUANTUM = "quantum"
    EXACT = "exact"


class AnnealerFactory:
    """
    Factory for creating Annealer instances.
    """

    @staticmethod
    def create(
        annealer_type: AnnealerType,
        size: int,
        **kwargs,
    ) -> Annealer:

        if annealer_type == AnnealerType.SIMULATED:
            return SimulatedAnnealing(
                size=size,
                settings=kwargs["settings"],
            )

        if annealer_type == AnnealerType.QUANTUM:
            return QuantumAnnealing(
                size=size,
                profile=kwargs.get("profile", "default"),
                num_reads=kwargs.get("num_reads", 1),
                num_workers=kwargs.get("num_workers", 1),
            )

        if annealer_type == AnnealerType.EXACT:
            return ExactAnnealing(size=size)

        raise ValueError(f"Unsupported annealer type: {annealer_type}")
