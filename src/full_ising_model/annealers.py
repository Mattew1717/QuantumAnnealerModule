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
    """Configuration container for annealing-based samplers."""

    def __init__(
        self,
        beta_range: list,
        num_reads: int,
        num_sweeps: int,
        num_sweeps_per_beta: int,
        beta_schedule_type: str = "geometric",
    ) -> None:
        self.beta_range = beta_range
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.num_sweeps_per_beta = num_sweeps_per_beta
        self.beta_schedule_type = beta_schedule_type


class Annealer(ABC):
    """Abstract base class for Ising model samplers."""

    def __init__(self, size: int):
        self.size = size

    @abstractmethod
    def sample(self, h: Dict[int, float], j: Dict[tuple, float]) -> SampleSet:
        ...


class SimulatedAnnealing(Annealer):
    """
    Classical simulated annealing using neal, with a thread-safe pool of
    independent samplers (one per worker).
    """

    def __init__(self, size: int, settings: AnnealingSettings, num_workers: int):
        super().__init__(size)
        self._settings = settings
        self._pool: _queue.Queue = _queue.Queue()
        for _ in range(num_workers):
            self._pool.put(SimulatedAnnealingSampler())

    def sample(self, h: dict, j: dict) -> SampleSet:
        sampler = self._pool.get()
        try:
            return sampler.sample_ising(
                h,
                j,
                beta_range=self._settings.beta_range,
                num_reads=self._settings.num_reads,
                num_sweeps=self._settings.num_sweeps,
                num_sweeps_per_beta=self._settings.num_sweeps_per_beta,
                beta_schedule_type=self._settings.beta_schedule_type,
            )
        finally:
            self._pool.put(sampler)


class ExactAnnealing(Annealer):
    """
    Exact Ising solver based on dimod ExactSolver, with a thread-safe pool
    of independent samplers (one per worker).
    """

    def __init__(self, size: int, num_workers: int):
        super().__init__(size)
        self._pool: _queue.Queue = _queue.Queue()
        for _ in range(num_workers):
            self._pool.put(ExactSolver())

    def sample(self, h: dict, j: dict) -> SampleSet:
        sampler = self._pool.get()
        try:
            return sampler.sample_ising(h, j)
        finally:
            self._pool.put(sampler)


class QuantumAnnealing(Annealer):
    """
    Quantum annealing on D-Wave QPU with fixed embedding and a thread-safe
    pool of FixedEmbeddingComposite instances (one per worker).
    """

    def __init__(
        self,
        size: int,
        profile: str,
        num_reads: int,
        num_workers: int,
    ):
        super().__init__(size)

        if DWaveSampler is None or EmbeddingComposite is None or FixedEmbeddingComposite is None:
            raise ImportError(
                "dwave.system (D-Wave SDK) is required for QuantumAnnealing but is not installed."
            )

        print("Searching QPU and computing embedding...")
        probe = EmbeddingComposite(DWaveSampler(profile=profile))
        embedding = probe.find_embedding(
            nx.complete_graph(size).edges(),
            probe.child.edgelist,
        )

        self._pool: _queue.Queue = _queue.Queue()
        for _ in range(num_workers):
            self._pool.put(
                FixedEmbeddingComposite(DWaveSampler(profile=profile), embedding)
            )
        self._num_reads = num_reads

        first = self._pool.get()
        print(f"Using QPU: {first.child.solver.id}  (pool size: {num_workers})")
        self._pool.put(first)

    def sample(self, h: dict, j: dict) -> SampleSet:
        sampler = self._pool.get()
        try:
            ss = sampler.sample_ising(h, j, num_reads=self._num_reads)
            ss.resolve()
            return ss
        finally:
            self._pool.put(sampler)


class AnnealerType(str, Enum):
    SIMULATED = "simulated"
    QUANTUM = "quantum"
    EXACT = "exact"


class AnnealerFactory:
    """Factory for creating Annealer instances. All parameters explicit, no defaults."""

    @staticmethod
    def create(
        annealer_type: AnnealerType,
        size: int,
        num_workers: int,
        settings: AnnealingSettings | None = None,
        profile: str | None = None,
        num_reads: int | None = None,
    ) -> Annealer:
        if annealer_type == AnnealerType.SIMULATED:
            if settings is None:
                raise ValueError("SIMULATED annealer requires `settings`.")
            return SimulatedAnnealing(size=size, settings=settings, num_workers=num_workers)

        if annealer_type == AnnealerType.EXACT:
            return ExactAnnealing(size=size, num_workers=num_workers)

        if annealer_type == AnnealerType.QUANTUM:
            if profile is None or num_reads is None:
                raise ValueError("QUANTUM annealer requires `profile` and `num_reads`.")
            return QuantumAnnealing(
                size=size,
                profile=profile,
                num_reads=num_reads,
                num_workers=num_workers,
            )

        raise ValueError(f"Unsupported annealer type: {annealer_type}")
