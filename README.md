# QuantumAnnealerModule

Research project in collaboration with an **University of Trento (UniTN)** group. Paper in preparation.

This work starts from the Ising-based machine learning model introduced in [Schmid, Zardini & Pastorello (2023) — arXiv:2310.18411](https://arxiv.org/abs/2310.18411) (referred to here as **SZP**) and pursues three goals:

1. Port the SZP model to **PyTorch** with a differentiable forward/backward pass.
2. Optimize it with **multithreading** for batch processing.
3. Empirically test it, standalone and inside **hybrid classical-quantum networks**; studying performance, limits, and trade-offs.

---

## How it works

A `FullIsingModule` is a standard `nn.Module`. Its forward pass solves an Ising minimization problem; its backward pass propagates gradients through the spin configuration's outer product, updating the coupling matrix `Γ`.

```
x  ──►  h (local fields)   +   J = f(Γ)  (couplings, learnable)
                │
           [ Annealer ]   ← simulated | exact | quantum
                │
           E₀ = min Ising energy
                │
        output = λ·E₀ + b       (λ, b also learnable)
```

The three learnable parameters are `Γ` (coupling matrix), `λ` (scale), and `b` (offset).
Gradients flow through `Γ` **and** the input biases `θ` (envelope theorem: `∂E₀/∂Γᵢⱼ = s*ᵢ s*ⱼ`, `∂E₀/∂θᵢ = s*ᵢ`, with `s*` the minimizing spin configuration), so the module can also be composed with upstream layers.

---

## Repository layout

| Path | Contents |
|------|----------|
| `src/full_ising_model/` | Installable PyTorch package: `FullIsingModule`, annealers, utils |
| `NeuralNetwork/` | `ModularNetwork` — N parallel Ising perceptrons + linear combiner |
| `SZP_Model/` | Original SZP reference implementation (no PyTorch), used for comparison |
| `Inference/` | Test scripts, datasets, plotting, logging |
| `Inference/Datasets/` | Two variants (`Datasets_balanced/`, `Datasets_unbalanced/`), each with 9 UCI binary classification CSVs (no header row) |
| `Inference/.env` | All hyperparameters (read strictly, no fallback defaults) |

---

## Annealer backends

| `AnnealerType` | Backend | Notes |
|----------------|---------|-------|
| `SIMULATED` | `dwave-neal` | Default — no hardware required |
| `EXACT` | `dimod.ExactSolver` | Brute-force — only for small N (≲ 20 spins) |
| `QUANTUM` | D-Wave QPU | Requires a D-Wave Leap token and profile |

Each backend keeps a thread-safe pool of `num_workers` independent samplers, so that batched forward calls can be parallelized without sharing sampler state.

---

## Networks

**`ModularNetwork`** — N parallel `FullIsingModule` perceptrons combined by a final `Linear(N→1)` layer. Each perceptron has its own learnable `Γ`, `λ`, `b`. Optionally the input features can be partitioned across perceptrons (`partition_input=True`).

```
x ──► FullIsingModule × N  ──►  Linear(N→1)  ──►  output
```

---

## Inference — test scripts

| Script | What it does |
|--------|-------------|
| `test_xor.py` | `FullIsingModule` vs `ModularNetwork` on 2D XOR (default entry point); `compare_xor_models_all_dimensions()` runs the full 1D-6D sweep |
| `test_datasetsUCI.py` | K-Fold CV of `FullIsingModule` vs `ModularNetwork` on 9 UCI datasets |
| `test_comparison_SZPvsTorch.py` | Direct comparison of original SZP vs `FullIsingModule` on Iris — validates the PyTorch port |
| `test_sim_vs_qpu.py` | 2D XOR with the backend chosen by `ANNEALER_TYPE`: multi-run statistics on simulated annealing, or a single run on the QPU |

All scripts seed `numpy` and `torch` from `RANDOM_SEED` for reproducibility. `test_xor.py`, `test_datasetsUCI.py` and `test_sim_vs_qpu.py` write timestamped output directories (SVG + PDF plots, CSVs, `run_<ts>.log`) into the current working directory; `test_comparison_SZPvsTorch.py` saves its plots to `Inference/plots/` and logs only to stdout.

Only `test_sim_vs_qpu.py` reads `ANNEALER_TYPE`; the other scripts always use simulated annealing.

---

## Datasets

9 UCI binary classification CSVs, available in two variants under `Inference/Datasets/`:

- `Datasets_balanced/` — class-balanced versions
- `Datasets_unbalanced/` — original class distribution

Both variants contain the same 9 datasets: Iris (versicolor vs virginica), Vertebral Column, Banknote, Breast Cancer, Contraceptive Method, Haberman's Survival, Heart Failure, Ionosphere, SPECTF Heart. The CSVs have no header row; the last column is the label (binary, with `-1` automatically remapped to `0`). The active variant is selected via `DATASETS_DIR` in `Inference/.env`.

---

## Installation

Requires **Python ≥ 3.10**.

```bash
git clone https://github.com/Mattew1717/QuantumAnnealerModule.git
cd QuantumAnnealerModule
```

**Package only** (installs `full_ising_model`, distribution name `FullIsingModel`):

```bash
pip install .            # simulated / exact annealers
pip install ".[qpu]"     # + dwave-system, needed for AnnealerType.QUANTUM
```

**Experiments** (`Inference/`, `NeuralNetwork/`, `SZP_Model/`): these also need scikit-learn, pandas, matplotlib, etc.

```bash
pip install -r requirements.txt   # includes an editable install of the package (-e .)
```

---

## Usage

```python
import torch
from full_ising_model import FullIsingModule, AnnealerType, AnnealingSettings

settings = AnnealingSettings(
    beta_range=[1, 10], num_reads=1, num_sweeps=1000, num_sweeps_per_beta=1,
)

model = FullIsingModule(
    size_annealer=8,                    # number of spins (must be ≥ input features)
    annealer_type=AnnealerType.SIMULATED,
    annealing_settings=settings,        # required for SIMULATED, ignored otherwise
    lambda_init=1.0,
    offset_init=0.0,
    num_workers=4,                      # sampler threads per forward pass
    hidden_nodes_offset_value=-0.02,    # ε for padding inputs up to size_annealer
)

x = torch.randn(16, 2)                  # (batch, features)
y = model(x)                            # shape (16,): λ·E₀ + b
y.sum().backward()                      # gradients on model.gamma, model.lmd, model.offset
```

For the QPU backend, pass `annealer_type=AnnealerType.QUANTUM`, `profile="<dwave profile>"` and `num_reads=<n>` (requires the `qpu` extra and a configured D-Wave Leap account).

---

## Configuration

All hyperparameters live in `Inference/.env` and are read strictly (a missing key raises `KeyError` at startup).

**General**
- `NUM_THREADS`: workers in the sampler pool
- `DATASETS_DIR`: which dataset variant to use (e.g. `Datasets/Datasets_unbalanced`)
- `RANDOM_SEED`: global seed for `numpy` and `torch`
- `LAMBDA_INIT`, `OFFSET_INIT`: initial values for `λ` and `b`
- `HIDDEN_NODES_OFFSET_VALUE`: ε for the offset padding rule

**Annealer**
- `ANNEALER_TYPE`: `simulated` | `exact` | `quantum` (used only by `test_sim_vs_qpu.py`)
- `DWAVE_PROFILE`: D-Wave profile name from `~/.config/dwave/dwave.conf` (used only when `ANNEALER_TYPE=quantum`)
- `NUM_READS`: samples per annealer call
- `SA_NUM_SWEEPS`, `SA_SWEEPS_PER_BETA`, `SA_BETA_MIN`, `SA_BETA_MAX`: simulated annealing schedule
- `MODEL_SIZE`: annealer size (`-1` for auto, i.e. `max(n_features, MINIMUM_MODEL_SIZE)`)
- `MINIMUM_MODEL_SIZE`: floor used by the auto-sizing rule above

**Training**
- `EPOCHS`, `BATCH_SIZE`
- Per-parameter learning rates: `LEARNING_RATE_GAMMA`, `LEARNING_RATE_LAMBDA`, `LEARNING_RATE_OFFSET`, `LEARNING_RATE_COMBINER`
- `PRINT_INTERVAL`: epoch interval for loss logging during training (no test-set evaluation runs during training)
- `K_FOLDS`: folds for K-Fold CV (`test_datasetsUCI.py`)
- `N_SAMPLES_PER_REGION`, `TEST_SIZE`: XOR experiment parameters (`test_xor.py`)

**ModularNetwork**
- `NUM_ISING_PERCEPTRONS`: number of parallel `FullIsingModule` perceptrons
- `PARTITION_INPUT`: whether to slice the input features across perceptrons

---

## Running tests

From the repository root:

```bash
python -m Inference.test_xor
python -m Inference.test_datasetsUCI
python -m Inference.test_comparison_SZPvsTorch
python -m Inference.test_sim_vs_qpu
```

---

## Notes

- Training is bottlenecked by the annealer: each forward call invokes one sample per element of the batch.
- `ExactAnnealing` is only tractable for small N (≲ 20 spins).
- QPU access requires a D-Wave Leap subscription and introduces network latency.
- When the input dimension is smaller than the annealer size, `θ` is extended by tiling it cyclically and adding a small constant offset at each repetition: `θ_new[k] = θ[k mod n] + (k // n) · ε`.
