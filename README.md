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
Gradients flow **only through `Γ`**; the input biases `θ` are treated as fixed.

---

## Repository layout

| Path | Contents |
|------|----------|
| `src/full_ising_model/` | Installable PyTorch package: `FullIsingModule`, annealers, utils |
| `NeuralNetwork/` | `ModularNetwork` — N parallel Ising perceptrons + linear combiner |
| `SZP_Model/` | Original SZP reference implementation (no PyTorch), used for comparison |
| `Inference/` | Test scripts, datasets, plotting, logging |
| `Inference/Datasets/` | 9 UCI binary classification datasets (no header row) |
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
| `test_xor.py` | `FullIsingModule` vs `ModularNetwork` on XOR, dimensions 1D-6D |
| `test_datasetsUCI.py` | K-Fold CV of `FullIsingModule` vs `ModularNetwork` on 9 UCI datasets |
| `test_comparison_SZPvsTorch.py` | Direct comparison of original SZP vs `FullIsingModule` on Iris — validates the PyTorch port |

All scripts seed `numpy` and `torch` from `RANDOM_SEED` for reproducibility, and write timestamped output directories (SVG + PDF plots, CSVs, `run_<ts>.log`) into the cwd.

---

## Datasets

9 UCI binary classification CSVs in `Inference/Datasets/`: Iris (versicolor vs virginica), Vertebral Column, Banknote, Breast Cancer, Contraceptive Method, Haberman's Survival, Heart Failure, Ionosphere, SPECTF Heart. The CSVs have no header row; the last column is the label (binary, with `-1` automatically remapped to `0`).

---

## Installation

```bash
git clone https://github.com/Mattew1717/QuantumAnnealerModule.git
cd QuantumAnnealerModule
pip install -r requirements.txt
pip install -e .          # installs the full_ising_model package
```

---

## Configuration

All hyperparameters live in `Inference/.env` and are read strictly (a missing key raises `KeyError` at startup). Highlights:

- `ANNEALER_TYPE`: `simulated` | `exact` | `quantum`
- `NUM_THREADS`: workers in the sampler pool
- `MODEL_SIZE`: `-1` for auto (`max(n_features, MINIMUM_MODEL_SIZE)`)
- `HIDDEN_NODES_OFFSET_VALUE`: ε for the offset padding rule
- `LAMBDA_INIT`, `OFFSET_INIT`: initial values for `λ` and `b`
- Per-parameter learning rates: `LEARNING_RATE_GAMMA`, `LEARNING_RATE_LAMBDA`, `LEARNING_RATE_OFFSET`, `LEARNING_RATE_COMBINER`
- `PRINT_INTERVAL`: epoch interval for loss logging during training (no test-set evaluation runs during training)

---

## Running tests

From the repository root:

```bash
python -m Inference.test_xor
python -m Inference.test_datasetsUCI
python -m Inference.test_comparison_SZPvsTorch
```

---

## Notes

- Training is bottlenecked by the annealer: each forward call invokes one sample per element of the batch.
- `ExactAnnealing` is only tractable for small N (≲ 20 spins).
- QPU access requires a D-Wave Leap subscription and introduces network latency.
- When the input dimension is smaller than the annealer size, `θ` is extended by tiling it cyclically and adding a small constant offset at each repetition: `θ_new[k] = θ[k mod n] + (k // n) · ε`.
