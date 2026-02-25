# QuantumAnnealerModule

Research project in collaboration with an **University of Trento (UniTN)** group. Paper in preparation.

This work starts from the Ising-based machine learning model introduced in [Schmid, Zardini & Pastorello (2023) — arXiv:2310.18411](https://arxiv.org/abs/2310.18411) (referred to here as **SZP**) and pursues three goals:

1. Port the SZP model to **PyTorch** with a differentiable forward/backward pass.
2. Optimize it with **multithreading** for batch processing.
3. Empirically test it — standalone and inside **hybrid classical-quantum networks** — studying performance, limits, and trade-offs.

---

## How it works

An `FullIsingModule` is a standard `nn.Module`. Its forward pass solves an Ising minimization problem; its backward pass propagates gradients through the spin configuration's outer product, updating the coupling matrix `Γ`.

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
| `SZP_Model/` | Original SZP reference implementation (no PyTorch), used for comparison |
| `ModularNetwork/` | `Network_1L` and `Network_2L` hybrid architectures |
| `Inference/` | All test scripts, datasets, utilities |
| `Inference/Datasets/` | 9 UCI binary classification datasets |
| `Inference/.env` | All hyperparameters |

---

## Annealer backends

| `AnnealerType` | Backend | Notes |
|----------------|---------|-------|
| `SIMULATED` | `dwave-neal` | Default — no hardware required |
| `EXACT` | `dimod.ExactSolver` | Brute-force — only for small N |
| `QUANTUM` | D-Wave QPU | Requires a D-Wave Leap token and profile |

---

## Networks

**`Network_1L`** (`MultiIsingNetwork`) — N parallel Ising perceptrons, outputs combined by a linear layer. The input features can be partitioned so that each node in the first layer sees a portion of the input.

```
x ──► IsingModule × N  ──►  Linear(N→1)  ──►  output
```

**`Network_2L`** (`TwoLayerIsingNetwork`) — two Ising layers with a linear mixing stage between them. Feature re-uploading.

```
x ──► [Ising ×N₁] ──► Linear ──► cat([E₁, x]) ──► [Ising ×N₂] ──► Linear ──► output
```

---

## Inference — test scripts

| Script | What it does |
|--------|-------------|
| `test_xor.py` | Trains `FullIsingModule`, `Network_1L`, `Network_2L` on XOR from 1D to 6D; outputs metrics, curves, confusion matrices |
| `test_matrix_xor.py` | Grid search over `num_perceptrons × node_size` on `Network_1L` for each XOR dimension; outputs accuracy/F1/AUC/timing heatmaps |
| `test_datasetsUCI.py` | K-Fold CV of `FullIsingModule` vs `Network_1L` on 9 UCI datasets |
| `test_comparison_SZPvsTorch.py` | Direct comparison of original SZP vs `FullIsingModule` on Iris — validates the PyTorch port |

---

## Datasets

9 UCI binary classification datasets are in `Inference/Datasets/`: Iris, Vertebral Column, Banknote, Breast Cancer, Contraceptive Method, Haberman's Survival, Heart Failure, Ionosphere, SPECTF Heart.

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

All parameters live in `Inference/.env`. 

---

## Running tests

From the repository root:

```bash
python -m Inference.test_xor
python -m Inference.test_matrix_xor
python -m Inference.test_datasetsUCI
python -m Inference.test_comparison_SZPvsTorch
```

Plots, CSVs and logs are saved in timestamped directories in the working directory.

---

## Notes

- Training is slow: the annealer is the bottleneck — each forward call invokes a sampler per sample.
- `ExactAnnealing` is only tractable for small N (≲ 20 spins).
- QPU access requires a D-Wave Leap subscription and introduces network latency.
- When the input dimension is smaller than the annealer size, `θ` is extended by tiling it cyclically and adding a small constant offset at each repetition: `θ[i] = x[i % d] + (i // d) · ε`. This fills the extra nodes with slightly shifted copies of the input, preserving structure while avoiding exact duplicates.
