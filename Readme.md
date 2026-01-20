# Two-Stage Ising Neural Network

This repository implements a **Two-Stage Ising Neural Network**, a hybrid architecture that combines
**energy-based Ising models** with **classical neural layers**.

Unlike standard MLPs, this network is designed to learn and compose **energy landscapes** rather than
smooth parametric functions.

---

## Overview

The network is composed of:

1. A **first layer of parallel Ising networks** (`FullIsingModule`)
2. A **classical linear projection** to a fixed latent Ising dimension
3. A **second layer of Ising networks**
4. A **final linear readout**

Each `FullIsingModule` represents a **learnable Ising Hamiltonian** whose minimum energy is used as a feature.

---

## Architecture

### High-level structure

Input θ ∈ ℝ^(n + hidden)
        │
        ▼
First Ising Layer (k modules)
 ┌────────────────────┐
 │ FullIsing(z) → E₁  │
 │ FullIsing(z) → E₂  │
 │ …                  │
 │ FullIsing(z) → E_k │
 └────────────────────┘
        │
        ▼
Stack energies → ℝᵏ
        │
        ▼
    LayerNorm
        │
        ▼
  Linear (k → 20)
        │
        ▼
      Tanh
        │
        ▼
Second Ising Layer (k₂ modules)
 ┌──────────────────────┐
 │ FullIsing(20) → E₁'  │
 │ FullIsing(20) → E₂'  │
 │ …                    │
 │ FullIsing(20) → E_k₂ │
 └──────────────────────┘
        │
        ▼
Stack energies → ℝ^(k₂)
        │
        ▼
    LayerNorm
        │
        ▼
  Linear (k₂ → 1)
        │
        ▼
     Output


---

## Core Components

### FullIsingModule

Each `FullIsingModule`:
- Interprets its input vector as **biases of an Ising model**
- Uses annealing (simulated, exact, or QPU) to find a low-energy configuration
- Outputs the **minimum energy**:
  
E(θ) = λ · E₀(θ) + offset


This makes each module a **non-smooth, combinatorial feature extractor**.

---

## Experimental Design Choices (Academic Rationale)

### 1. Parallel Ising Modules (First Layer)

**Why**:
- Each Ising module learns a different Hamiltonian
- Encourages diverse energy landscapes
- Acts as a bank of *energy filters*

**Interpretation**:
> Analogous to feature detectors, but operating in energy space rather than activation space.

---

### 2. Layer Normalization After Ising Layers

**Why**:
- Ising energies can vary wildly in scale
- Normalization stabilizes gradients
- Prevents dominance of a single Ising module

**Academic motivation**:
Energy-based models often suffer from scale instability; normalization enforces comparability
between independent Hamiltonians.

---

### 3. Projection to a Fixed Latent Ising Dimension (20)

**Why**:
- Second Ising layer requires fixed-size input
- Dimension 20 is large enough to encode interactions but small enough to remain tractable
- Acts as a shared “physical space” for energies

**Interpretation**:
> This layer performs a **change of coordinates in energy space**, not classical feature learning.

---

### 4. Tanh Activation Before Second Ising Layer

**Why**:
- Keeps Ising biases bounded
- Avoids extreme bias values that trivialize annealing
- Preserves sign information

**Why not ReLU**:
- ReLU destroys symmetry
- Introduces artificial sparsity incompatible with Ising physics

---

### 5. Second Ising Layer (Composition of Hamiltonians)

**Why**:
- Enables **hierarchical energy modeling**
- Second layer reasons over energies produced by the first layer

**Key idea**:
> This architecture composes Hamiltonians instead of composing activations.

This is rare in standard neural networks and closer to concepts from statistical physics.

---

### 6. Residual Connection (Implicit)

The residual is applied **before** the second Ising layer.

**Why**:
- Prevents loss of information due to hard energy minimization
- Improves training stability
- Does not violate the physical interpretation of Ising modules

---

## What Problems Should This Network Perform Well On?

This architecture is **not universal**.  
It is designed for problems with specific structure.

### Well-suited problems ✅

#### 1. Combinatorial or Constraint-Based Problems
- SAT / Max-SAT (soft)
- Max-Cut
- Matching and assignment problems

**Why**:
These problems are naturally expressed as Ising Hamiltonians.

---

#### 2. Global Interaction Problems
- Parity-like tasks
- XOR with long-range dependencies
- Decisions depending on many interacting variables

**Why**:
Ising models naturally encode global dependencies.

---

#### 3. Energy Landscape Learning
- Systems with multiple competing minima
- Physical or chemical systems
- Discrete optimization with noise

---

#### 4. Small-Data, High-Bias Regimes
- Few samples
- Strong inductive structure
- Interpretability matters more than raw accuracy

---

### Poorly suited problems ❌

- Vision (images, CNN-style tasks)
- NLP and sequence modeling
- Smooth regression problems
- Large-scale noisy tabular datasets

For these, standard deep learning architectures are more efficient and accurate.

---

## Expected Behavior in Practice

- Training is **slower** than MLPs
- Gradients are **noisy but meaningful**
- Predictions tend to be **sharp**, with threshold-like behavior
- Learned parameters are **interpretable as interactions**

This model is intended primarily for **research and exploration**, not production-scale deployment.

---

## Summary

| Aspect | Characteristic |
|------|----------------|
Inductive bias | Strong, physics-inspired |
Function type | Energy-based, non-smooth |
Interpretability | High |
Scalability | Limited |
Best use | Structured, combinatorial problems |

---

## Final Note

This network is best viewed as a **deep energy-based model built from Ising systems**, not as a replacement for MLPs.

If your problem admits an energetic or combinatorial interpretation, this architecture provides a principled and expressive modeling approach.
