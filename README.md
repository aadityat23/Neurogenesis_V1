# NeuroGenesis

<p align="center">
  <b>Evolution, Neural Plasticity, and Genetic Assimilation in Volatile Environments</b>
</p>

<p align="center">
  Artificial Life • Evolutionary Computation • CTRNNs • Baldwin Effect • Adaptive Systems
</p>

---

## Overview

NeuroGenesis is an Artificial Life research framework designed to investigate one of the oldest questions in evolutionary theory:

> When should organisms learn, and when should evolution hard-code behavior directly into the genome?

The project combines:

* Continuous-Time Recurrent Neural Networks (CTRNNs)
* Hebbian synaptic plasticity
* Evolutionary optimization
* Dynamic ecological environments

to study how learning and evolution interact across different levels of environmental volatility.

The central objective is to experimentally investigate:

* The Baldwin Effect
* Genetic Assimilation
* Evolution of Plasticity
* Adaptation in Non-Stationary Environments

---

## Research Question

Can environmental volatility create selective pressure for neural plasticity?

More specifically:

| Question                                                    | Goal                        |
| ----------------------------------------------------------- | --------------------------- |
| Does learning improve adaptation in volatile worlds?        | Measure learning benefit    |
| Does evolution eventually replace learning?                 | Detect genetic assimilation |
| Is plasticity favored in unstable environments?             | Measure selection on η      |
| Are evolved neural controllers sufficient without learning? | Compare against controls    |

---

## Experimental Design

Four evolutionary conditions are evaluated under multiple volatility regimes.

### Experimental Conditions

| Condition  | Evolution | Hebbian Learning | Purpose                |
| ---------- | --------- | ---------------- | ---------------------- |
| EVO_ONLY   | ✓         | ✗                | Evolution alone        |
| LEARN_ONLY | ✗         | ✓                | Learning alone         |
| EVO_LEARN  | ✓         | ✓                | Evolution + Plasticity |
| BASELINE   | ✗         | ✗                | Control                |

---

### Environmental Volatility

Volatility is controlled by:

σ = number of simulation steps between resource relocations.

| σ   | Interpretation  |
| --- | --------------- |
| 50  | Highly volatile |
| 100 | Volatile        |
| 200 | Moderate        |
| 400 | Stable          |
| 800 | Very stable     |

Experimental sweep:

```python
VOLATILITY_VALUES = [50, 100, 200, 400, 800]
```

---

## Environment

### World

| Property         | Value     |
| ---------------- | --------- |
| Topology         | Toroidal  |
| Grid Size        | 100 × 100 |
| Resource Patches | 12        |
| Hazard Zones     | 5         |
| Agent Lifetime   | 200 steps |

Resources regenerate continuously while hazards impose energy penalties.

### Resource Dynamics

* Gaussian resource distributions
* Regenerative ecosystem
* Resource depletion through consumption
* Volatility-driven patch relocation

### Hazards

* Spatially distributed Gaussian danger fields
* Continuous energy damage
* Independent of resource locations

---

## Agent Architecture

Each organism is controlled by a Continuous-Time Recurrent Neural Network.

### CTRNN Structure

| Component       | Count |
| --------------- | ----- |
| Sensory Neurons | 8     |
| Hidden Neurons  | 6     |
| Motor Neurons   | 2     |
| Total Neurons   | 16    |

---

### Sensory Inputs

| Input          |
| -------------- |
| Resource North |
| Resource South |
| Resource East  |
| Resource West  |
| Hazard North   |
| Hazard South   |
| Hazard East    |
| Hazard West    |

---

### Motor Outputs

| Output  | Function           |
| ------- | ------------------ |
| Motor 1 | Movement Direction |
| Motor 2 | Consumption Gate   |

---

## Genome

Each organism possesses a 272-dimensional genome.

### Encoding

| Segment | Size | Purpose                |
| ------- | ---- | ---------------------- |
| W       | 128  | Synaptic weights       |
| τ       | 8    | Neuron time constants  |
| Bias    | 8    | Neural biases          |
| η       | 128  | Hebbian learning rates |

Total:

```text
Genome Size = 272 float32 values
```

---

## Learning Rule

Plastic agents modify synapses during their lifetime using Hebbian adaptation.

```text
Δw ∝ η × pre × post
```

where:

* η = genetically encoded learning rate
* pre = presynaptic activation
* post = postsynaptic activation

This allows agents to adapt within a single lifetime.

---

## Evolution

Evolution proceeds using:

| Mechanism       | Value      |
| --------------- | ---------- |
| Population Size | 150        |
| Selection       | Tournament |
| Tournament Size | 5          |
| Elitism         | 10%        |
| Mutation Rate   | 2%         |
| Mutation Sigma  | 0.15       |

No crossover is used.

---

## Metrics

NeuroGenesis records population-level and neural-level statistics every generation.

### Fitness Metrics

* Mean Fitness
* Median Fitness
* Maximum Fitness
* Survival Rate

### Neural Metrics

* Birth Weight Norm
* Death Weight Norm
* Weight Delta Frobenius

### Plasticity Metrics

* Mean η
* η Variance
* η-Fitness Correlation
* Top-10% η
* Bottom-10% η

---

## Genetic Assimilation Diagnostic

The primary Baldwin-effect indicator is:

```text
ΔW = ||W_death − W_birth||F
```

Interpretation:

| Observation                       | Meaning                            |
| --------------------------------- | ---------------------------------- |
| Large ΔW                          | Learning heavily modifies behavior |
| Shrinking ΔW over generations     | Genetic assimilation               |
| ΔW → 0 while fitness remains high | Learned behavior became innate     |

---

## Repository Structure

```text
neurogenesis/
│
├── main.py
├── config.py
│
├── agent/
│   ├── genome.py
│   ├── ctrnn.py
│   └── agent.py
│
├── environment/
│   ├── grid.py
│   └── volatility.py
│
├── evolution/
│   └── population.py
│
├── simulation/
│   └── simulator.py
│
├── experiment_log/
│   └── logger.py
│
├── analysis/
│   └── plots.py
│
└── tests/
    └── test_components.py
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/neurogenesis.git

cd neurogenesis

pip install -r requirements.txt
```

---

### Run Tests

```bash
python tests/test_components.py
```

---

### Smoke Test

```bash
python main.py --sigma 200 --n-gens 10 --no-log
```

---

### Single Experiment

```bash
python main.py --sigma 200
```

---

### Full Volatility Sweep

```bash
python main.py
```

---

### Automatic Analysis

```bash
python main.py --analyze
```

---

## Output Artifacts

Each experiment produces a fully reproducible run directory.

```text
data/runs/
└── sigma200_YYYYMMDD_HHMMSS/
    ├── config.json
    ├── generations.csv
    ├── agents_sampled.jsonl
    └── weights_genXXXXX.npz
```

---

## Current Findings

Initial experiments reveal:

* Evolution rapidly discovers highly effective foraging strategies.
* Hebbian plasticity produces substantial lifetime weight modification.
* Plasticity parameters currently exhibit weak evolutionary pressure.
* Preliminary evidence suggests the environment may be solvable through reactive policies without requiring memory.

These findings motivate ongoing investigations into:

* Sensor uncertainty
* Plasticity costs
* Hebbian ablation
* Environmental complexity
* Evolution of memory

---

## Reproducibility

Every run automatically records:

* Full configuration snapshot
* Volatility level
* Random seed
* Evolutionary statistics
* Neural weight trajectories

allowing exact experiment reconstruction.

---

## Roadmap

### NeuroGenesis v0.2

* Multi-seed statistical evaluation
* Hebbian ablation experiments
* Plasticity-cost mechanisms
* Sensor noise experiments
* Memory-demanding environments
* Publication-quality analysis suite

### NeuroGenesis v1.0

* Multi-agent ecosystems
* Predator-prey dynamics
* Co-evolution
* Variable-topology neural networks
* Open-ended evolution

---

## Citation

If you use NeuroGenesis in academic work:

```bibtex
@software{thokal_neurogenesis,
  author = {Aaditya Thokal},
  title = {NeuroGenesis: Evolution and Plasticity in Volatile Environments},
  year = {2026},
}
```

---

## Author

**Aaditya Thokal**

Artificial Life • Evolutionary Computation • Neural Systems • Adaptive Intelligence

*"When should evolution learn, and when should it remember?"*
