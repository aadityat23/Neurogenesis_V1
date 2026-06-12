# NeuroGenesis v0.1

**Baldwin Effect simulation: Evolution vs. Hebbian Learning in volatile environments.**

---

## What this is

A 100×100 toroidal grid world where agents with CTRNNs forage for resources and avoid hazards. Four conditions test how evolution and within-lifetime learning interact across five levels of environmental volatility (σ).

| Condition    | Evolution | Hebbian Learning |
|--------------|-----------|-----------------|
| `EVO_ONLY`   | ✓         | ✗               |
| `LEARN_ONLY` | ✗         | ✓               |
| `EVO_LEARN`  | ✓         | ✓               |
| `BASELINE`   | ✗         | ✗               |

Volatility σ = steps between resource reshuffles. Sweep: **[50, 100, 200, 400, 800]**.

---

## Quickstart

```bash
pip install -r requirements.txt

# Smoke test (5 gens, no disk writes)
python main.py --sigma 200 --n-gens 5 --no-log

# Run tests
python tests/test_components.py

# Single σ run (1000 generations, ~5 min)
python main.py --sigma 200

# Full experiment (all 5 σ values, ~25 min)
python main.py

# Full experiment + auto-analyze
python main.py --analyze
```

---

## Implementation Order (follow this)

```
Step 1: config.py                   ← done
Step 2: agent/genome.py             ← done
Step 3: agent/ctrnn.py              ← done
Step 4: environment/grid.py         ← done
       environment/volatility.py    ← done
Step 5: agent/agent.py              ← done
Step 6: evolution/population.py     ← done
Step 7: simulation/simulator.py     ← done
Step 8: logging/logger.py           ← done
Step 9: main.py                     ← done
Step 10: analysis/plots.py          ← done (run after data exists)
```

---

## Architecture

```
neurogenesis/
├── main.py                 # entry point
├── config.py               # all constants
├── environment/
│   ├── grid.py             # Grid: resource + hazard layers
│   └── volatility.py       # VolatilityScheduler: patch teleportation
├── agent/
│   ├── genome.py           # Genome: 272-float flat array, decode/mutate
│   ├── ctrnn.py            # CTRNN: Euler-discretized, Hebbian updates
│   └── agent.py            # Agent: sense→think→act→learn loop
├── evolution/
│   └── population.py       # Population: tournament selection + elitism
├── simulation/
│   └── simulator.py        # Simulator: main orchestrator
├── logging/
│   └── logger.py           # ExperimentLogger: CSV + JSONL + NPZ
├── analysis/
│   └── plots.py            # Post-hoc plots (run after data)
└── tests/
    └── test_components.py  # Unit + integration tests
```

---

## Genome layout (272 float32 values)

```
[0:128]   W      — weight matrix (8×16), unconstrained
[128:136] tau    — time constants → [0.5, 5.0] via sigmoid
[136:144] bias   — neuron biases → [-3.0, 3.0] via tanh
[144:272] eta    — Hebbian rates → [0.0, 0.1] via sigmoid
```

---

## CTRNN

- 8 sensory inputs (4× resource N/S/E/W, 4× hazard N/S/E/W)
- 6 hidden neurons (recurrent)
- 2 motor outputs: movement direction + eat gate
- Euler dt=0.1, τ ∈ [0.5, 5.0] → effective memory 5–50 steps

---

## Output files

```
data/runs/sigma200_YYYYMMDD_HHMMSS_XXXXXX/
├── config.json              # full config snapshot
├── generations.csv          # one row per generation per condition
├── agents_sampled.jsonl     # per-agent detail every 10 gens
└── weights_gen*.npz         # weight snapshots every 50 gens
```

### Key metric: `mean_weight_delta_frobenius`

This is `||W_death - W_birth||_F` averaged over agents. In `EVO_LEARN`, if this **decreases** over generations, genetic assimilation is occurring — that's the Baldwin Effect.

---

## Sanity checks (first experiment success criteria)

1. `EVO_ONLY` and `EVO_LEARN` show upward fitness trend
2. `BASELINE` fitness is flat
3. `LEARN_ONLY` fitness at σ=50 > `LEARN_ONLY` at σ=800
4. No crashes over 1000 generations

---

## Expected performance (CPU / NumPy)

- ~0.3s per generation
- ~5 min per σ value (1000 gens)
- ~25 min full experiment

If you see >60s/generation: profile `agent.step()`, vectorize population batch.

---

## What NOT to add in v0.1

- Multi-species, crossover, variable topology
- GPU/parallelization (only if >60s/gen)
- Real-time visualization
- Predator agents, niche construction
- Multi-seed averaging (do this in v0.2)
