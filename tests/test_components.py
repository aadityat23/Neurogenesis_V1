#!/usr/bin/env python3
# tests/test_components.py
"""
Unit tests for all NeuroGenesis v0.1 components.
Run from project root:
    python tests/test_components.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import traceback


PASS = "✓"
FAIL = "✗"
results = []


def test(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS}  {name}")
    except Exception as e:
        results.append((FAIL, name))
        print(f"  {FAIL}  {name}")
        traceback.print_exc()


# -----------------------------------------------------------------------
# Genome tests
# -----------------------------------------------------------------------

def test_genome_shape():
    from agent.genome import Genome
    g = Genome.random()
    assert g.data.shape == (272,), f"Expected (272,), got {g.data.shape}"

def test_genome_decode_ranges():
    from agent.genome import Genome
    g = Genome.random()
    d = g.decode()
    assert d["W"].shape   == (8, 16)
    assert d["tau"].shape == (8,)
    assert d["bias"].shape== (8,)
    assert d["eta"].shape == (8, 16)
    assert d["tau"].min()  >= 0.499,  f"tau min={d['tau'].min()}"
    assert d["tau"].max()  <= 5.001,  f"tau max={d['tau'].max()}"
    assert d["bias"].min() >= -3.001, f"bias min={d['bias'].min()}"
    assert d["bias"].max() <= 3.001,  f"bias max={d['bias'].max()}"
    assert d["eta"].min()  >= -1e-9,  f"eta min={d['eta'].min()}"
    assert d["eta"].max()  <= 0.1001, f"eta max={d['eta'].max()}"

def test_genome_mutate_different():
    from agent.genome import Genome
    g  = Genome.random()
    g2 = g.mutate()
    assert not np.allclose(g.data, g2.data), "Mutated genome should differ"

def test_genome_copy_independent():
    from agent.genome import Genome
    g  = Genome.random()
    g2 = g.copy()
    g2.data[0] = 9999.0
    assert g.data[0] != 9999.0, "Copy should be independent"


# -----------------------------------------------------------------------
# CTRNN tests
# -----------------------------------------------------------------------

def test_ctrnn_output_range():
    from agent.genome import Genome
    from agent.ctrnn import CTRNN
    g    = Genome.random()
    net  = CTRNN(g.decode())
    sensors = np.random.rand(8).astype(np.float32)
    motor   = net.step(sensors)
    assert motor.shape == (2,), f"Motor shape {motor.shape}"
    assert motor.min() >= 0.0 and motor.max() <= 1.0, f"Motor out of [0,1]: {motor}"

def test_ctrnn_reset():
    from agent.genome import Genome
    from agent.ctrnn import CTRNN
    g   = Genome.random()
    net = CTRNN(g.decode())
    for _ in range(50):
        net.step(np.random.rand(8).astype(np.float32))
    net.reset()
    assert np.allclose(net.y, 0.5), "After reset, y should be 0.5"

def test_ctrnn_hebbian_changes_W():
    from agent.genome import Genome
    from agent.ctrnn import CTRNN
    g    = Genome.random()
    d    = g.decode()
    net  = CTRNN(d)
    W_before = net.W.copy()
    sensors  = np.full(8, 0.8, dtype=np.float32)
    net.step(sensors)
    net.apply_hebbian(d["eta"])
    assert not np.allclose(net.W, W_before), "Hebbian should modify W"

def test_ctrnn_weight_norm_cap():
    from agent.genome import Genome
    from agent.ctrnn import CTRNN
    from config import WEIGHT_NORM_CAP
    g   = Genome.random()
    d   = g.decode()
    d["eta"] = np.full_like(d["eta"], 0.1)   # max learning rate
    net = CTRNN(d)
    for _ in range(200):
        net.step(np.ones(8, dtype=np.float32))
        net.apply_hebbian(d["eta"])
    norms = np.linalg.norm(net.W, axis=1)
    assert (norms <= WEIGHT_NORM_CAP + 1e-5).all(), f"Norm exceeded cap: {norms.max()}"

def test_ctrnn_stable_output():
    """Network should converge to stable output for constant input."""
    from agent.genome import Genome
    from agent.ctrnn import CTRNN
    g       = Genome.random()
    net     = CTRNN(g.decode())
    sensors = np.full(8, 0.5, dtype=np.float32)
    prev    = None
    for _ in range(500):
        motor = net.step(sensors)
        prev  = motor
    # Just check it doesn't explode
    assert np.isfinite(prev).all(), f"Network output not finite: {prev}"


# -----------------------------------------------------------------------
# Grid tests
# -----------------------------------------------------------------------

def test_grid_init():
    from environment.grid import Grid
    g = Grid()
    assert g.resource_grid.shape == (100, 100)
    assert g.hazard_grid.shape   == (100, 100)
    assert g.resource_grid.max() <= 1.001
    assert g.hazard_grid.max()   <= 1.001

def test_grid_sensors_shape():
    from environment.grid import Grid
    g       = Grid()
    sensors = g.get_sensors(50, 50)
    assert sensors.shape == (8,), f"Expected (8,), got {sensors.shape}"
    assert sensors.min() >= 0.0 and sensors.max() <= 1.001

def test_grid_consume():
    from environment.grid import Grid
    g = Grid()
    g.resource_grid[50, 50] = 1.0
    energy = g.consume(50, 50)
    assert energy > 0, "Should gain energy"
    assert g.resource_grid[50, 50] < 1.0, "Cell should be depleted"

def test_grid_regen():
    from environment.grid import Grid
    g = Grid()
    g.resource_grid[50, 50] = 0.5
    before = float(g.resource_grid[50, 50])
    g.apply_regeneration()
    after  = float(g.resource_grid[50, 50])
    assert after > before, "Regeneration should increase resource"

def test_grid_toroidal_sensors():
    """Sensors at edge should wrap correctly."""
    from environment.grid import Grid
    g = Grid()
    sensors = g.get_sensors(0, 0)   # corner cell
    assert sensors.shape == (8,) and np.isfinite(sensors).all()


# -----------------------------------------------------------------------
# Volatility scheduler tests
# -----------------------------------------------------------------------

def test_volatility_triggers_at_sigma():
    from environment.grid import Grid
    from environment.volatility import VolatilityScheduler
    g   = Grid()
    vs  = VolatilityScheduler(volatility=10, grid=g)
    original = g.resource_grid.copy()

    for step in range(1, 35):
        vs.step(step)
        if step == 10:
            # Should have started transition
            assert vs.in_transition, "Should be in transition at step σ"

def test_volatility_ends_transition():
    from environment.grid import Grid
    from environment.volatility import VolatilityScheduler
    from config import TRANSITION_STEPS
    g  = Grid()
    # σ=50: first transition triggers at step 50, ends at step 50+TRANSITION_STEPS=70.
    # Next trigger at step 100. We stop at step 75 — safely after end, before next trigger.
    sigma = 50
    vs = VolatilityScheduler(volatility=sigma, grid=g)
    stop = sigma + TRANSITION_STEPS + 5   # step 75 — after end (70), before next trigger (100)
    for step in range(1, stop + 1):
        vs.step(step)
    assert not vs.in_transition, (
        f"Transition should have ended by step {stop} (σ={sigma}, "
        f"transition ends at step {sigma + TRANSITION_STEPS}), "
        f"but in_transition=True t={vs._transition_t}"
    )


# -----------------------------------------------------------------------
# Agent tests
# -----------------------------------------------------------------------

def test_agent_step():
    from environment.grid import Grid
    from agent.genome import Genome
    from agent.agent import Agent
    from config import Condition
    g     = Grid()
    genome= Genome.random()
    agent = Agent(genome, Condition.EVO_ONLY)
    agent.reset()
    agent.step(g)
    assert agent.age == 1
    assert 0.0 <= agent.energy <= 3.0

def test_agent_fitness():
    from environment.grid import Grid
    from agent.genome import Genome
    from agent.agent import Agent
    from config import Condition, LIFETIME
    g      = Grid()
    genome = Genome.random()
    agent  = Agent(genome, Condition.EVO_LEARN)
    agent.reset()
    for _ in range(LIFETIME):
        if agent.alive:
            agent.step(g)
    assert agent.fitness >= 0.0

def test_agent_death():
    from environment.grid import Grid
    from agent.genome import Genome
    from agent.agent import Agent
    from config import Condition, ENERGY_INIT
    g      = Grid()
    genome = Genome.random()
    agent  = Agent(genome, Condition.EVO_ONLY)
    agent.reset()
    agent.energy = 0.001   # near death
    # Drain energy by not eating and stepping into hazard
    g.hazard_grid[:] = 1.0  # full hazard everywhere
    agent.step(g)
    assert not agent.alive, "Agent should die from hazard"


# -----------------------------------------------------------------------
# Population tests
# -----------------------------------------------------------------------

def test_population_init():
    from evolution.population import Population
    from config import Condition, POPULATION_SIZE
    pop = Population(Condition.EVO_ONLY, volatility=200)
    assert len(pop.agents) == POPULATION_SIZE

def test_population_evolve():
    from environment.grid import Grid
    from evolution.population import Population
    from config import Condition, POPULATION_SIZE
    pop = Population(Condition.EVO_ONLY, volatility=200)
    g   = Grid()
    pop.reset_agents()
    for agent in pop.agents:
        for _ in range(10):
            if agent.alive:
                agent.step(g)
    pop.evolve()
    assert len(pop.agents) == POPULATION_SIZE

def test_population_baseline_no_evolve():
    from environment.grid import Grid
    from evolution.population import Population
    from config import Condition
    pop = Population(Condition.BASELINE, volatility=200)
    genome_data_before = pop.agents[0].genome.data.copy()
    pop.reset_agents()
    for agent in pop.agents:
        agent.step(Grid())
    pop.evolve()
    assert np.allclose(pop.agents[0].genome.data, genome_data_before), \
        "BASELINE genome should not change"


# -----------------------------------------------------------------------
# Mini simulation smoke test
# -----------------------------------------------------------------------

def test_mini_simulation():
    """Run a 3-generation, 5-step simulation — should not crash."""
    import config as cfg
    original_lifetime = cfg.LIFETIME
    original_gens     = cfg.N_GENERATIONS
    cfg.LIFETIME      = 5
    cfg.N_GENERATIONS = 3

    try:
        from simulation.simulator import Simulator
        sim = Simulator(volatility=50, logger=None)
        sim.run_experiment()
    finally:
        cfg.LIFETIME      = original_lifetime
        cfg.N_GENERATIONS = original_gens


# -----------------------------------------------------------------------
# Run all tests
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*55)
    print(" NeuroGenesis v0.1 — Test Suite")
    print("="*55 + "\n")

    print("[Genome]")
    test("genome_shape",            test_genome_shape)
    test("genome_decode_ranges",    test_genome_decode_ranges)
    test("genome_mutate_different", test_genome_mutate_different)
    test("genome_copy_independent", test_genome_copy_independent)

    print("\n[CTRNN]")
    test("ctrnn_output_range",      test_ctrnn_output_range)
    test("ctrnn_reset",             test_ctrnn_reset)
    test("ctrnn_hebbian_changes_W", test_ctrnn_hebbian_changes_W)
    test("ctrnn_weight_norm_cap",   test_ctrnn_weight_norm_cap)
    test("ctrnn_stable_output",     test_ctrnn_stable_output)

    print("\n[Grid]")
    test("grid_init",               test_grid_init)
    test("grid_sensors_shape",      test_grid_sensors_shape)
    test("grid_consume",            test_grid_consume)
    test("grid_regen",              test_grid_regen)
    test("grid_toroidal_sensors",   test_grid_toroidal_sensors)

    print("\n[Volatility]")
    test("volatility_triggers",     test_volatility_triggers_at_sigma)
    test("volatility_ends",         test_volatility_ends_transition)

    print("\n[Agent]")
    test("agent_step",              test_agent_step)
    test("agent_fitness",           test_agent_fitness)
    test("agent_death",             test_agent_death)

    print("\n[Population]")
    test("population_init",         test_population_init)
    test("population_evolve",       test_population_evolve)
    test("population_baseline",     test_population_baseline_no_evolve)

    print("\n[Integration]")
    test("mini_simulation",         test_mini_simulation)

    print("\n" + "="*55)
    passed = sum(1 for r, _ in results if r == PASS)
    failed = sum(1 for r, _ in results if r == FAIL)
    print(f" Results: {passed} passed, {failed} failed")
    print("="*55 + "\n")

    if failed > 0:
        sys.exit(1)
