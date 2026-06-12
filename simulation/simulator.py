# simulation/simulator.py
import numpy as np
import time
import config as cfg
from config import Condition, POPULATION_SIZE
from environment.grid import Grid
from environment.volatility import VolatilityScheduler
from evolution.population import Population


class Simulator:
    """
    Orchestrates one full experiment for a given volatility σ.

    Owns:
        - one Grid + VolatilityScheduler
        - four Population objects (one per condition)
        - an optional ExperimentLogger

    Usage:
        sim = Simulator(volatility=200, logger=my_logger)
        sim.run_experiment()
    """

    def __init__(self, volatility: int, logger=None):
        self.volatility = volatility
        self.logger = logger

        # Environment
        self.grid = Grid()
        self.scheduler = VolatilityScheduler(volatility, self.grid)

        # Four populations
        self.populations = {
            cond: Population(cond, volatility)
            for cond in Condition.ALL
        }
        self._population_list = list(self.populations.values())

        # Global step counter (persists across generations for scheduler)
        self._global_step = 0

        print(f"[Simulator] Initialized σ={volatility}")
        print(f"  Grid: {self.grid}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_experiment(self):
        """Run N_GENERATIONS generations."""
        t0_total = time.time()

        for gen in range(cfg.N_GENERATIONS):
            t0_gen = time.time()

            self.run_generation(gen)

            # Console progress
            elapsed = time.time() - t0_gen
            self._print_progress(gen, elapsed, t0_total)

            # Log
            if self.logger is not None:
                self.logger.log_generation(
                    gen,
                    self.populations,
                    self.volatility
                )

            # Evolve after logging
            for pop in self._population_list:
                pop.evolve()

        total_time = time.time() - t0_total
        print(f"\n[Simulator] σ={self.volatility} complete — {total_time:.1f}s total")

    # ------------------------------------------------------------------

    def run_generation(self, gen: int):
        """
        Run LIFETIME steps for all agents across all conditions.
        """

        # Reset agents
        for pop in self._population_list:
            pop.reset_agents()

        # Build flat agent list
        all_agents = [
            agent
            for pop in self._population_list
            for agent in pop.agents
        ]

        # Lifetime loop
        for t in range(cfg.LIFETIME):
            self.scheduler.step(self._global_step)
            self.grid.apply_regeneration()

            np.random.shuffle(all_agents)

            for agent in all_agents:
                if agent.alive:
                    agent.step(self.grid)

            self._global_step += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _print_progress(self, gen: int, gen_time: float, t0_total: float):
        elapsed = time.time() - t0_total

        lines = [
            f"[Gen {gen:4d}] σ={self.volatility} | {gen_time:.2f}s | total={elapsed:.0f}s"
        ]

        for cond, pop in self.populations.items():
            fits = pop.get_fitnesses()
            alive = sum(1 for a in pop.agents if a.alive)

            line = (
                f"  {cond:12s}: mean={fits.mean():.3f}  "
                f"max={fits.max():.3f}  "
                f"alive={alive}/{POPULATION_SIZE}"
            )

            if cond in (Condition.LEARN_ONLY, Condition.EVO_LEARN):
                mean_delta = np.mean([
                    a.weight_delta_frobenius()
                    for a in pop.agents
                ])

                line += f"  ΔW={mean_delta:.4f}"

            lines.append(line)

        print("\n".join(lines))

    def __repr__(self):
        return f"Simulator(σ={self.volatility}, step={self._global_step})"