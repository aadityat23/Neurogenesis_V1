# evolution/population.py
import numpy as np
from config import (
    POPULATION_SIZE, ELITE_FRACTION, TOURNAMENT_K, Condition, GRID_SIZE,
)
from agent.genome import Genome
from agent.agent import Agent


class Population:
    """
    Manages one population of POPULATION_SIZE agents for a single condition.

    Knows about selection and mutation. Does NOT know about the grid or steps.
    """

    def __init__(self, condition: str, volatility: int, base_genome: Genome = None):
        """
        Args:
            condition:   one of Condition.{EVO_ONLY, LEARN_ONLY, EVO_LEARN, BASELINE}
            volatility:  σ value (stored for logging; not used in selection)
            base_genome: if not None, clone this genome for BASELINE condition
        """
        self.condition  = condition
        self.volatility = volatility

        self.n_elite = max(1, int(POPULATION_SIZE * ELITE_FRACTION))

        if condition == Condition.BASELINE:
            # One fixed random genome, cloned across all agents — never evolved
            if base_genome is None:
                base_genome = Genome.random()
            self._baseline_genome = base_genome
            self.agents = [
                Agent(base_genome.copy(), condition) for _ in range(POPULATION_SIZE)
            ]
        else:
            self.agents = [
                Agent(Genome.random(), condition) for _ in range(POPULATION_SIZE)
            ]

    # ------------------------------------------------------------------
    # Reset for new generation
    # ------------------------------------------------------------------

    def reset_agents(self):
        """Reset all agents to birth state with random spawn positions."""
        for agent in self.agents:
            agent.reset(GRID_SIZE)

    # ------------------------------------------------------------------
    # Fitness access
    # ------------------------------------------------------------------

    def get_fitnesses(self) -> np.ndarray:
        return np.array([a.fitness for a in self.agents], dtype=np.float64)

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve(self):
        """
        Selection + reproduction. Updates self.agents with next generation.

        - BASELINE:    do nothing (genome never changes)
        - LEARN_ONLY:  re-randomize all genomes (no selection, pure random)
        - EVO_ONLY:    truncation elite + tournament selection + mutation
        - EVO_LEARN:   same as EVO_ONLY
        """
        if self.condition == Condition.BASELINE:
            # Frozen — just reset agents (genomes unchanged)
            return

        if self.condition == Condition.LEARN_ONLY:
            # No selection — randomize each generation to isolate within-lifetime learning
            self.agents = [
                Agent(Genome.random(), self.condition) for _ in range(POPULATION_SIZE)
            ]
            return

        # EVO_ONLY and EVO_LEARN: real evolution
        fitnesses = self.get_fitnesses()
        sorted_idx = np.argsort(fitnesses)[::-1]   # descending

        new_genomes = []

        # Elites pass unchanged
        for i in range(self.n_elite):
            new_genomes.append(self.agents[sorted_idx[i]].genome.copy())

        # Rest: tournament selection + mutation
        while len(new_genomes) < POPULATION_SIZE:
            parent_genome = self._tournament_select(fitnesses)
            new_genomes.append(parent_genome.mutate())

        self.agents = [
            Agent(g, self.condition) for g in new_genomes
        ]

    def _tournament_select(self, fitnesses: np.ndarray) -> Genome:
        """Tournament selection. Returns winner's genome copy."""
        contestants = np.random.choice(POPULATION_SIZE, TOURNAMENT_K, replace=False)
        winner_idx  = contestants[np.argmax(fitnesses[contestants])]
        return self.agents[winner_idx].genome.copy()

    # ------------------------------------------------------------------
    # Weight snapshots for Baldwin tracking
    # ------------------------------------------------------------------

    def get_top_birth_weights(self, n: int = 10) -> np.ndarray:
        """Stack birth weight matrices of top-n agents. Shape: (n, 8, 16)."""
        fitnesses  = self.get_fitnesses()
        top_idx    = np.argsort(fitnesses)[-n:]
        Ws = []
        for i in top_idx:
            bw = self.agents[i]._birth_W
            if bw is not None:
                Ws.append(bw)
        return np.stack(Ws) if Ws else np.zeros((1, 8, 16), dtype=np.float32)

    def get_top_death_weights(self, n: int = 10) -> np.ndarray:
        """Stack live weight matrices of top-n agents. Shape: (n, 8, 16)."""
        fitnesses = self.get_fitnesses()
        top_idx   = np.argsort(fitnesses)[-n:]
        return np.stack([self.agents[i].ctrnn.W for i in top_idx])

    # ------------------------------------------------------------------
    # Plasticity diagnostics
    # ------------------------------------------------------------------

    def get_plasticity_stats(self) -> dict:
        """
        Compute population-level plasticity diagnostics. Called by the
        logger once per generation, after the lifetime has run.

        Returns a dict with keys:
            mean_eta        — population mean of per-synapse η (primary metric)
            std_eta         — spread: is η converging or staying diffuse?
            min_eta         — lower bound: how close to zero does selection push?
            max_eta         — upper bound
            eta_fit_corr    — Pearson r between mean_eta and fitness across all
                              agents. Negative = evolution penalises plasticity.
                              Only meaningful for EVO_ONLY and EVO_LEARN.
            mean_eta_top10  — mean η of the top-10% agents by fitness
            mean_eta_bot10  — mean η of the bottom-10% agents by fitness
            eta_top_bot_gap — mean_eta_top10 - mean_eta_bot10. Negative means
                              fitter agents have lower plasticity (suppression).
        """
        etas     = np.array([a.mean_eta()  for a in self.agents], dtype=np.float64)
        fits     = self.get_fitnesses()

        n        = len(self.agents)
        n10      = max(1, n // 10)
        sorted_by_fit = np.argsort(fits)
        top_idx  = sorted_by_fit[-n10:]
        bot_idx  = sorted_by_fit[:n10]

        mean_eta_top = float(etas[top_idx].mean())
        mean_eta_bot = float(etas[bot_idx].mean())

        # Pearson correlation — skip if all eta or all fitness values are identical
        if etas.std() > 1e-10 and fits.std() > 1e-10:
            eta_fit_corr = float(np.corrcoef(etas, fits)[0, 1])
        else:
            eta_fit_corr = 0.0

        return {
            "mean_eta":        float(etas.mean()),
            "std_eta":         float(etas.std()),
            "min_eta":         float(etas.min()),
            "max_eta":         float(etas.max()),
            "eta_fit_corr":    eta_fit_corr,
            "mean_eta_top10":  mean_eta_top,
            "mean_eta_bot10":  mean_eta_bot,
            "eta_top_bot_gap": mean_eta_top - mean_eta_bot,
        }

    def __repr__(self):
        fitnesses = self.get_fitnesses()
        return (
            f"Population({self.condition}, "
            f"n={len(self.agents)}, "
            f"mean_fit={fitnesses.mean():.3f}, "
            f"max_fit={fitnesses.max():.3f})"
        )