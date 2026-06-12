# experiment_log/logger.py
import os
import csv
import json
import hashlib
import datetime
import numpy as np
from config import (
    LOG_AGENTS_EVERY, LOG_WEIGHTS_EVERY, N_AGENTS_SAMPLE,
    GRID_SIZE, N_PATCHES, PATCH_SIGMA, RESOURCE_REGEN, N_HAZARDS,
    HAZARD_SIGMA, HAZARD_DAMAGE, LIFETIME, POPULATION_SIZE, N_GENERATIONS,
    ELITE_FRACTION, TOURNAMENT_K, MUTATION_RATE, MUTATION_SIGMA,
    N_SENSORY, N_HIDDEN, N_MOTOR, DT, ENERGY_INIT, ENERGY_MAX, ENERGY_STEP,
    VOLATILITY_VALUES,
)


CONFIG_SNAPSHOT = {
    "GRID_SIZE": GRID_SIZE, "N_PATCHES": N_PATCHES, "PATCH_SIGMA": PATCH_SIGMA,
    "RESOURCE_REGEN": RESOURCE_REGEN, "N_HAZARDS": N_HAZARDS,
    "HAZARD_SIGMA": HAZARD_SIGMA, "HAZARD_DAMAGE": HAZARD_DAMAGE,
    "LIFETIME": LIFETIME, "POPULATION_SIZE": POPULATION_SIZE,
    "N_GENERATIONS": N_GENERATIONS, "ELITE_FRACTION": ELITE_FRACTION,
    "TOURNAMENT_K": TOURNAMENT_K, "MUTATION_RATE": MUTATION_RATE,
    "MUTATION_SIGMA": MUTATION_SIGMA, "N_SENSORY": N_SENSORY,
    "N_HIDDEN": N_HIDDEN, "N_MOTOR": N_MOTOR, "DT": DT,
    "ENERGY_INIT": ENERGY_INIT, "ENERGY_MAX": ENERGY_MAX, "ENERGY_STEP": ENERGY_STEP,
    "VOLATILITY_VALUES": VOLATILITY_VALUES,
}

_CSV_FIELDS = [
    "generation", "condition",
    "mean_fitness", "std_fitness", "median_fitness", "max_fitness", "min_fitness",
    "n_alive_at_end",
    "mean_birth_weight_norm", "mean_death_weight_norm", "mean_weight_delta_frobenius",
    # --- plasticity diagnostics (new) ---
    "mean_eta",         # population mean of per-synapse η — primary metric
    "std_eta",          # spread of η across the population
    "min_eta",          # lowest η in population
    "max_eta",          # highest η in population
    "eta_fit_corr",     # Pearson r(η, fitness) — negative = suppression signal
    "mean_eta_top10",   # mean η of fittest 10% of agents
    "mean_eta_bot10",   # mean η of weakest 10% of agents
    "eta_top_bot_gap",  # mean_eta_top10 - mean_eta_bot10 — sign is the key test
]


class ExperimentLogger:
    """
    Creates a timestamped run directory and writes:
        - config.json
        - generations.csv  (one row per generation per condition)
        - agents_sampled.jsonl  (detailed per-agent, sampled every LOG_AGENTS_EVERY gens)
        - weights_sampled.npz  (weight snapshots every LOG_WEIGHTS_EVERY gens)
    """

    def __init__(self, volatility: int, base_dir: str = "data/runs"):
        timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(
            json.dumps(CONFIG_SNAPSHOT, sort_keys=True).encode()
        ).hexdigest()[:6]

        run_name = f"sigma{volatility}_{timestamp}_{config_hash}"
        self.run_dir = os.path.join(base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # Write config
        config_out = dict(CONFIG_SNAPSHOT)
        config_out["volatility"] = volatility
        config_out["run_name"]   = run_name
        try:
            import subprocess
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            config_out["git_commit"] = git_hash
        except Exception:
            config_out["git_commit"] = "unknown"

        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config_out, f, indent=2)

        # Open generations.csv
        self._gen_csv_path = os.path.join(self.run_dir, "generations.csv")
        self._gen_csv_fh   = open(self._gen_csv_path, "w", newline="")
        self._gen_writer   = csv.DictWriter(self._gen_csv_fh, fieldnames=_CSV_FIELDS)
        self._gen_writer.writeheader()
        self._gen_csv_fh.flush()

        # agents_sampled.jsonl — open in append mode
        self._agents_path = os.path.join(self.run_dir, "agents_sampled.jsonl")
        self._agents_fh   = open(self._agents_path, "a")

        # weights_sampled.npz — collected in memory, saved at milestones
        self._weight_snapshots = {}   # key: gen → {condition: {birth, death}}

        print(f"[Logger] Writing to {self.run_dir}")

    # ------------------------------------------------------------------

    def log_generation(self, gen: int, populations: dict, volatility: int):
        """
        Called at the end of every generation.

        Args:
            gen:         generation index
            populations: dict {condition_str: Population}
            volatility:  σ value (for context)
        """
        # 1. generations.csv — always
        for cond, pop in populations.items():
            fits   = pop.get_fitnesses()
            agents = pop.agents
            alive  = sum(1 for a in agents if a.alive)

            bw_norms = [a.birth_weight_norm() for a in agents]
            dw_norms = [a.death_weight_norm() for a in agents]
            deltas   = [a.weight_delta_frobenius() for a in agents]

            # Plasticity stats — one call, all metrics
            pstats = pop.get_plasticity_stats()

            row = {
                "generation":                  gen,
                "condition":                   cond,
                "mean_fitness":                float(np.mean(fits)),
                "std_fitness":                 float(np.std(fits)),
                "median_fitness":              float(np.median(fits)),
                "max_fitness":                 float(np.max(fits)),
                "min_fitness":                 float(np.min(fits)),
                "n_alive_at_end":              alive,
                "mean_birth_weight_norm":      float(np.mean(bw_norms)),
                "mean_death_weight_norm":      float(np.mean(dw_norms)),
                "mean_weight_delta_frobenius": float(np.mean(deltas)),
                # plasticity diagnostics
                "mean_eta":        pstats["mean_eta"],
                "std_eta":         pstats["std_eta"],
                "min_eta":         pstats["min_eta"],
                "max_eta":         pstats["max_eta"],
                "eta_fit_corr":    pstats["eta_fit_corr"],
                "mean_eta_top10":  pstats["mean_eta_top10"],
                "mean_eta_bot10":  pstats["mean_eta_bot10"],
                "eta_top_bot_gap": pstats["eta_top_bot_gap"],
            }
            self._gen_writer.writerow(row)
        self._gen_csv_fh.flush()

        # 2. agents_sampled.jsonl — every LOG_AGENTS_EVERY gens
        if gen % LOG_AGENTS_EVERY == 0:
            self._log_agent_sample(gen, populations)

        # 3. weights_sampled.npz — every LOG_WEIGHTS_EVERY gens
        if gen % LOG_WEIGHTS_EVERY == 0:
            self._log_weight_snapshot(gen, populations)

    # ------------------------------------------------------------------

    def _log_agent_sample(self, gen: int, populations: dict):
        for cond, pop in populations.items():
            fits    = pop.get_fitnesses()
            indices = np.random.choice(len(pop.agents), min(N_AGENTS_SAMPLE, len(pop.agents)), replace=False)
            for i, idx in enumerate(indices):
                agent = pop.agents[idx]
                record = {
                    "generation":             gen,
                    "condition":              cond,
                    "agent_id":               agent.agent_id,
                    "fitness":                round(agent.fitness, 6),
                    "age_at_death":           agent.age,
                    "birth_weight_norm":      round(agent.birth_weight_norm(), 4),
                    "death_weight_norm":      round(agent.death_weight_norm(), 4),
                    "weight_delta_frobenius": round(agent.weight_delta_frobenius(), 4),
                    # eta distribution per agent
                    "mean_eta":               round(agent.mean_eta(), 6),
                    "std_eta":                round(agent.std_eta(), 6),
                    "min_eta":                round(agent.min_eta(), 6),
                    "max_eta":                round(agent.max_eta(), 6),
                    "mean_tau":               round(agent.mean_tau(), 4),
                }
                self._agents_fh.write(json.dumps(record) + "\n")
        self._agents_fh.flush()

    def _log_weight_snapshot(self, gen: int, populations: dict):
        arrays = {}
        for cond, pop in populations.items():
            birth_W = pop.get_top_birth_weights(n=10)
            death_W = pop.get_top_death_weights(n=10)
            arrays[f"{cond}_gen{gen}_birth"] = birth_W
            arrays[f"{cond}_gen{gen}_death"] = death_W

        path = os.path.join(self.run_dir, f"weights_gen{gen:05d}.npz")
        np.savez_compressed(path, **arrays)

    # ------------------------------------------------------------------

    def close(self):
        self._gen_csv_fh.close()
        self._agents_fh.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"ExperimentLogger(run_dir={self.run_dir})"