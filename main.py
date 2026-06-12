#!/usr/bin/env python3
# main.py — NeuroGenesis v0.1 experiment runner
"""
Usage:
    # Full experiment (all 5 σ values)
    python main.py

    # Single σ value (fast test)
    python main.py --sigma 200

    # Short smoke test (10 generations)
    python main.py --sigma 200 --n-gens 10

    # Disable logging (just run, no disk writes)
    python main.py --no-log --sigma 200 --n-gens 5
"""
import argparse
import os
import sys
import time
import numpy as np

# Ensure package root is on path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import VOLATILITY_VALUES, N_GENERATIONS
from simulation.simulator import Simulator
from experiment_log.logger import ExperimentLogger


def run_sigma(sigma: int, n_gens: int, use_logging: bool, seed: int):
    """Run one experiment for a given σ value."""
    np.random.seed(seed)
    print(f"\n{'='*60}")
    print(f" NeuroGenesis v0.1 | σ={sigma} | {n_gens} generations | seed={seed}")
    print(f"{'='*60}")

    import config as cfg
    cfg.N_GENERATIONS = n_gens

    logger = None
    if use_logging:
        from experiment_log.logger import ExperimentLogger
        logger = ExperimentLogger(volatility=sigma)

    try:
        sim = Simulator(volatility=sigma, logger=logger)
        sim.run_experiment()
    finally:
        if logger is not None:
            logger.close()

    return sim


def main():
    parser = argparse.ArgumentParser(
        description="NeuroGenesis v0.1 — Baldwin Effect simulation"
    )
    parser.add_argument(
        "--sigma", type=int, default=None,
        help="Single σ value to run (default: run all 5 values)"
    )
    parser.add_argument(
        "--n-gens", type=int, default=None,
        help="Override number of generations (default: use config value)"
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Disable disk logging (for smoke tests)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run analysis/plots after experiment completes"
    )
    args = parser.parse_args()

    # Determine which σ values to run
    sigmas = [args.sigma] if args.sigma is not None else VOLATILITY_VALUES
    n_gens = args.n_gens if args.n_gens is not None else N_GENERATIONS

    print(f"\nNeuroGenesis v0.1")
    print(f"  σ values : {sigmas}")
    print(f"  gens     : {n_gens}")
    print(f"  logging  : {'disabled' if args.no_log else 'enabled'}")
    print(f"  seed     : {args.seed}")

    run_dirs = []
    t0 = time.time()

    for sigma in sigmas:
        # Use different seeds for different σ to make runs independent
        seed = args.seed + sigma
        sim = run_sigma(
            sigma=sigma,
            n_gens=n_gens,
            use_logging=not args.no_log,
            seed=seed,
        )
        # Collect run dir for analysis
        # (logger stores it; if no logging we skip)

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f" All runs complete in {total:.1f}s")
    print(f"{'='*60}")

    if args.analyze and not args.no_log:
        from analysis.plots import analyze_all
        import glob
        all_runs = glob.glob("data/runs/*/config.json")
        run_dirs = [os.path.dirname(p) for p in all_runs]
        if run_dirs:
            analyze_all(run_dirs)
        else:
            print("No run directories found for analysis.")


if __name__ == "__main__":
    main()
