# analysis/plots.py
"""
Post-hoc analysis and plotting for NeuroGenesis experiments.

Run after data exists:
    python -m analysis.plots data/runs/sigma200_*/

All plots saved as PNG in the run directory. No GUI required.
"""
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Optional


# -----------------------------------------------------------------------
# Colour palette per condition
# -----------------------------------------------------------------------
CONDITION_COLORS = {
    "EVO_ONLY":   "#2196F3",   # blue
    "EVO_LEARN":  "#4CAF50",   # green
    "LEARN_ONLY": "#FF9800",   # orange
    "BASELINE":   "#9E9E9E",   # grey
}

CONDITION_ORDER = ["EVO_ONLY", "EVO_LEARN", "LEARN_ONLY", "BASELINE"]


# -----------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------

def load_generations(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "generations.csv")
    df = pd.read_csv(path)
    return df


def load_config(run_dir: str) -> dict:
    path = os.path.join(run_dir, "config.json")
    with open(path) as f:
        return json.load(f)


def load_agents(run_dir: str) -> pd.DataFrame:
    import json
    path = os.path.join(run_dir, "agents_sampled.jsonl")
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


# -----------------------------------------------------------------------
# Plot 1: Fitness over generations (per condition)
# -----------------------------------------------------------------------

def plot_fitness_curves(df: pd.DataFrame, config: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    sigma = config.get("volatility", "?")

    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond].sort_values("generation")
        if sub.empty:
            continue

        # Smooth with rolling window
        mean_s = sub["mean_fitness"].rolling(window=10, min_periods=1).mean()
        max_s  = sub["max_fitness"].rolling(window=10, min_periods=1).mean()

        color = CONDITION_COLORS.get(cond, "#000000")
        ax.plot(sub["generation"], mean_s, label=f"{cond} (mean)", color=color, linewidth=2)
        ax.plot(sub["generation"], max_s, label=f"{cond} (max)",  color=color, linewidth=1,
                linestyle="--", alpha=0.6)
        ax.fill_between(
            sub["generation"],
            mean_s - sub["std_fitness"].rolling(10, min_periods=1).mean(),
            mean_s + sub["std_fitness"].rolling(10, min_periods=1).mean(),
            color=color, alpha=0.1
        )

    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Mean Agent Fitness", fontsize=13)
    ax.set_title(f"Fitness Curves — σ={sigma}", fontsize=15)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "fitness_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------
# Plot 2: Baldwin Effect — weight delta over generations
# -----------------------------------------------------------------------

def plot_baldwin_signal(df: pd.DataFrame, config: dict, out_dir: str):
    """
    Weight delta (Frobenius norm of W_death - W_birth) over generations.
    Decreasing delta in EVO_LEARN = assimilation (Baldwin Effect).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sigma = config.get("volatility", "?")

    for cond in ["EVO_LEARN", "LEARN_ONLY"]:
        sub = df[df["condition"] == cond].sort_values("generation")
        if sub.empty:
            continue
        delta_s = sub["mean_weight_delta_frobenius"].rolling(window=20, min_periods=1).mean()
        color   = CONDITION_COLORS.get(cond, "#000")
        ax.plot(sub["generation"], delta_s, label=cond, color=color, linewidth=2)

    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Mean ||W_death − W_birth||_F", fontsize=12)
    ax.set_title(f"Baldwin Assimilation Signal — σ={sigma}", fontsize=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "baldwin_signal.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------
# Plot 3: Survival rate over generations
# -----------------------------------------------------------------------

def plot_survival(df: pd.DataFrame, config: dict, out_dir: str):
    pop_size = config.get("POPULATION_SIZE", 150)
    sigma    = config.get("volatility", "?")

    fig, ax = plt.subplots(figsize=(10, 5))

    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond].sort_values("generation")
        if sub.empty:
            continue
        survival = sub["n_alive_at_end"] / pop_size
        survival_s = survival.rolling(10, min_periods=1).mean()
        color = CONDITION_COLORS.get(cond, "#000")
        ax.plot(sub["generation"], survival_s, label=cond, color=color, linewidth=2)

    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Fraction Alive at End of Lifetime", fontsize=12)
    ax.set_title(f"Survival Rate — σ={sigma}", fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "survival_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------
# Plot 4: Phase diagram — fitness vs volatility (needs multiple runs)
# -----------------------------------------------------------------------

def plot_phase_diagram(run_dirs: List[str], out_dir: str, generation_window: int = 100):
    """
    Each run_dir is one σ value. Plots mean fitness in final `generation_window`
    generations vs σ, per condition.
    """
    data = {}   # {sigma: {condition: mean_late_fitness}}

    for rd in run_dirs:
        try:
            cfg = load_config(rd)
            df  = load_generations(rd)
        except Exception as e:
            print(f"  Skipping {rd}: {e}")
            continue

        sigma = cfg.get("volatility", -1)
        last_gens = df[df["generation"] >= df["generation"].max() - generation_window]

        data[sigma] = {}
        for cond in CONDITION_ORDER:
            sub = last_gens[last_gens["condition"] == cond]
            if not sub.empty:
                data[sigma][cond] = sub["mean_fitness"].mean()

    if not data:
        print("  No data for phase diagram.")
        return

    sigmas = sorted(data.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in CONDITION_ORDER:
        ys = [data[s].get(cond, np.nan) for s in sigmas]
        color = CONDITION_COLORS.get(cond, "#000")
        ax.plot(sigmas, ys, marker="o", label=cond, color=color, linewidth=2, markersize=8)

    ax.set_xscale("log")
    ax.set_xticks(sigmas)
    ax.set_xticklabels([str(s) for s in sigmas])
    ax.set_xlabel("Volatility σ (steps between reshuffles)", fontsize=13)
    ax.set_ylabel(f"Mean Fitness (last {generation_window} gens)", fontsize=12)
    ax.set_title("Phase Diagram: Fitness vs. Environmental Volatility", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "phase_diagram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------
# Plot 5: Tau and Eta evolution (EVO conditions)
# -----------------------------------------------------------------------

def plot_network_params(agents_df: pd.DataFrame, config: dict, out_dir: str):
    sigma = config.get("volatility", "?")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for cond in ["EVO_ONLY", "EVO_LEARN"]:
        sub   = agents_df[agents_df["condition"] == cond].sort_values("generation")
        if sub.empty:
            continue
        color = CONDITION_COLORS.get(cond, "#000")
        grp   = sub.groupby("generation")
        gens  = sorted(sub["generation"].unique())

        tau_mean = [grp.get_group(g)["mean_tau"].mean() for g in gens]
        eta_mean = [grp.get_group(g)["mean_eta"].mean() for g in gens]

        axes[0].plot(gens, tau_mean, label=cond, color=color, linewidth=2)
        axes[1].plot(gens, eta_mean, label=cond, color=color, linewidth=2)

    axes[0].set_title(f"Mean τ (time constant) — σ={sigma}")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Mean τ")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f"Mean η (learning rate) — σ={sigma}")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Mean η")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "network_params.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------

def analyze_run(run_dir: str):
    print(f"\n[Analysis] {run_dir}")
    cfg    = load_config(run_dir)
    df     = load_generations(run_dir)
    out    = run_dir   # save plots alongside data

    plot_fitness_curves(df, cfg, out)
    plot_baldwin_signal(df, cfg, out)
    plot_survival(df, cfg, out)

    agents_path = os.path.join(run_dir, "agents_sampled.jsonl")
    if os.path.exists(agents_path):
        agents_df = load_agents(run_dir)
        if not agents_df.empty:
            plot_network_params(agents_df, cfg, out)


def analyze_all(run_dirs: List[str], phase_diagram_dir: str = "data/runs"):
    for rd in run_dirs:
        analyze_run(rd)

    # Phase diagram requires multiple σ values
    if len(run_dirs) > 1:
        print("\n[Analysis] Building phase diagram...")
        os.makedirs(phase_diagram_dir, exist_ok=True)
        plot_phase_diagram(run_dirs, phase_diagram_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze NeuroGenesis experiment runs")
    parser.add_argument("run_dirs", nargs="+", help="Paths to run directories")
    parser.add_argument("--phase-dir", default="data/runs",
                        help="Where to save the phase diagram (default: data/runs)")
    args = parser.parse_args()

    # Expand globs
    expanded = []
    for p in args.run_dirs:
        expanded.extend(glob.glob(p) or [p])

    analyze_all(expanded, args.phase_dir)
