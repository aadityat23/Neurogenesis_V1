# agent/genome.py
import numpy as np
from config import (
    GENOME_SIZE, GENOME_W_START, GENOME_W_END,
    GENOME_TAU_START, GENOME_TAU_END,
    GENOME_BIAS_START, GENOME_BIAS_END,
    GENOME_ETA_START, GENOME_ETA_END,
    N_RECURRENT, N_TOTAL,
    TAU_MIN, TAU_MAX, BIAS_RANGE, ETA_MAX,
    MUTATION_RATE, MUTATION_SIGMA,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class Genome:
    """
    Flat float32 array of length 272. All values unconstrained.
    Decoding maps to valid parameter ranges via sigmoid/tanh.

    Layout:
        [0:128]   W        — weight matrix, row-major
        [128:136] tau      — time constants (decoded to [0.5, 5.0])
        [136:144] bias     — biases (decoded to [-3.0, 3.0])
        [144:272] eta      — Hebbian rates (decoded to [0.0, 0.1])
    """

    def __init__(self, data: np.ndarray):
        assert data.shape == (GENOME_SIZE,), f"Expected ({GENOME_SIZE},), got {data.shape}"
        self.data = data.astype(np.float32)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def random(cls) -> "Genome":
        """Random genome sampled from N(0, 1)."""
        return cls(np.random.randn(GENOME_SIZE).astype(np.float32))

    def copy(self) -> "Genome":
        return Genome(self.data.copy())

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self) -> dict:
        """
        Returns parameter-space values:
            W    : (8, 16) float32 — unconstrained weights
            tau  : (8,)    float32 — time constants in [0.5, 5.0]
            bias : (8,)    float32 — biases in [-3.0, 3.0]
            eta  : (8, 16) float32 — Hebbian rates in [0.0, 0.1]
        """
        raw_W    = self.data[GENOME_W_START:GENOME_W_END]
        raw_tau  = self.data[GENOME_TAU_START:GENOME_TAU_END]
        raw_bias = self.data[GENOME_BIAS_START:GENOME_BIAS_END]
        raw_eta  = self.data[GENOME_ETA_START:GENOME_ETA_END]

        W    = raw_W.reshape(N_RECURRENT, N_TOTAL).copy()
        tau  = TAU_MIN + (TAU_MAX - TAU_MIN) * _sigmoid(raw_tau)
        bias = BIAS_RANGE * np.tanh(raw_bias)
        eta  = ETA_MAX * _sigmoid(raw_eta).reshape(N_RECURRENT, N_TOTAL)

        return {"W": W, "tau": tau, "bias": bias, "eta": eta}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate(self) -> "Genome":
        """Returns a new mutated Genome. Does not modify self."""
        new_data = self.data.copy()
        mask  = np.random.random(GENOME_SIZE) < MUTATION_RATE
        noise = np.random.normal(0.0, MUTATION_SIGMA, GENOME_SIZE).astype(np.float32)
        new_data += mask * noise
        return Genome(new_data)

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"Genome(mean={self.data.mean():.4f}, std={self.data.std():.4f})"
