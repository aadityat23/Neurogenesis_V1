# agent/ctrnn.py
import numpy as np
from config import N_TOTAL, N_SENSORY, N_RECURRENT, DT, WEIGHT_NORM_CAP


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class CTRNN:
    """
    Continuous-Time RNN, Euler-discretized at dt=0.1.

    Architecture:
        - 8 sensory neurons (pure pass-through, no dynamics)
        - 6 hidden neurons  (recurrent)
        - 2 motor neurons   (recurrent)

    Only hidden+motor neurons (indices 8-15) have dynamics.
    Sensory neurons (indices 0-7) are just filled with sensor values
    each step and used as input to the recurrent layer.

    Neuron activations y: shape (16,)
        y[0:8]  = sensory  (written from outside each step)
        y[8:14] = hidden   (CTRNN dynamics)
        y[14:16]= motor    (CTRNN dynamics)
    """

    def __init__(self, decoded_genome: dict):
        """
        Args:
            decoded_genome: dict with keys W, tau, bias, eta
                W   : (8, 16) initial weight matrix
                tau : (8,)    time constants for recurrent neurons
                bias: (8,)    biases for recurrent neurons
                eta : (8, 16) per-synapse Hebbian rates
        """
        self.W_init = decoded_genome["W"].copy()   # birth weights (immutable)
        self.W      = decoded_genome["W"].copy()   # live weights (mutated by Hebbian)
        self.tau    = decoded_genome["tau"]         # shape (8,)
        self.bias   = decoded_genome["bias"]        # shape (8,)

        # Neuron state
        self.y = np.full(N_TOTAL, 0.5, dtype=np.float32)

    # ------------------------------------------------------------------

    def reset(self):
        """Reset to birth state (called at start of each lifetime)."""
        self.y[:] = 0.5
        self.W[:] = self.W_init   # reset to genome weights, not learned weights

    # ------------------------------------------------------------------

    def step(self, sensors: np.ndarray) -> np.ndarray:
        """
        One CTRNN step.

        Args:
            sensors: (8,) float32 sensor values

        Returns:
            motor: (2,) float32 motor outputs = y[14:16]
        """
        # Write sensory inputs
        self.y[:N_SENSORY] = sensors

        # Compute net input for recurrent neurons (hidden + motor = indices 8-15)
        # net_i = bias_i + sum_j(W_ij * y_j)
        # W shape: (8, 16), y shape: (16,) → net shape: (8,)
        net = self.bias + self.W @ self.y   # (8,)

        # Euler update: y_i += (dt / tau_i) * (-y_i + sigmoid(net_i))
        recurrent = self.y[N_SENSORY:]      # view into y[8:16]
        dy = (DT / self.tau) * (-recurrent + _sigmoid(net))
        self.y[N_SENSORY:] = np.clip(recurrent + dy, 0.0, 1.0)

        return self.y[N_SENSORY + 6:].copy()   # motor outputs = y[14:16]

    # ------------------------------------------------------------------

    def apply_hebbian(self, eta: np.ndarray):
        """
        BCM-style covariance Hebbian update:
            ΔW_ij = η_ij * (y_i - 0.5) * (y_j - 0.5)

        Only updates W for recurrent neurons (post-synaptic = hidden+motor).

        Args:
            eta: (8, 16) per-synapse learning rates
        """
        post = self.y[N_SENSORY:].reshape(-1, 1)   # (8, 1)
        pre  = self.y.reshape(1, -1)               # (1, 16)
        dW   = eta * (post - 0.5) * (pre - 0.5)
        self.W += dW

        # L2-norm clipping per row to prevent runaway weights
        norms = np.linalg.norm(self.W, axis=1, keepdims=True)   # (8, 1)
        too_large = (norms > WEIGHT_NORM_CAP).flatten()
        self.W[too_large] *= WEIGHT_NORM_CAP / norms[too_large]

    # ------------------------------------------------------------------

    def weight_delta_frobenius(self) -> float:
        """||W_current - W_init||_F — Baldwin assimilation metric."""
        return float(np.linalg.norm(self.W - self.W_init, "fro"))

    def __repr__(self):
        return (f"CTRNN(tau_mean={self.tau.mean():.2f}, "
                f"W_norm={np.linalg.norm(self.W):.2f})")
