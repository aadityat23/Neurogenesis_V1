# environment/volatility.py
import numpy as np
from config import TRANSITION_STEPS


class VolatilityScheduler:
    """
    Manages resource grid reshuffling based on volatility σ.

    Every σ steps, a new target resource grid is generated.
    Over the next TRANSITION_STEPS, the current grid linearly interpolates
    toward the target. After transition, current = target.

    This is called once per simulation step with the global step count.
    """

    def __init__(self, volatility: int, grid):
        """
        Args:
            volatility: σ value (steps between reshuffles)
            grid: Grid instance (holds resource_grid)
        """
        self.volatility = volatility
        self.grid = grid

        # Interpolation state
        self._in_transition  = False
        self._transition_t   = 0          # steps into current transition
        self._source_grid    = None       # grid at start of transition
        self._target_grid    = None       # grid we're interpolating toward
        self._target_centers = None

    # ------------------------------------------------------------------

    def step(self, global_step: int):
        """
        Called once per simulation step.

        Args:
            global_step: absolute step count across the entire experiment
        """
        # Trigger a new transition every `volatility` steps
        # (but not at step 0, and not if already in a transition)
        if global_step > 0 and global_step % self.volatility == 0 and not self._in_transition:
            self._start_transition()

        if self._in_transition:
            self._advance_transition()

    # ------------------------------------------------------------------

    def _start_transition(self):
        """Begin interpolating toward a freshly generated resource grid."""
        self._source_grid   = self.grid.resource_grid.copy()
        new_centers, new_grid = self.grid.new_resource_grid()
        self._target_grid    = new_grid
        self._target_centers = new_centers
        self._transition_t   = 0
        self._in_transition  = True

    def _advance_transition(self):
        """Linearly interpolate one step toward target grid."""
        self._transition_t += 1
        alpha = self._transition_t / TRANSITION_STEPS   # 0 → 1

        self.grid.resource_grid = (
            (1.0 - alpha) * self._source_grid + alpha * self._target_grid
        ).astype(np.float32)

        if self._transition_t >= TRANSITION_STEPS:
            # Snap to target; update centers
            self.grid.resource_grid      = self._target_grid.copy()
            self.grid._resource_centers  = self._target_centers
            self._in_transition          = False
            self._source_grid            = None
            self._target_grid            = None

    # ------------------------------------------------------------------

    @property
    def in_transition(self) -> bool:
        return self._in_transition

    def __repr__(self):
        return f"VolatilityScheduler(σ={self.volatility}, in_transition={self._in_transition})"
