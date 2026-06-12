# environment/grid.py
import numpy as np
from scipy.ndimage import gaussian_filter
from config import (
    GRID_SIZE, N_PATCHES, PATCH_SIGMA, RESOURCE_REGEN, RESOURCE_CAP,
    DEPLETION_RATE, N_HAZARDS, HAZARD_SIGMA, HAZARD_MIN_DIST,
)


class Grid:
    """
    100×100 toroidal grid with resource and hazard layers.

    Resource and hazard grids are float32 arrays of shape (100, 100).
    Agents do NOT live here — the grid only tracks environmental state.
    """

    def __init__(self):
        self.resource_grid  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.hazard_grid    = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self._resource_centers = []   # [(x, y), ...]
        self._hazard_centers   = []

        self._initialize()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self):
        self._resource_centers = self._place_patches(N_PATCHES, [])
        self.resource_grid = self._build_gaussian_grid(self._resource_centers, PATCH_SIGMA)

        self._hazard_centers = self._place_patches(
            N_HAZARDS, self._resource_centers, min_dist=HAZARD_MIN_DIST
        )
        self.hazard_grid = self._build_gaussian_grid(self._hazard_centers, HAZARD_SIGMA)

    def _place_patches(self, n: int, exclusion_centers: list, min_dist: float = 0.0) -> list:
        """Place n patch centers, avoiding being too close to exclusion_centers."""
        centers = []
        max_attempts = 10_000
        attempts = 0
        while len(centers) < n and attempts < max_attempts:
            attempts += 1
            cx = np.random.randint(0, GRID_SIZE)
            cy = np.random.randint(0, GRID_SIZE)
            if min_dist > 0:
                too_close = any(
                    self._toroidal_dist(cx, cy, ex, ey) < min_dist
                    for (ex, ey) in exclusion_centers
                )
                if too_close:
                    continue
            centers.append((cx, cy))
        if len(centers) < n:
            raise RuntimeError(f"Could only place {len(centers)}/{n} patches after {max_attempts} attempts")
        return centers

    @staticmethod
    def _toroidal_dist(x1, y1, x2, y2) -> float:
        dx = min(abs(x1 - x2), GRID_SIZE - abs(x1 - x2))
        dy = min(abs(y1 - y2), GRID_SIZE - abs(y1 - y2))
        return np.sqrt(dx**2 + dy**2)

    def _build_gaussian_grid(self, centers: list, sigma: float) -> np.ndarray:
        """
        Build a (100,100) grid as sum of Gaussians centered at `centers`.
        Uses scipy gaussian_filter on a spike grid — fast and clean.
        """
        spike_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for (cx, cy) in centers:
            spike_grid[cy, cx] += 1.0   # note: grid[row=y, col=x]

        # gaussian_filter handles boundary by mode; 'wrap' for toroidal
        blurred = gaussian_filter(spike_grid, sigma=sigma, mode="wrap")

        # Normalize so peak ≈ 1.0
        if blurred.max() > 0:
            blurred = blurred / blurred.max()

        return blurred.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API called by Grid's consumers (VolatilityScheduler, Agent)
    # ------------------------------------------------------------------

    def new_resource_grid(self) -> tuple:
        """Generate a new random resource grid. Returns (centers, grid)."""
        centers = self._place_patches(N_PATCHES, self._hazard_centers, min_dist=0.0)
        grid    = self._build_gaussian_grid(centers, PATCH_SIGMA)
        return centers, grid

    def apply_regeneration(self):
        """Add RESOURCE_REGEN to all non-zero cells, cap at RESOURCE_CAP."""
        mask = self.resource_grid > 0.01
        self.resource_grid[mask] += RESOURCE_REGEN
        np.clip(self.resource_grid, 0.0, RESOURCE_CAP, out=self.resource_grid)

    # ------------------------------------------------------------------
    # Sensor / interaction API
    # ------------------------------------------------------------------

    def get_sensors(self, x: int, y: int) -> np.ndarray:
        """
        Returns 8-element sensor array for agent at (x, y):
            [res_N, res_S, res_E, res_W, haz_N, haz_S, haz_E, haz_W]
        """
        n = (y - 1) % GRID_SIZE
        s = (y + 1) % GRID_SIZE
        e = (x + 1) % GRID_SIZE
        w = (x - 1) % GRID_SIZE

        return np.array([
            self.resource_grid[n, x],
            self.resource_grid[s, x],
            self.resource_grid[y, e],
            self.resource_grid[y, w],
            self.hazard_grid[n, x],
            self.hazard_grid[s, x],
            self.hazard_grid[y, e],
            self.hazard_grid[y, w],
        ], dtype=np.float32)

    def consume(self, x: int, y: int) -> float:
        """Agent eats at (x, y). Returns energy gained."""
        current = float(self.resource_grid[y, x])
        if current < 1e-6:
            return 0.0
        consumed = DEPLETION_RATE * current
        self.resource_grid[y, x] = max(0.0, current - consumed)
        return consumed

    def get_hazard_damage(self, x: int, y: int) -> float:
        """Returns hazard value at (x, y) ∈ [0, 1]."""
        return float(self.hazard_grid[y, x])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def resource_total(self) -> float:
        return float(self.resource_grid.sum())

    def __repr__(self):
        return (f"Grid(resource_total={self.resource_total():.1f}, "
                f"n_resource_centers={len(self._resource_centers)}, "
                f"n_hazard_centers={len(self._hazard_centers)})")
