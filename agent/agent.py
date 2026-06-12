# agent/agent.py
import numpy as np
from config import (
    GRID_SIZE, DIRECTION_MAP, ENERGY_INIT, ENERGY_MAX, ENERGY_STEP,
    HAZARD_DAMAGE, LIFETIME, Condition, N_SENSORY,
)
from agent.genome import Genome
from agent.ctrnn import CTRNN


class Agent:
    """
    An individual agent in the simulation.

    Owns a Genome and a CTRNN. Tracks position, energy, age, fitness.
    Does NOT know about population structure or generation number.
    """

    _id_counter = 0

    def __init__(self, genome: Genome, condition: str, agent_id: int = None):
        self.genome    = genome
        self.condition = condition
        self.agent_id  = agent_id if agent_id is not None else Agent._id_counter
        Agent._id_counter += 1

        # Build CTRNN from decoded genome
        decoded = genome.decode()

        # Cache decoded parameters (avoid repeated decoding)
        self.decoded = decoded
        self.eta = decoded["eta"]
        self.tau = decoded["tau"]

        self.ctrnn = CTRNN(decoded)

        # Runtime state (reset each generation)
        self.x      = 0
        self.y_pos  = 0
        self.energy = ENERGY_INIT
        self.age    = 0
        self.alive  = True
        self.fitness_accumulator = 0.0

        # Birth-weight snapshot for Baldwin tracking
        self._birth_W = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, grid_size: int = GRID_SIZE):
        """Reset to birth state. Called at start of every generation."""
        self.ctrnn.reset()
        self.x      = np.random.randint(0, grid_size)
        self.y_pos  = np.random.randint(0, grid_size)
        self.energy = ENERGY_INIT
        self.age    = 0
        self.alive  = True
        self.fitness_accumulator = 0.0
        self._birth_W = self.ctrnn.W.copy()

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self, grid):
        """
        One simulation step:
            sense → think → move → eat → costs → learn → age
        """
        if not self.alive:
            return

        # 1. Sense
        sensors = grid.get_sensors(self.x, self.y_pos)

        # 2. Think
        motor = self.ctrnn.step(sensors)    # (2,)

        # 3. Move
        direction = int(motor[0] * 4)
        direction = min(direction, 3)       # guard against motor[0] == 1.0
        dx, dy = DIRECTION_MAP[direction]
        self.x      = (self.x     + dx) % GRID_SIZE
        self.y_pos  = (self.y_pos + dy) % GRID_SIZE

        # 4. Eat
        if motor[1] > 0.5:
            energy_gained = grid.consume(self.x, self.y_pos)
            self.energy += energy_gained

        # 5. Environmental costs
        self.energy -= grid.get_hazard_damage(self.x, self.y_pos) * HAZARD_DAMAGE
        self.energy -= ENERGY_STEP
        self.energy  = float(np.clip(self.energy, 0.0, ENERGY_MAX))

        # 6. Hebbian learning (only in LEARN conditions)
        if self.condition in (Condition.EVO_LEARN, Condition.LEARN_ONLY):
            self.ctrnn.apply_hebbian(self.eta)

        # 7. Accumulate fitness and age
        self.fitness_accumulator += self.energy
        self.age += 1

        if self.energy <= 0.0:
            self.alive = False

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    @property
    def fitness(self) -> float:
        if self.age == 0:
            return 0.0
        return self.fitness_accumulator / self.age

    def is_alive(self) -> bool:
        return self.alive and self.energy > 0.0 and self.age < LIFETIME

    # ------------------------------------------------------------------
    # Baldwin tracking helpers
    # ------------------------------------------------------------------

    def birth_weight_norm(self) -> float:
        if self._birth_W is None:
            return 0.0
        return float(np.linalg.norm(self._birth_W))

    def death_weight_norm(self) -> float:
        return float(np.linalg.norm(self.ctrnn.W))

    def weight_delta_frobenius(self) -> float:
        return self.ctrnn.weight_delta_frobenius()

    def mean_eta(self) -> float:
        return float(self.eta.mean())

    def max_eta(self) -> float:
        return float(self.eta.max())

    def min_eta(self) -> float:
        return float(self.eta.min())

    def std_eta(self) -> float:
        return float(self.eta.std())

    def mean_tau(self) -> float:
        return float(self.tau.mean())

    def __repr__(self):
        return (f"Agent({self.condition}, id={self.agent_id}, "
                f"fitness={self.fitness:.3f}, age={self.age}, alive={self.alive})")