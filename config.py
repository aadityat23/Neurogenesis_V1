# config.py — Single source of truth for all NeuroGenesis constants
import numpy as np

# Grid
GRID_SIZE           = 100

# Resources
N_PATCHES           = 12
PATCH_PEAK          = 1.0
PATCH_SIGMA         = 6.0
RESOURCE_REGEN      = 0.02
RESOURCE_CAP        = 1.0
DEPLETION_RATE      = 0.4

# Hazards
N_HAZARDS           = 5
HAZARD_SIGMA        = 4.0
HAZARD_DAMAGE       = 0.3
HAZARD_MIN_DIST     = 15     # min cells from any resource center

# Agent lifetime
LIFETIME            = 200
TRANSITION_STEPS    = 20

# Population
POPULATION_SIZE     = 150
N_GENERATIONS       = 1000
ELITE_FRACTION      = 0.1
TOURNAMENT_K        = 5

# Mutation
MUTATION_RATE       = 0.02
MUTATION_SIGMA      = 0.15

# CTRNN
N_SENSORY           = 8
N_HIDDEN            = 6
N_MOTOR             = 2
N_TOTAL             = 16     # N_SENSORY + N_HIDDEN + N_MOTOR
N_RECURRENT        = 8      # N_HIDDEN + N_MOTOR (neurons with dynamics)
DT                  = 0.1
TAU_MIN             = 0.5
TAU_MAX             = 5.0
BIAS_RANGE          = 3.0
ETA_MAX             = 0.1
WEIGHT_NORM_CAP     = 4.0

# Energy
ENERGY_INIT         = 1.0
ENERGY_MAX          = 3.0
ENERGY_STEP         = 0.01

# Volatility sweep (your x-axis)
VOLATILITY_VALUES   = [50, 100, 200, 400, 800]

# Genome layout (flattened 272-element array)
GENOME_W_START      = 0
GENOME_W_END        = 128    # 8 * 16 = 128
GENOME_TAU_START    = 128
GENOME_TAU_END      = 136
GENOME_BIAS_START   = 136
GENOME_BIAS_END     = 144
GENOME_ETA_START    = 144
GENOME_ETA_END      = 272    # 8 * 16 = 128
GENOME_SIZE         = 272

# Direction map: 0=N, 1=E, 2=S, 3=W  (dx, dy in grid coords)
DIRECTION_MAP = {
    0: (0, -1),   # North: y decreases
    1: (1,  0),   # East:  x increases
    2: (0,  1),   # South: y increases
    3: (-1, 0),   # West:  x decreases
}

# Conditions
class Condition:
    EVO_ONLY   = "EVO_ONLY"
    LEARN_ONLY = "LEARN_ONLY"
    EVO_LEARN  = "EVO_LEARN"
    BASELINE   = "BASELINE"
    ALL        = [EVO_ONLY, LEARN_ONLY, EVO_LEARN, BASELINE]

# Logging
LOG_AGENTS_EVERY    = 10     # sample detailed agent data every N generations
LOG_WEIGHTS_EVERY   = 50     # save weight snapshots every N generations
N_AGENTS_SAMPLE     = 10     # agents sampled per condition per log event
