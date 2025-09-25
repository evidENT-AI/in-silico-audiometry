"""Constants and default values for audiometry testing."""

# Maximum output levels for different conduction types
AIR_CONDUCTION_MAX_LEVELS = {
    250: 105, 500: 110, 750: 110,
    1000: 120, 1500: 120, 2000: 120, 3000: 120,
    4000: 120, 6000: 110, 8000: 105
}

BONE_CONDUCTION_MAX_LEVELS = {
    250: 25, 500: 55, 750: 55,
    1000: 70, 1500: 70, 2000: 70, 
    3000: 70, 4000: 70
}

# Default testing parameters
DEFAULT_TEST_FREQUENCIES = [1000, 2000, 4000, 8000, 500, 250]
DEFAULT_STARTING_LEVEL = 40
DEFAULT_EXPERT_STARTING_LEVEL = 60
MIN_TEST_LEVEL = -10
MAX_ITERATIONS = 50
EXPERT_MAX_ITERATIONS = 20

# Response model default parameters
DEFAULT_SLOPE = 0.2
DEFAULT_GUESS_RATE = 0.01
DEFAULT_LAPSE_RATE = 0.01