from datetime import datetime
import os

TIMESTAMP = datetime.now().strftime(("%d-%m_%H_%M")) #DAY_MONTH_HOUR_MINUTE

RUN_DIR = os.path.join(
    "ai",
    "training_data",
    TIMESTAMP,
)

GENOME_DIR = os.path.join(RUN_DIR, "genomes")

TRAIN_LOG_PATH = os.path.join(
    RUN_DIR,
    "training_log.csv",
)

#best genome - used by game/watch_best_ai.py
# TODO: should be refactored to load all the genomes in a given folder, then visually show them
# TODO: this is very hardcoded - probably only works for me right now
BEST_GENOME = "ai/training_data/06-06_16_28/genomes/gen_019_best.npy"

# the range of car sensors
# a shorter range = faster training, better performance MAYBE worse car???
# a longer range = slower training, worse performance MAYBE better car???
# 250 seems like a good value - more testing needed
SENSOR_RANGE_PIXELS = 250

# stats for GA
POPULATION_SIZE = 20
GENERATIONS = 25
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.5
ELITE_COUNT = 1
MAX_STEPS = 1250  # so the agent don't live for ever if never crash
CHECKPOINT_TIMEOUT = 250  # so the agent don't just stand still.

USE_ELITISM = True
USE_CROSSOVER = True