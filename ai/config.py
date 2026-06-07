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

MAP_SIZE = (1600, 900)

#best genome - used by game/watch_best_ai.py
# TODO: should be refactored to load all the genomes in a given folder, then visually show them
# TODO: this is very hardcoded - probably only works for me right now
BEST_GENOME = "ai/training_data/07-06_13_11/genomes/gen_019_best.npy"

# the range of car sensors
# a shorter range = faster training, better performance MAYBE worse car???
# a longer range = slower training, worse performance MAYBE better car???
# 250 seems like a good value - more testing needed
SENSOR_RANGE_PIXELS = 300

#defines the amount of laps required and bonus for finising them without crashing
TOTAL_AMOUNT_OF_LAPS = 5
FITNESS_BONUS_FOR_COMPLETING_AMOUNT_OF_LAPS = 1000

#calculation looks like:
# reward -= DEFAULT_NEGATIVE_REWARD
DEFAULT_NEGATIVE_REWARD = 0.15

#this is multiplied with car.speed calculation is like this in world.py - step()
#reward += self.car.speed * BONUS_REWARD_SPEED_MULTIPLIER
BONUS_REWARD_SPEED_MULTIPLIER = 0.1 #0.1 was the original

# stats for GA
POPULATION_SIZE = 15
GENERATIONS = 20
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.4
ELITE_COUNT = 3
MAX_STEPS = 4000            # so the agent don't live for ever if never crash
CHECKPOINT_TIMEOUT = 100    # so the agent don't just stand still. it needs to hit those checkpoints fast

USE_ELITISM = True
USE_CROSSOVER = True