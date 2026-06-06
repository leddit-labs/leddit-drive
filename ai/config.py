from datetime import datetime
import os


TIMESTAMP = datetime.now().strftime(("%d-%m_%H_%M"))

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

# stats for GA
POPULATION_SIZE = 20
GENERATIONS = 10
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.5
ELITE_COUNT = 1
MAX_STEPS = 1250  # so the agent don't live for ever if never crash
CHECKPOINT_TIMEOUT = 250  # so the agent don't just stand still.
