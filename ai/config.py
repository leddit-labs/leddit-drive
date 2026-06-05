import os

#for saving data
DATA_DIR = "ai/training_data"
TRAIN_LOG_PATH = os.path.join(DATA_DIR, "training_log.csv")
BEST_MODEL_PATH = os.path.join(DATA_DIR, "best_genome.npy")
SAVE_FILE = f"{DATA_DIR}/best_genome.npy"

#stats for GA
POPULATION_SIZE = 25
GENERATIONS = 30
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.5
ELITE_COUNT = 3
MAX_STEPS = 1000            # so the agent don't live for ever if never crash
CHECKPOINT_TIMEOUT = 250    # so the agent don't just stand still.

