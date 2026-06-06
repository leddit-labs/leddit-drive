import csv
import numpy as np

from ai.config import (
    BEST_MODEL_PATH,
    GENERATIONS,
    MAX_STEPS,
    CHECKPOINT_TIMEOUT,
    SAVE_FILE,
    TRAIN_LOG_PATH,
)

from ai.genetic_algorithm import GeneticAlgorithm
from environment.world import World


def evaluate_agent(agent, verbose=False, agent_id=None):
    world = World()

    state = world.get_state()

    total_reward = 0
    steps_since_checkpoint = 0
    checkpoints_hit = 0

    last_reward = 0

    for step in range(MAX_STEPS):
        action = agent.act(state)                   #the agents infer
        state, reward, done = world.step(action)    #run a game tick. get the state, reward and if agent crashes

        total_reward += reward

        if reward > last_reward:
            steps_since_checkpoint = 0
            checkpoints_hit += 1
        else:
            steps_since_checkpoint += 1

        last_reward = reward

        if done:
            break

        if steps_since_checkpoint > CHECKPOINT_TIMEOUT:
            print(f"agent: {agent_id} timed out")
            break

        if verbose and agent_id is not None and step % 50 == 0:
            print(
                f"    [Agent {agent_id}] step={step} "
                f"reward={reward:.2f} "
                f"total={total_reward:.2f}"
            )

    agent.fitness = total_reward

    if verbose:
        print(
            f"    FINISHED | steps={step} "
            f"checkpoints={checkpoints_hit} "
            f"fitness={agent.fitness:.2f}"
        )


def train():
    ga = GeneticAlgorithm()

    with open(TRAIN_LOG_PATH, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["generation", "best_fitness", "mean_fitness"])

        for generation in range(GENERATIONS):
            print(f"GENERATION {generation + 1}/{GENERATIONS}")

            # --- evaluate population ---
            for i, agent in enumerate(ga.population):
                evaluate_agent(agent, verbose=True, agent_id=i)

            # --- stats ---
            best_agent = ga.get_best_agent()
            mean_fitness = np.mean([a.fitness for a in ga.population])

            print("\n--- GENERATION RESULT ---")
            print(f"Best fitness: {best_agent.fitness:.2f}")
            print(f"Mean fitness: {mean_fitness:.2f}")

            # --- log ---
            writer.writerow([generation, best_agent.fitness, mean_fitness])
            file.flush()

            # --- save best genome ---
            np.save(BEST_MODEL_PATH, best_agent.genome)

            print(f"Saved best genome to {SAVE_FILE}.npy")

            # --- evolve ---
            ga.evolve()

    print("\nTRAINING COMPLETE")

if __name__ == "__main__":
    train()