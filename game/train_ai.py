import csv
import os
import shutil

import numpy as np

from ai.config import (
    GENERATIONS,
    MAX_STEPS,
    CHECKPOINT_TIMEOUT,
    TRAIN_LOG_PATH,
    GENOME_DIR,
    RUN_DIR,
)

from ai.genetic_algorithm import GeneticAlgorithm
from environment.world import World


def evaluate_agent(agent, verbose=False, agent_id=None):
    world = World()

    state = world.get_state()

    total_reward = 0

    steps_since_checkpoint = 0
    checkpoints_hit = 0

    for step in range(MAX_STEPS):
        action = agent.act(state)

        state, reward, done, checkpoint_hit = world.step(action)

        total_reward += reward

        # checkpoint progress
        if checkpoint_hit:
            checkpoints_hit += 1
            steps_since_checkpoint = 0
        else:
            steps_since_checkpoint += 1

        # crashed
        if done:
            if verbose:
                print(f"Agent {agent_id} crashed")
            break

        # stuck
        if steps_since_checkpoint > CHECKPOINT_TIMEOUT:
            if verbose:
                print(f"Agent {agent_id} timed out")
            break

        # debug print every 50 ticks
        if verbose and step % 50 == 0:
            print(
                f"[Agent {agent_id}] "
                f"step={step} "
                f"reward={reward:.2f} "
                f"fitness={total_reward:.2f} "
                f"checkpoints={checkpoints_hit}"
            )

    agent.fitness = total_reward

    if verbose:
        print(
            f"FINISHED | "
            f"steps={step} | "
            f"checkpoints={checkpoints_hit} | "
            f"fitness={agent.fitness:.2f}"
        )

#take snapshot of config file. luksus
def save_config_snapshot():
    shutil.copy(
        "ai/config.py",
        os.path.join(RUN_DIR, "config_snapshot.txt"),
    )


def train(verbose=False):
    os.makedirs(GENOME_DIR, exist_ok=True)
    save_config_snapshot()

    ga = GeneticAlgorithm()

    previous_best_genome = None

    with open(TRAIN_LOG_PATH, "w", newline="") as file:
        writer = csv.writer(file)

        #init the .csv
        writer.writerow(
            [
                "generation",
                "best_fitness",
                "mean_fitness",
            ]
        )

        for generation in range(GENERATIONS):
            print(f"\n=== GENERATION {generation}/{GENERATIONS} ===")

            # evaluate a agent at a time
            for i, agent in enumerate(ga.population):
                evaluate_agent(
                    agent,
                    verbose=verbose,
                    agent_id=i,
                )

            # get the best agent. used for best fitness and elitism
            best_agent = ga.get_best_agent()

            #calculate the mean_fitness. average fitness for all agents in population
            mean_fitness = np.mean([agent.fitness for agent in ga.population])

            print("\n--- GENERATION RESULT ---")
            print(f"Best fitness: {best_agent.fitness:.2f}")
            print(f"Mean fitness: {mean_fitness:.2f}")

            #append this generation stats to .csv file
            writer.writerow(
                [
                    generation,
                    best_agent.fitness,
                    mean_fitness,
                ]
            )

            file.flush()

            # check if the new best genome/agent is the same as last generation. if yes - change the naming of that genome
            is_duplicate = False

            if (
                previous_best_genome is not None
            ):  # not None since first generation has no prev
                is_duplicate = np.array_equal(
                    previous_best_genome,
                    best_agent.genome,
                )

            duplicate_tag = "_DUPLICATE" if is_duplicate else ""

            genome_path = os.path.join(
                GENOME_DIR,
                f"gen_{generation:03d}_best{duplicate_tag}.npy",  # append the duplicate_tag to file name only if duplicate is true
            )

            np.save(
                genome_path,
                best_agent.genome,
            )

            # save a new previous best, that will be used to compare with next generation
            previous_best_genome = best_agent.genome.copy()

            print(f"Saved genome: {genome_path}")

            ga.evolve()

    print("\nTRAINING COMPLETE")


if __name__ == "__main__":
    train(True)
