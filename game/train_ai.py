import argparse
import csv
import os
import shutil
from multiprocessing import Pool

import numpy as np

from ai.config import (
    FITNESS_BONUS_FOR_COMPLETING_AMOUNT_OF_LAPS,
    GENERATIONS,
    MAX_STEPS,
    CHECKPOINT_TIMEOUT,
    TOTAL_AMOUNT_OF_LAPS,
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
    total_laps_completed = 0

    step = 0  # guard in case MAX_STEPS is 0, so the prints below never NameError

    for step in range(MAX_STEPS):
        action = agent.act(state)

        state, reward, done, checkpoint_hit, lap_completed = world.step(action)

        total_reward += reward

        # checkpoint progress
        if checkpoint_hit:
            checkpoints_hit += 1
            steps_since_checkpoint = 0
        else:
            steps_since_checkpoint += 1

        if lap_completed:
            total_laps_completed += 1
            if total_laps_completed >= TOTAL_AMOUNT_OF_LAPS:
                print(
                    f"Agent {agent_id} completed {TOTAL_AMOUNT_OF_LAPS} without crashing"
                )
                # was: total_reward += FITNESS_BONUS_FOR_COMPLETING_AMOUNT_OF_LAPS
                speed_factor = (
                    MAX_STEPS - step
                ) / MAX_STEPS  # in [0,1], higher = faster
                total_reward += FITNESS_BONUS_FOR_COMPLETING_AMOUNT_OF_LAPS * (
                    1 + speed_factor
                )
                break

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

    return total_reward


def _eval_seed(base_seed, generation, agent_index):
    return int((base_seed * 1_000_003 + generation * 1009 + agent_index) % (2**31 - 1))


def _evaluate_worker(job):
    agent, eval_seed = job
    if eval_seed is not None:
        np.random.seed(eval_seed)
    return evaluate_agent(agent, verbose=False)


# take snapshot of config file. luksus
def save_config_snapshot():
    shutil.copy(
        "ai/config.py",
        os.path.join(RUN_DIR, "config_snapshot.txt"),
    )


def _evaluate_population(ga, pool, generation, seed, verbose):
    if pool is None:
        for i, agent in enumerate(ga.population):
            if seed is not None:
                rng_state = np.random.get_state()
                np.random.seed(_eval_seed(seed, generation, i))

            evaluate_agent(agent, verbose=verbose, agent_id=i)

            if seed is not None:
                np.random.set_state(rng_state)
        return

    # ---- parallel path ----
    jobs = [
        (
            agent,
            _eval_seed(seed, generation, i) if seed is not None else None,
        )
        for i, agent in enumerate(ga.population)
    ]
    results = pool.map(_evaluate_worker, jobs)

    # pool.map preserves order, so zip back onto the parent's population objects.
    for agent, fitness in zip(ga.population, results):
        agent.fitness = fitness


def train(verbose=False, workers=12, seed=None):
    os.makedirs(GENOME_DIR, exist_ok=True)
    save_config_snapshot()

    if seed is not None:
        np.random.seed(seed)
        print(f"Seed: {seed}")

    print(f"Workers: {workers}")

    ga = GeneticAlgorithm()

    previous_best_genome = None

    # one persistent pool for the whole run (cheaper than re-spawning each gen)
    pool = Pool(workers) if workers and workers > 1 else None

    try:
        with open(TRAIN_LOG_PATH, "w", newline="") as file:
            writer = csv.writer(file)

            # init the .csv
            writer.writerow(
                [
                    "generation",
                    "best_fitness",
                    "mean_fitness",
                ]
            )

            for generation in range(GENERATIONS):
                print(f"\n=== GENERATION {generation}/{GENERATIONS} ===")

                # evaluate the whole population (serial or across workers)
                _evaluate_population(ga, pool, generation, seed, verbose)

                # get the best agent. used for best fitness and elitism
                best_agent = ga.get_best_agent()

                # calculate the mean_fitness. average fitness for all agents in population
                mean_fitness = np.mean([agent.fitness for agent in ga.population])

                print("\n--- GENERATION RESULT ---")
                print(f"Best fitness: {best_agent.fitness:.2f}")
                print(f"Mean fitness: {mean_fitness:.2f}")

                # append this generation stats to .csv file
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

                population_genomes = np.array([a.genome for a in ga.population])
                np.save(
                    os.path.join(GENOME_DIR, f"gen_{generation:03d}_population.npy"),
                    population_genomes,
                )

                # save a new previous best, that will be used to compare with next generation
                previous_best_genome = best_agent.genome.copy()

                print(f"Saved genome: {genome_path}")

                ga.evolve()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print("\nTRAINING COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="parallel processes for agent evaluation (1 = serial). Default: 12.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for reproducible runs. Omit for a random run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="per-agent debug prints (only meaningful with --workers 1).",
    )
    args = parser.parse_args()

    train(verbose=args.verbose, workers=args.workers, seed=args.seed)
