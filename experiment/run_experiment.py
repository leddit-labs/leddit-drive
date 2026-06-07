"""
Usage:
    uv run python -m game.run_experiment # all cores
    uv run python -m game.run_experiment --workers 1
    uv run python -m game.run_experiment --runs 3 --generations 8

Use --workers to set workers
"""

import argparse
import csv
import os
from datetime import datetime
from multiprocessing import Pool

import numpy as np

from ai.config import POPULATION_SIZE, ELITE_COUNT, MUTATION_RATE, MUTATION_STRENGTH
from ai.genetic_algorithm import GeneticAlgorithm
from game.train_ai import evaluate_agent  # reuse the EXACT fitness under test


N_RUNS_PER_CONDITION = 30
N_GENERATIONS = 20

SEED_BASE = {
    "crossover": 1000,
    "mutation_only": 2000,
}

CONDITIONS = {
    "crossover": True,
    "mutation_only": False,
}


def run_single(condition_name, use_crossover, seed, n_generations):
    np.random.seed(seed)

    ga = GeneticAlgorithm(use_crossover=use_crossover, use_elitism=True)

    per_gen_rows = []
    final_best = None

    for generation in range(n_generations):
        for agent in ga.population:
            evaluate_agent(agent, verbose=False)

        fitnesses = [a.fitness for a in ga.population]
        best = float(np.max(fitnesses))
        mean = float(np.mean(fitnesses))

        per_gen_rows.append(
            {
                "condition": condition_name,
                "seed": seed,
                "generation": generation,
                "best_fitness": best,
                "mean_fitness": mean,
            }
        )

        final_best = best  # best in the population at the (current) final gen

        if generation < n_generations - 1:
            ga.evolve()

    final_row = {
        "condition": condition_name,
        "seed": seed,
        "final_best_fitness": final_best,
    }
    return per_gen_rows, final_row


def _run_job(job):
    return run_single(*job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=N_RUNS_PER_CONDITION)
    parser.add_argument("--generations", type=int, default=N_GENERATIONS)
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="parallel processes (1 = serial). Default: all cores.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(
            "ai", "experiments", datetime.now().strftime("%d-%m_%H_%M")
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    jobs = []
    for condition_name, use_crossover in CONDITIONS.items():
        for run_idx in range(args.runs):
            seed = SEED_BASE[condition_name] + run_idx
            jobs.append((condition_name, use_crossover, seed, args.generations))

    print(
        f"Running {len(jobs)} jobs "
        f"({args.runs} runs x {len(CONDITIONS)} conditions) on {args.workers} worker(s)..."
    )

    if args.workers == 1:
        results = [_run_job(j) for j in jobs]
    else:
        with Pool(args.workers) as pool:
            results = pool.map(_run_job, jobs)

    all_gen_rows = []
    final_rows = []
    for gen_rows, final_row in results:
        all_gen_rows.extend(gen_rows)
        final_rows.append(final_row)

    all_gen_rows.sort(key=lambda r: (r["condition"], r["seed"], r["generation"]))
    final_rows.sort(key=lambda r: (r["condition"], r["seed"]))

    for r in final_rows:
        print(
            f"  {r['condition']:>14} seed {r['seed']}: final_best={r['final_best_fitness']:.2f}"
        )

    per_gen_path = os.path.join(args.out, "results_per_generation.csv")
    final_path = os.path.join(args.out, "results_final.csv")
    meta_path = os.path.join(args.out, "experiment_meta.txt")

    with open(meta_path, "w") as f:
        f.write(f"runs_per_condition={args.runs}\n")
        f.write(f"generations={args.generations}\n")
        f.write(f"population_size={POPULATION_SIZE}\n")
        f.write(f"elite_count={ELITE_COUNT}\n")
        f.write(f"mutation_rate={MUTATION_RATE}\n")
        f.write(f"mutation_strength={MUTATION_STRENGTH}\n")
        f.write(f"seed_base={SEED_BASE}\n")

    with open(per_gen_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "seed",
                "generation",
                "best_fitness",
                "mean_fitness",
            ],
        )
        writer.writeheader()
        writer.writerows(all_gen_rows)

    with open(final_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["condition", "seed", "final_best_fitness"]
        )
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\nWrote:\n  {per_gen_path}\n  {final_path}\n  {meta_path}")


if __name__ == "__main__":
    main()
