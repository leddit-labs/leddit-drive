"""
Ablation experiment harness for the crossover-vs-mutation-only hypothesis.

Runs BOTH conditions across many seeded, independent runs while holding
everything else constant, and writes two tidy CSV files that the analysis
script (game/analyze_results.py) consumes.

Usage:
    uv run python -m game.run_experiment                       # all cores
    uv run python -m game.run_experiment --workers 1           # serial
    uv run python -m game.run_experiment --runs 3 --generations 8   # quick pilot

The only thing that differs between the two conditions is `use_crossover`
(the independent variable). Population size, generations, network, selection,
mutation rate/strength, elitism, track and evaluation budget are identical.

Runs are independent and embarrassingly parallel; --workers spreads them across
CPU cores. Because every run reseeds np.random itself, results are identical
regardless of the number of workers or completion order.
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


# ---- experiment configuration (edit these for the real run) ----------------
N_RUNS_PER_CONDITION = 30   # independent runs per condition (>=20 recommended)
N_GENERATIONS = 20          # generations per run

# Distinct, non-overlapping seed blocks => the two groups are independent.
SEED_BASE = {
    "crossover": 1000,       # seeds 1000, 1001, ...
    "mutation_only": 2000,   # seeds 2000, 2001, ...
}

CONDITIONS = {
    "crossover": True,        # Condition A: crossover + mutation
    "mutation_only": False,   # Condition B: mutation only
}
# ----------------------------------------------------------------------------


def run_single(condition_name, use_crossover, seed, n_generations):
    """One independent run. Returns (per_gen_rows, final_row)."""
    # Reseeding at the START of every run makes each run fully reproducible
    # and independent of execution order (and of worker count). Evaluation is
    # deterministic, so the seed fixes the entire run.
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
    """Top-level worker wrapper so it is picklable for multiprocessing."""
    return run_single(*job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=N_RUNS_PER_CONDITION)
    parser.add_argument("--generations", type=int, default=N_GENERATIONS)
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
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

    # build the full job list: both conditions x all runs
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

    # collect; sort deterministically so the CSVs are identical regardless of
    # worker count or completion order
    all_gen_rows = []
    final_rows = []
    for gen_rows, final_row in results:
        all_gen_rows.extend(gen_rows)
        final_rows.append(final_row)

    all_gen_rows.sort(key=lambda r: (r["condition"], r["seed"], r["generation"]))
    final_rows.sort(key=lambda r: (r["condition"], r["seed"]))

    for r in final_rows:
        print(f"  {r['condition']:>14} seed {r['seed']}: final_best={r['final_best_fitness']:.2f}")

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
            f, fieldnames=["condition", "seed", "generation", "best_fitness", "mean_fitness"]
        )
        writer.writeheader()
        writer.writerows(all_gen_rows)

    with open(final_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "seed", "final_best_fitness"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\nWrote:\n  {per_gen_path}\n  {final_path}\n  {meta_path}")


if __name__ == "__main__":
    main()
