"""
uv run python -m experiment.analyze_results --in experiment/experiments/<timestamp>
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: write a file, don't open a window
import matplotlib.pyplot as plt

from scipy import stats


PRETTY = {"crossover": "Crossover + mutation", "mutation_only": "Mutation only"}


def load_per_generation(path):
    data = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data[row["condition"]][int(row["generation"])].append(
                float(row["best_fitness"])
            )
    return data


def load_final(path):
    data = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data[row["condition"]].append(float(row["final_best_fitness"]))
    return {k: np.array(v) for k, v in data.items()}


def plot_learning_curves(per_gen, out_path):
    plt.figure(figsize=(7, 4.5))
    for condition, gen_map in per_gen.items():
        gens = sorted(gen_map.keys())
        means = np.array([np.mean(gen_map[g]) for g in gens])
        stds = np.array([np.std(gen_map[g], ddof=1) for g in gens])
        plt.plot(gens, means, label=PRETTY.get(condition, condition))
        plt.fill_between(gens, means - stds, means + stds, alpha=0.2)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness in population (mean over runs)")
    plt.title("Mean best-fitness per generation (band = $\\pm$1 SD across runs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved figure: {out_path}")


def describe(name, x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return (
        f"  {PRETTY.get(name, name):>22}: "
        f"n={len(x)}  mean={x.mean():.2f}  sd={x.std(ddof=1):.2f}  "
        f"median={med:.2f}  IQR=[{q1:.2f}, {q3:.2f}]"
    )


def analyse_final(final, out_path):
    a = final["crossover"]
    b = final["mutation_only"]

    lines = []
    lines.append("=== FINAL FITNESS: DESCRIPTIVE STATISTICS ===")
    lines.append(describe("crossover", a))
    lines.append(describe("mutation_only", b))

    # normality (justifies the non-parametric choice)
    lines.append("\n=== NORMALITY (Shapiro-Wilk) ===")
    for name, x in (("crossover", a), ("mutation_only", b)):
        if len(x) >= 3:
            w, p = stats.shapiro(x)
            lines.append(f"  {PRETTY[name]:>22}: W={w:.3f}, p={p:.4f}")

    u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    # rank-biserial effect size from U
    rank_biserial = 1.0 - (2.0 * u) / (n1 * n2)
    lines.append("\n=== PRIMARY TEST: Mann-Whitney U (two-sided) ===")
    lines.append(f"  U = {u:.1f}")
    lines.append(f"  p = {p_u:.4g}")
    lines.append(f"  rank-biserial effect size = {rank_biserial:.3f}")
    lines.append(
        f"  significant at alpha=0.05? {'YES, reject H0' if p_u < 0.05 else 'NO, fail to reject H0'}"
    )

    t, p_t = stats.ttest_ind(a, b, equal_var=False)  # Welch
    lines.append("\n=== ALTERNATIVE: Welch's independent t-test ===")
    lines.append(f"  t = {t:.3f}, p = {p_t:.4g}")

    text = "\n".join(lines)
    print("\n" + text)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved stats summary: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="indir", required=True)
    args = parser.parse_args()

    per_gen = load_per_generation(
        os.path.join(args.indir, "results_per_generation.csv")
    )
    final = load_final(os.path.join(args.indir, "results_final.csv"))

    plot_learning_curves(per_gen, os.path.join(args.indir, "learning_curves.png"))
    analyse_final(final, os.path.join(args.indir, "stats_summary.txt"))


if __name__ == "__main__":
    main()
