"""
Reframed analysis for the crossover-vs-mutation experiment.

The original analysis compared the MEAN final fitness, but that metric is
dominated by a small failure tail: the distribution is bimodal (a tight
"solved the track" cluster around ~5100, plus a few runs that never converge).
Comparing means therefore mostly measures the failure rate, while the ceiling
hides any difference in solution quality.

This script analyses the two phenomena separately:
  1. SUCCESS RATE  -> did a run converge at all?  (Fisher's exact test)
  2. SOLUTION QUALITY among successes -> Mann-Whitney U (ceiling check)
  3. PAIRED TEST   -> Wilcoxon signed-rank, valid only if the two conditions
                      were run with MATCHED seeds (see run_experiment patch).

Usage:
    uv run python -m game.analyze_results_reframed --in ai/experiments/<timestamp>
    uv run python -m game.analyze_results_reframed --in <dir> --success-threshold 4500
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from scipy import stats

PRETTY = {"crossover": "Crossover + mutation", "mutation_only": "Mutation only"}


def load_final(path):
    """Return {condition: [(seed, fitness, completed_or_None), ...]} preserving order."""
    data = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            # 'completed' is optional: present only if run_experiment recorded it.
            completed = row.get("completed")
            completed = None if completed in (None, "") else completed.strip().lower() in ("1", "true", "yes")
            data[row["condition"]].append(
                (int(row["seed"]), float(row["final_best_fitness"]), completed)
            )
    return data


def classify_success(records, threshold):
    """Return a boolean array: did each run converge?

    Prefers a recorded 'completed' flag; falls back to a fitness threshold.
    The fitness gap between the failure tail and the converged cluster is wide,
    so the exact threshold does not matter (anything in ~3600..4900 is identical).
    """
    have_flag = all(c is not None for _, _, c in records)
    if have_flag:
        return np.array([c for _, _, c in records], dtype=bool)
    return np.array([fit >= threshold for _, fit, _ in records], dtype=bool)


def analyse(final, threshold, out_path):
    lines = []
    a = final["crossover"]
    b = final["mutation_only"]
    fa = np.array([fit for _, fit, _ in a])
    fb = np.array([fit for _, fit, _ in b])
    sa = classify_success(a, threshold)
    sb = classify_success(b, threshold)

    # ---- 1. SUCCESS RATE -------------------------------------------------
    cross_succ, cross_fail = int(sa.sum()), int((~sa).sum())
    mut_succ, mut_fail = int(sb.sum()), int((~sb).sum())
    table = [[cross_succ, cross_fail], [mut_succ, mut_fail]]
    odds, p_fisher = stats.fisher_exact(table, alternative="two-sided")

    lines.append("=== 1. CONVERGENCE / SUCCESS RATE ===")
    lines.append(f"  {PRETTY['crossover']:>22}: {cross_succ}/{len(a)} converged "
                 f"({100*cross_succ/len(a):.0f}%)")
    lines.append(f"  {PRETTY['mutation_only']:>22}: {mut_succ}/{len(b)} converged "
                 f"({100*mut_succ/len(b):.0f}%)")
    lines.append(f"  Fisher's exact (two-sided): odds ratio={odds:.3f}, p={p_fisher:.4f}")
    lines.append(f"  significant at alpha=0.05? "
                 f"{'YES' if p_fisher < 0.05 else 'NO, fail to reject H0'}")

    # ---- 2. SOLUTION QUALITY AMONG SUCCESSES (ceiling check) -------------
    qa, qb = fa[sa], fb[sb]
    lines.append("\n=== 2. SOLUTION QUALITY (converged runs only) ===")
    lines.append(f"  {PRETTY['crossover']:>22}: n={len(qa)}  "
                 f"mean={qa.mean():.1f}  sd={qa.std(ddof=1):.1f}  "
                 f"range=[{qa.min():.0f}, {qa.max():.0f}]")
    lines.append(f"  {PRETTY['mutation_only']:>22}: n={len(qb)}  "
                 f"mean={qb.mean():.1f}  sd={qb.std(ddof=1):.1f}  "
                 f"range=[{qb.min():.0f}, {qb.max():.0f}]")
    if len(qa) >= 1 and len(qb) >= 1:
        u, p_u = stats.mannwhitneyu(qa, qb, alternative="two-sided")
        lines.append(f"  Mann-Whitney U (two-sided): U={u:.1f}, p={p_u:.4f}")
        lines.append("  -> A tight cluster + non-significant p indicates a fitness "
                     "ceiling:\n     successful runs of both methods saturate at the "
                     "same value.")

    # ---- 3. PAIRED TEST (only valid with matched seeds) ------------------
    lines.append("\n=== 3. PAIRED TEST (Wilcoxon signed-rank) ===")
    seeds_a = [s for s, _, _ in a]
    seeds_b = [s for s, _, _ in b]
    matched = (len(seeds_a) == len(seeds_b)) and (sorted(seeds_a) == sorted(seeds_b))
    if not matched:
        lines.append("  SKIPPED: conditions used different seeds, so runs are not "
                     "paired.\n  Re-run with matched seeds (see run_experiment patch) "
                     "to enable this\n  more powerful test.")
    else:
        # pair by shared seed
        map_b = {s: fit for s, fit, _ in b}
        pairs = [(fit, map_b[s]) for s, fit, _ in a]
        da = np.array([x for x, _ in pairs])
        db = np.array([y for _, y in pairs])
        diff = da - db
        try:
            w, p_w = stats.wilcoxon(da, db)
            lines.append(f"  n pairs = {len(pairs)}")
            lines.append(f"  median paired diff (crossover - mutation) = "
                         f"{np.median(diff):.1f}")
            lines.append(f"  Wilcoxon W={w:.1f}, p={p_w:.4f}")
            lines.append(f"  significant at alpha=0.05? "
                         f"{'YES' if p_w < 0.05 else 'NO, fail to reject H0'}")
        except ValueError as e:
            lines.append(f"  Could not run Wilcoxon: {e}")

    text = "\n".join(lines)
    print("\n" + text)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="indir", required=True)
    parser.add_argument("--success-threshold", type=float, default=4500.0,
                        help="fitness above which a run counts as converged "
                             "(ignored if a 'completed' column is present)")
    args = parser.parse_args()

    final = load_final(os.path.join(args.indir, "results_final.csv"))
    analyse(final, args.success_threshold,
            os.path.join(args.indir, "stats_summary_reframed.txt"))


if __name__ == "__main__":
    main()
