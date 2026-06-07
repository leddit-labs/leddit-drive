# Leddit Drive Project Idea

Create a car game - probably using Pygame.
Use AI to drive the car successfully through the one track.

## Problem Statement

- How can we utilize Genetic Algorithms to navigate a given racing track 1 standard deviation
  better than average human players?
  - What selection... Mutation ... Crossover
    (Make one combined sub-question for finetunning)

## HOW TO GO VROOOM

Use this command to run the experiment with 8 generations for 3 different seeds (runs) for mutation-only and 3 for crossover+mutation.

```bash
uv run python -m game.run_experiment --runs 3 --generations 8 --out ai/experiments/pilot
```

Generates:

- experiment_meta.txt
- results_final.csv
- results_per_generation.csv

The files are saved in /ai/experiments/pilot/
