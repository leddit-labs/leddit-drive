# Leddit Drive — Genetic Algorithm Racing AI

Leddit Drive is a small 2D racing game built with Python and visualized with pygame where AI agents learn to drive using a Genetic Algorithm (GA).

The video below shows how the agents evolve and improve over time across 20 generations of training.


[![YouTube Video](https://img.youtube.com/vi/OBVZv0vsnD4/hqdefault.jpg)](https://youtu.be/OBVZv0vsnD4)

The project explores the following research question:

> Does crossover actually improve learning in a Genetic Algorithm, or is mutation alone enough?

Each AI agent is a small feed-forward neural network that receives sensor data and the car speed and outputs steering and throttle controls. The neural network weights form the genome evolved over generations by the GA.

The repository contains:

* A pygame racing game
* AI-controlled cars
* Genetic Algorithm training
* Experiment tooling
* Track editor
* AI playback tools
* Data export utilities for analysis

---
# Table of Contents

- [Installation & Setup](#installation--setup)
- [Running the Project](#running-the-project)
  - [Play the Game (Human)](#play-the-game-human)
  - [Watch the Best AI](#watch-the-best-ai)
  - [Train the AI](#train-the-ai)
  - [Open Track Editor](#open-track-editor)
- [Running Experiments](#running-experiments)
- [Make Commands](#make-commands)

# Installation & Setup

This project uses `uv` for dependency management.

Clone the repository:

```bash
git clone https://github.com/leddit-labs/leddit-drive
```

Install dependencies:

```bash
uv sync
```

This creates the virtual environment and installs all dependencies.

---

# Running the Project





## Play the Game (Human)
Launches the playable racing game.


```bash
make human-up
```

OR

```bash
uv run python -m game.human_play -v
```

---
## Watch the Best AI
Loads and visualizes the best trained AI agent



```bash
make watch-up
```

OR

```bash
uv run python -m game.watch_best_ai -v
```


---
## Train the AI
Runs a simple Genetic Algorithm training.

Training details and settings are defined in config.py


```bash
make train-up
```

OR

```bash
uv run python -m game.train_ai --verbose
```

---


# Running Experiments

Example experiment:

```bash
uv run python -m game.run_experiment --runs 3 --generations 8
```

This runs:

* 3 independent experiment runs
* 8 generations each
* Multiple GA configurations


Optional flags
| Flag | Description | Default | Example |
| ---- | ----------- | ------- | ------- |
| `--runs` | Number of independent runs per condition | `30` | `--runs 10` |
| `--generations` | Number of generations per run | `20` | `--generations 50` |
| `--populations` | Population size for the GA | `15` | `--populations 30` |
| `--workers` | Number of parallel CPU processes | `os.cpu_count()` | `--workers 4` |
| `--out` | Output folder for experiment results | timestamped folder | `--out ai/experiments/test` |

Generated files:

* `experiment_meta.txt`
* `results_final.csv`
* `results_per_generation.csv`


---

## Open Track Editor
Opens the track creation/editor tool.

```bash
make track-up
```

OR

```bash
uv run python -m game.track_making -v
```
---


# Make Commands

| Command         | Description                 |
| --------------- | --------------------------- |
| `make human-up` | Launch the playable game    |
| `make track-up` | Open the track editor       |
| `make train-up` | Train AI agents with the GA |
| `make watch-up` | Watch the best trained AI   |

---

