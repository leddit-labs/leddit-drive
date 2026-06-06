import numpy as np

from ai.agent import Agent

from ai.config import (
    POPULATION_SIZE,
    MUTATION_RATE,
    MUTATION_STRENGTH,
    ELITE_COUNT,
    USE_ELITISM,
    USE_CROSSOVER,
)


class GeneticAlgorithm:
    def __init__(self):
        self.population = [Agent() for _ in range(POPULATION_SIZE)]

        self.generation = 0

    def evolve(self):
        self.population.sort(
            key=lambda agent: agent.fitness,
            reverse=True,
        )

        next_population = []

        if USE_ELITISM: # flag defined in config.py
            elites = self.population[:ELITE_COUNT]

            for elite in elites:
                copied_agent = Agent(np.copy(elite.genome))

                next_population.append(copied_agent)

        while len(next_population) < POPULATION_SIZE:
            parent1 = self.select_parent()

            if USE_CROSSOVER: # flag defined in config.py
                parent2 = self.select_parent()

                child_genome = self.crossover(
                    parent1.genome,
                    parent2.genome,
                )

            # mutation-only evolution - no crossover
            else:
                child_genome = np.copy(parent1.genome)

            # mutation happens anyway
            child_genome = self.mutate(child_genome)

            next_population.append(Agent(child_genome))

        self.population = next_population

        self.generation += 1

    def select_parent(self):
        # tournament selection
        tournament = np.random.choice(
            self.population,
            5,
            replace=False,
        )

        tournament = sorted(
            tournament,
            key=lambda agent: agent.fitness,
            reverse=True,
        )

        return tournament[0]

    def crossover(self, genome1, genome2):
        # mask is a random boolean array where each index decides
        # whether the child inherits the gene from parent1 or parent2
        mask = np.random.rand(len(genome1)) < 0.5
        child = np.where(
            mask,
            genome1,
            genome2,
        )

        return child

    def mutate(self, genome):
        for i in range(len(genome)):
            if np.random.rand() < MUTATION_RATE:
                genome[i] += np.random.randn() * MUTATION_STRENGTH

        return genome

    def get_best_agent(self):
        self.population.sort(
            key=lambda agent: agent.fitness,
            reverse=True,
        )

        return self.population[0]
