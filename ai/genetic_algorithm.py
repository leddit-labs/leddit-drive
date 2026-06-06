import numpy as np

from ai.agent import Agent
from ai.config import (
    POPULATION_SIZE,
    MUTATION_RATE,
    MUTATION_STRENGTH,
    ELITE_COUNT,
)


class GeneticAlgorithm:
    def __init__(self):
        self.population = [Agent() for _ in range(POPULATION_SIZE)]

        self.generation = 0

    def evolve(self):
        # sort highest fitness first
        self.population.sort(
            key=lambda agent: agent.fitness,
            reverse=True,
        )

        next_population = []

        # elitism
        elites = self.population[:ELITE_COUNT]

        for elite in elites:
            copied = Agent(np.copy(elite.genome))
            next_population.append(copied)

        # fill remaining population
        while len(next_population) < POPULATION_SIZE:
            parent1 = self.select_parent()
            parent2 = self.select_parent()

            child_genome = self.crossover(
                parent1.genome,
                parent2.genome,
            )

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
            key=lambda a: a.fitness,
            reverse=True,
        )

        return tournament[0]

    def crossover(self, genome1, genome2):
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
