import numpy as np

class GA:
    def __init__(self, size=30):
        self.size = size

    def evolve(self, agents, fitnesses):
        ranked = [x for _, x in sorted(zip(fitnesses, agents), reverse=True)]

        new_agents = ranked[:5]  # elitism

        while len(new_agents) < self.size:
            parent = np.random.choice(ranked[:10])
            child = parent.copy()
            child.mutate(0.05)
            new_agents.append(child)

        return new_agents