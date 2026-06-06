import numpy as np

from ai.network import NeuralNetwork


class Agent:
    def __init__(self, genome=None):
        if genome is None:
            genome = np.random.randn(
                NeuralNetwork.genome_size()
            )

        self.genome = genome

        self.fitness = 0

        self.network = NeuralNetwork(self.genome)

    def act(self, state):
        return self.network.forward(state)