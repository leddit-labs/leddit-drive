import numpy as np


class NeuralNetwork:
    def __init__(self, genome):
        self.genome = genome

        self.w1 = genome[:24].reshape(4, 6)

        self.w2 = genome[24:36].reshape(6, 2)

    def forward(self, inputs):
        x = np.array(inputs)

        hidden = np.tanh(x @ self.w1)

        output = np.tanh(hidden @ self.w2)

        steer = output[0]
        throttle = output[1]

        return steer, throttle

    @staticmethod
    def genome_size():
        return (4 * 6) + (6 * 2)
