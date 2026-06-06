import numpy as np

INPUT_SIZE = 4
HIDDEN_SIZE = 6
OUTPUT_SIZE = 2

class NeuralNetwork:
    def __init__(self, genome):
        self.genome = genome

        self.w1 = genome[:INPUT_SIZE * HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
        self.w2 = genome[INPUT_SIZE * HIDDEN_SIZE:].reshape(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, inputs):
        x = np.array(inputs)

        hidden = np.tanh(x @ self.w1)

        output = np.tanh(hidden @ self.w2)

        steer = output[0]
        throttle = output[1]

        return steer, throttle

    @staticmethod
    def genome_size():
        return (INPUT_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE)
