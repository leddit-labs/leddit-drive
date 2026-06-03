import numpy as np

class Agent:
    def __init__(self, w1=None, w2=None):
        self.input_size = 7
        self.hidden = 6
        self.output = 2

        self.w1 = w1 if w1 is not None else np.random.randn(self.input_size, self.hidden)
        self.w2 = w2 if w2 is not None else np.random.randn(self.hidden, self.output)

    def forward(self, x):
        x = np.tanh(np.dot(x, self.w1))
        x = np.tanh(np.dot(x, self.w2))
        return x

    def mutate(self, rate=0.1):
        self.w1 += np.random.randn(*self.w1.shape) * rate
        self.w2 += np.random.randn(*self.w2.shape) * rate

    def copy(self):
        return Agent(self.w1.copy(), self.w2.copy())