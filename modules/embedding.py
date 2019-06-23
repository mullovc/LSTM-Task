import numpy as np
from modules.module import Module
from modules.LSTM import LSTM

class Embedding(Module):
    def __init__(self, input_size, embedding_size):
        super(Embedding, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size

        std = 1.0 / np.sqrt(input_size + embedding_size)

        self.W = np.random.normal(0, std, [input_size, embedding_size])

    def forward(self, x):
        shape = x.shape + (self.embedding_size,)
        x_flat = x.flatten()

        out = self.W[x_flat]

        self.x = (x_flat,)
        return out.reshape(shape)

    def backward(self, dLdOut):
        x, = self.x
        # flatten all extra dimensions into one batch dimension
        dLdOut2D = dLdOut.reshape([-1, dLdOut.shape[-1]])
        for x_i, dOut in zip(x, dLdOut2D):
            self.gradients["W"][x_i] += dOut

        # doesn't return input gradient
