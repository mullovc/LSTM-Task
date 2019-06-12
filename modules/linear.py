import numpy as np
from modules.module import Module
from modules.LSTM import LSTM
from util.util import sigmoid

class Linear(Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        std = 1.0 / np.sqrt(input_size + output_size)

        self.W = np.random.normal(0, std, [input_size, output_size])

        self.parameters = { id(self.W) : self.W }

    def forward(self, x):
        self.x = x
        out = np.matmul(self.x, self.W)
        return out

    def backward(self, dLdOut):
        # self.dLdW = np.dot(dLdOut.transpose(), self.x)
        # self.gradients[id(self.W)] += np.dot(dLdOut.transpose(), self.x)
        self.gradients[id(self.W)] += np.dot(self.x.transpose(), dLdOut)
        dLdIn = np.dot(dLdOut, self.W.transpose())
        return dLdIn
