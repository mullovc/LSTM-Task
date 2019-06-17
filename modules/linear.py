import numpy as np
from modules.module import Module
from modules.LSTM import LSTM

class Linear(Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        std = 1.0 / np.sqrt(input_size + output_size)

        self.W = np.random.normal(0, std, [input_size, output_size])

    def forward(self, x):
        self.x = x
        out = np.matmul(self.x, self.W)
        return out

    def backward(self, dLdOut):
        # flatten all extra dimensions into one batch dimension
        x2D = self.x.reshape([-1,self.x.shape[-1]])
        dLdOut2D = dLdOut.reshape([-1,dLdOut.shape[-1]])

        self.gradients["W"] += np.matmul(x2D.transpose(), dLdOut2D)
        dLdIn = np.matmul(dLdOut, self.W.transpose())
        return dLdIn
