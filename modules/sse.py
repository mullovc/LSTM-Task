import numpy as np
from modules.module import Module
from modules.LSTM import LSTM
from util.util import sigmoid

class SSE(Module):
    def __init__(self):
        super(SSE, self).__init__()

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = np.divide(np.sum(np.sqare(self.x - self.y), 1), 2)
        return out

    def backward(self, dLdOut):
        dLdIn = self.x - self.y
        return dLdIn
