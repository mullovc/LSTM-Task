import numpy as np
from modules.module import Module
from modules.LSTM import LSTM
from util.util import sigmoid

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dLdOut):
        dLdIn = dLdOut * self.out * (1 - self.out)
        return dLdIn
