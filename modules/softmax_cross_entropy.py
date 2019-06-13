import numpy as np
from modules.module import Module
from modules.LSTM import LSTM

class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        o = np.exp(x)
        out = np.divide(o, np.sum(o, 1))
        return out

    def backward(self, dLdOut):
        raise NotImplementedError

class SoftmaxCrossEntropy(Module):
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, t):
        self.t = t
        self.exp_x = np.exp(x)
        self.sum_exp_x = np.sum(self.exp_x, 1)
        return -x[:,t] + np.log(self.sum_exp_x)

    def backward(self):
        sm = np.divide(self.exp_x, self.sum_exp_x)
        dLdx = np.zeros_like(self.exp_x)
        dLdx[:,self.t] = -sm[:,self.t]
        return dLdx
