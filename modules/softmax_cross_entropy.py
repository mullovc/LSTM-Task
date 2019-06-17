import numpy as np
from modules.module import Module
from modules.LSTM import LSTM

class Softmax(Module):
    def __init__(self, dim=1):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        exp_x = np.exp(x)
        return np.divide(exp_x, np.sum(exp_x, self.dim, keepdims=True))

    def backward(self, dLdOut):
        raise NotImplementedError

class SoftmaxCrossEntropy(Module):
    def __init__(self, dim=1):
        super(SoftmaxCrossEntropy, self).__init__()
        self.dim = dim

    def forward(self, x, t):
        self.t = t
        self.exp_x = np.exp(x)
        self.sum_exp_x = np.sum(self.exp_x, self.dim, keepdims=True)
        return -x[t] + np.log(self.sum_exp_x)

    def backward(self):
        '''
        $ dL(x)/dx = -t * softmax(x) $

        where $x$ is the input, $t$ is the target and $L$ is the cross entropy
        loss function. Assuming that $t$ is a one-hot vector (given as index
        for the one-hot dimension), the gradient becomes a vector with
        zero-entries in all but the one-hot dimension.
        '''
        sm = np.divide(self.exp_x, self.sum_exp_x)
        dLdx = np.zeros_like(self.exp_x)
        dLdx[self.t] = -sm[self.t]
        return dLdx
