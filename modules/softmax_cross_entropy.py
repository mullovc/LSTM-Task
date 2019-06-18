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
        $ dL(softmax(x))/dx = softmax(x) - t $

        where $x$ is the input, $t$ is the target and $L$ is the cross entropy
        loss function. $t$ is a one-hot vector given as index of the one-hot
        dimension.
        '''
        dLdx = np.divide(self.exp_x, self.sum_exp_x)
        dLdx[self.t] -= 1
        return dLdx
