import numpy as np

from util.util import sigmoid
from numpy import tanh

class Module(object):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        pass

    def backward(self, *args, **kwargs):
        pass

class LSTM(Module):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_hh = np.random.randn(4*hidden_size, hidden_size).astype(np.float32)
        self.W_ih = np.random.randn(4*hidden_size, input_size).astype(np.float32)

    def forward(self, X, (h_n, c_n)):
        seq_len = X.shape[1]
        hiddens = []
        gates   = []
        for i in range(seq_len):
            x_t = X[:, i, :].transpose()
            y_t = np.dot(self.W_ih, x_t) + np.dot(self.W_hh, h_n)
            #i_t, f_t, g_t, o_t = y_t.chunk(4)
            i_t, f_t, g_t, o_t = y_t.reshape([4, self.hidden_size, -1])

            i_t = sigmoid(i_t)
            f_t = sigmoid(f_t)
            g_t = tanh(g_t)
            o_t = sigmoid(o_t)

            c_n = f_t * c_n + i_t * g_t
            h_n = o_t * tanh(c_n)

            hiddens.append(h_n)
            gates.append(c_n)

        self.gates = np.stack(gates).transpose([2, 0, 1])
        self.hiddens = np.stack(hiddens).transpose([2, 0, 1])

        return self.hiddens, (h_n, c_n)

    def backward(self, out):
        '''
        Differentiate error w.r.t. weights and w.r.t. input.
        Steps for each layer:
            1. differentiate outputs w.r.t. weights and inputs
            2. apply chain rule: accumulate "outer" gradient
        '''
        raise NotImplementedError
