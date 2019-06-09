import numpy as np
from util.util import sigmoid
from modules.module import Module
from numpy import tanh

class LSTMLayer(Module):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_hh = np.random.randn(4*hidden_size, hidden_size).astype(np.float32)
        self.W_ih = np.random.randn(4*hidden_size, input_size).astype(np.float32)

    def forward(self, x_t, (h_n, c_n)):
        y_t = np.dot(self.W_ih, x_t) + np.dot(self.W_hh, h_n)
        #i_t, f_t, g_t, o_t = y_t.chunk(4)
        i_t, f_t, g_t, o_t = y_t.reshape([4, self.hidden_size, -1])

        i_t = sigmoid(i_t)
        f_t = sigmoid(f_t)
        g_t = tanh(g_t)
        o_t = sigmoid(o_t)

        c_n = f_t * c_n + i_t * g_t
        h_n = o_t * tanh(c_n)

        return h_n, c_n

    def backward(self, out):
        '''
        Differentiate layer output w.r.t. weights and w.r.t. input.
        '''
        raise NotImplementedError
