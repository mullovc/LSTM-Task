import numpy as np
from modules.module import Module
from modules.lstm_layer import LSTMLayer

class LSTM(Module):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_layer = LSTMLayer(input_size, hidden_size)

    def forward(self, X, (h_n, c_n)):
        seq_len = X.shape[1]
        hiddens = []
        gates   = []
        for i in range(seq_len):
            x_t = X[:, i, :].transpose()

            h_n, c_n = self.lstm_layer(x_t, (h_n, c_n))

            hiddens.append(h_n)
            gates.append(c_n)

        self.gates = np.stack(gates).transpose([2, 0, 1])
        self.hiddens = np.stack(hiddens).transpose([2, 0, 1])

        return self.hiddens, (h_n, c_n)

    def backward(self, out):
        '''
        Differentiate error w.r.t. weights and w.r.t. input.
        Steps:
            1. differentiate outputs w.r.t. weights and inputs
                - delegate gradient computation to LSTMLayer
            2. apply chain rule: accumulate "outer" gradient
        '''
        raise NotImplementedError
