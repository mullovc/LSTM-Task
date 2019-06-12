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

        # gates = []
        hiddens = []
        activations = []
        for i in range(seq_len):
            x_t = X[:, i, :].transpose()

            h_n, c_n, act_t = self.lstm_layer(x_t, (h_n, c_n))

            # gates.append(c_n)
            hiddens.append(h_n)
            activations.append(act_t)

        # gates = np.stack(gates).transpose([2, 0, 1])
        hiddens = np.stack(hiddens).transpose([2, 0, 1])
        # activations = np.stack(activations).transpose([2, 0, 1])

        return hiddens, (h_n, c_n), activations

    def backward(self, activations, dLdOut):
        '''
        Differentiate error w.r.t. weights and w.r.t. input.
        Steps:
            1. differentiate outputs w.r.t. weights and inputs
                - delegate gradient computation to LSTMLayer
            2. apply chain rule: accumulate "outer" gradient
        '''
        dLdOut_t = (dLdOut, 0)
        dW_ih = []
        dW_hh = []
        for act_t in activations[::-1]:
            dLdOut_t, (dW_iht, dW_hht) = self.lstm_layer.backward(act_t, dLdOut_t)

            dW_ih.append(dW_iht)
            dW_hh.append(dW_hht)

        dW_ih = np.mean(dW_ih, 0)
        dW_hh = np.mean(dW_hh, 0)

        return dW_ih, dW_hh
