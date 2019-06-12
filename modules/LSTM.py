import numpy as np
from modules.module import Module
from modules.lstm_layer import LSTMLayer

class LSTM(Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_layer = LSTMLayer(input_size, hidden_size)

    def forward(self, X, (h_n, c_n)):
        seq_len = X.shape[1]

        hiddens = []
        self.activations = []
        for i in range(seq_len):
            x_t = X[:, i, :].transpose()

            h_n, c_n, act_t = self.lstm_layer(x_t, (h_n, c_n))

            hiddens.append(h_n)
            self.activations.append(act_t)

        hiddens = np.stack(hiddens).transpose([2, 0, 1])

        return hiddens, (h_n, c_n)

    def backward(self, dLdOut):
        '''
        Differentiate error w.r.t. weights and w.r.t. input.
        Steps:
            1. differentiate outputs w.r.t. weights and inputs
                - delegate gradient computation to LSTMLayer
            2. apply chain rule: accumulate "outer" gradient
        '''
        dLdOut_t = (dLdOut.transpose(), 0)
        for act_t in self.activations[::-1]:
            dLdOut_t = self.lstm_layer.backward(act_t, dLdOut_t)

    def apply_gradient(self, learning_rate):
        self.lstm_layer.apply_gradient(learning_rate)

    def zero_grad(self):
        self.lstm_layer.zero_grad()
