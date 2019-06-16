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
        '''
        Does the LSTM forward for a sequence `X`.

        Arguments:
        X: numpy array of shape (seq_len x batch_size x input_size)
            The input sequence.
        h_n: numpy array of shape (batch_size x hidden_size)
            The initial hidden state.
        c_n: numpy array of shape (batch_size x hidden_size)
            The initial cell state.

        Returns: output, (h_n, c_n)
        output: numpy array of shape (seq_len x batch_size x hidden_size)
            Stack of LSTM outputs (hidden states) at each timestep.
        h_n: numpy array of shape (batch_size x hidden_size)
            Final hidden state after consuming the input sequence.
        c_n: numpy array of shape (batch_size x hidden_size)
            Final cell state after consuming the input sequence.
        '''
        seq_len = X.shape[0]

        hiddens = []
        self.activations = []
        for t in range(seq_len):
            h_n, c_n, act_t = self.lstm_layer(X[t], (h_n, c_n))

            hiddens.append(h_n)
            self.activations.append(act_t)

        return np.stack(hiddens), (h_n, c_n)

    def backward(self, dLdOut):
        '''
        Differentiate error w.r.t. weights and w.r.t. input.
        Steps:
            1. differentiate outputs w.r.t. weights and inputs
                - delegate gradient computation to LSTMLayer
            2. apply chain rule: accumulate "outer" gradient

        Arguments:
        dLdOut: numpy array of shape (batch_size x hidden_size)
            Gradient w.r.t. the LSTM output.

        Returns: dLdX
        dLdX: list of numpy arrays of shape (batch_size x hidden_size)
            Gradients for each timestep w.r.t. the LSTM inputs.
        '''
        dLdOut_t = (dLdOut, 0)
        dLdX = []
        for act_t in self.activations[::-1]:
            dLdx_t, dLdOut_t = self.lstm_layer.backward(act_t, dLdOut_t)
            dLdX.append(dLdx_t)
        return dLdX
