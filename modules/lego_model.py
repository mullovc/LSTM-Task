import numpy as np
from modules.module import Module
from modules.LSTM import LSTM
from modules.linear import Linear
from modules.sigmoid import Sigmoid
from modules.softmax_cross_entropy import Softmax

class LegoModel(Module):
    def __init__(self, input_size, lstm_size, output_size):
        super(LegoModel, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.output_size = output_size

        self.lstm = LSTM(input_size, lstm_size)
        self.out = Linear(lstm_size, output_size)
        # self.f = Sigmoid()
        self.f = Softmax()

    def forward(self, X):
        seq_len    = X.shape[0]
        batch_size = X.shape[1]

        h_0 = np.random.randn(batch_size, self.lstm_size).astype(np.float32)
        c_0 = np.random.randn(batch_size, self.lstm_size).astype(np.float32)

        lstm_out, _ = self.lstm(X, (h_0, c_0))

        # out = self.out(lstm_out)
        out = self.out(lstm_out[-1])

        return out


    def backward(self, dLdOut):
        # dLdIn = self.f.backward(dLdOut)
        # dLdIn = self.out.backward(dLdIn)
        dLdIn = self.out.backward(dLdOut)
        dLdIn = self.lstm.backward(dLdIn)
        return dLdIn
