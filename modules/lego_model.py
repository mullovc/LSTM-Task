import numpy as np
from modules.module import Module
from modules.LSTM import LSTM
from modules.linear import Linear
from modules.sigmoid import Sigmoid
from modules.sse import SSE

class LegoModel(Module):
    def __init__(self, input_size, lstm_size, output_size):
        super(LegoModel, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.output_size = output_size

        self.lstm = LSTM(input_size, lstm_size)
        self.out = Linear(lstm_size, output_size)
        self.f = Sigmoid()

    def forward(self, X):
        batch_size = X.shape[0]
        seq_len = X.shape[1]

        h_0 = np.random.randn(self.lstm_size, batch_size).astype(np.float32)
        c_0 = np.random.randn(self.lstm_size, batch_size).astype(np.float32)

        lstm_out, _ = self.lstm(X, (h_0, c_0))

        # out = self.f(self.out(lstm_out))
        out = self.f(self.out(lstm_out[:,-1,:]))

        return out

    #def backward(self, dLdOut):
    #    # lstm_activations, lstm_out, out_activation = activations
    #    lstm_out = lstm_out[0]

    #    # dLdLSTM = np.dot(self.W_out.transpose(), dLdOut * out_activation)
    #    dLdW = np.dot(dLdOut.transpose() * out_activation[0].transpose() * (1 - out_activation[0].transpose()), lstm_out)
    #    dLdLSTM = np.dot(dLdOut * out_activation[0] * (1 - out_activation[0]), self.W_out)

    #    dW_ih, dW_hh = [], []
    #    for i, dl in enumerate(dLdLSTM):
    #        acts_t = lstm_activations[:len(lstm_activations)-i]
    #        #dW_iht, dW_hht = self.lstm.backward(acts_t, np.expand_dims(dl, 1))
    #        #dW_iht, dW_hht = self.lstm.backward(acts_t, dl)
    #        dW_iht, dW_hht = self.lstm.backward(acts_t, dl.reshape([1, -1]))
    #        dW_ih.append(dW_iht)
    #        dW_hh.append(dW_hht)
    #    dW_ih = np.mean(dW_ih, 0)
    #    dW_hh = np.mean(dW_hh, 0)

    #    return dLdW, dW_ih, dW_hh

    def backward(self, dLdOut):
        dLdIn = self.f.backward(dLdOut)
        dLdIn = self.out.backward(dLdIn)
        dLdIn = self.lstm.backward(dLdIn)
        return dLdIn
