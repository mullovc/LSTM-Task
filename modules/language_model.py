import numpy as np
from modules.module import Module
from modules.embedding import Embedding
from modules.LSTM import LSTM
from modules.linear import Linear
# from modules.softmax_cross_entropy import Softmax

class LanguageModel(Module):
    def __init__(self, input_size, embedding_size, lstm_size, output_size):
        super(LanguageModel, self).__init__()
        self.input_size     = input_size
        self.embedding_size = embedding_size
        self.lstm_size      = lstm_size
        self.output_size    = output_size

        self.emb  = Embedding(input_size, embedding_size)
        self.lstm = LSTM(embedding_size, lstm_size)
        self.out  = Linear(lstm_size, output_size)  

        self.h_0, self.c_0 = None, None

    def reset_hidden(self, batch_size):

        self.h_0 = np.random.randn(batch_size, self.lstm_size).astype(np.float32)
        self.c_0 = np.random.randn(batch_size, self.lstm_size).astype(np.float32)


    def forward(self, X):
        seq_len    = X.shape[0]
        batch_size = X.shape[1]

        E = self.emb(X)

        if self.h_0 is None:
            self.reset_hidden()

        lstm_out, _ = self.lstm(E, (self.h_0, self.c_0))

        out = self.out(lstm_out)

        return out

    def backward(self, dLdOut):
        # dLdIn = self.f.backward(dLdOut)
        # dLdIn = self.out.backward(dLdIn)
        dLdIn = self.out.backward(dLdOut)
        dLdIn = self.lstm.backward(dLdIn)
        dLdIn = np.stack(dLdIn, 0)
        self.emb.backward(dLdIn)
