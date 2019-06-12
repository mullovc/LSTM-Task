import numpy as np
# from util.util import cosine
from modules.LSTM import LSTM
from modules.model import Model

EPS=10e-5
lr = 0.02
input_size = 8
hidden_size = 6
output_size = 4
batch_size = 1
seq_len = 5

# determines whether to do weight updates for just the final time step, or each
# time step output
FINAL_ONLY = True

x = np.random.randn(1, seq_len, input_size).astype(np.float32)
h_0 = np.random.randn(hidden_size, batch_size).astype(np.float32)
c_0 = np.random.randn(hidden_size, batch_size).astype(np.float32)


# ignore this (commented out) part
# lstm = LSTM(input_size, hidden_size)
# # gradient descent
# for i in range(250):
#     outs, _, activations = lstm(x, (h_0, c_0))
#     dW_ih, dW_hh = lstm.backward(activations, 1)
#     lstm.lstm_layer.W_ih -= lr * dW_ih
#     lstm.lstm_layer.W_hh -= lr * dW_hh
#     print(outs.mean())



# Define target vectors. If doing weight updates for last time step only,
# target is a 1-D vector. If doing weight updates for each times step output,
# target is a sequence.
if FINAL_ONLY:
    y = np.array([0, 1, 0, 0], dtype=np.float32)#.reshape([output_size, 1])
else:
    # if target is a sequence, manually define a target sequence with `seq_len` 5
    y = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)


m = Model(input_size, hidden_size, output_size)

for i in range(10000):
    outs, activations = m(x)

    if FINAL_ONLY:
        dLdOut = outs[:,-1,:] - y
        dLdWout, dW_ih, dW_hh = m.backward_last_only(activations, dLdOut)
    else:
        dLdOut = outs[0] - y
        dLdWout, dW_ih, dW_hh = m.backward(activations, dLdOut)

    m.W_out -= lr * dLdWout
    m.lstm.lstm_layer.W_ih -= lr * dW_ih
    m.lstm.lstm_layer.W_hh -= lr * dW_hh

    if FINAL_ONLY:
        print(outs[:,-1,:].round(1))
    else:
        print(outs[0].round(1))
