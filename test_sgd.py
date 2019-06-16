import numpy as np
# from util.util import cosine
from modules.LSTM import LSTM
from modules.lego_model import LegoModel
from modules.sse import SSE
from modules.softmax_cross_entropy import Softmax, SoftmaxCrossEntropy

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


# Define target vectors. If doing weight updates for last time step only,
# target is a 1-D vector. If doing weight updates for each times step output,
# target is a sequence.
if FINAL_ONLY:
    # target for sigmoid output
    #y = np.array([0, 1, 0, 0], dtype=np.float32)#.reshape([output_size, 1])
    # target index for sofmax output
    y = 1
else:
    # if target is a sequence, manually define a target sequence with `seq_len` 5
    y = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)


m = LegoModel(input_size, hidden_size, output_size)
criterion = SoftmaxCrossEntropy()
softmax = Softmax()

for i in range(10000):
    out = m(x)
    loss = criterion(out, y)

    m.zero_grad()
    dLdOut = criterion.backward()
    _ = m.backward(dLdOut)

    m.apply_gradient(lr)

    if FINAL_ONLY:
        print(softmax(out).round(1))
    else:
        print(out[0].round(1))
